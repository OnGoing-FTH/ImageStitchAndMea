'''
修正roi计算方法
'''
import sys
import cv2
import numpy as np
import time
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QSpinBox,
                             QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
                             QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsRectItem,
                             QGraphicsPolygonItem,
                             QFileDialog, QDoubleSpinBox, QGroupBox, QRadioButton, QMessageBox, QCheckBox)
from PyQt5.QtCore import Qt, pyqtSignal, QPointF, QLineF, QRectF
from PyQt5.QtGui import QPixmap, QImage, QPen, QColor, QBrush, QPainter, QPolygonF

# ==============================================================================
# 1. Measure Utilities (算法核心)
# ==============================================================================

class MeasureUtils:
    @staticmethod
    def get_region_mean_color(image, rect):
        """ 计算图像中指定矩形区域的平均颜色 (B, G, R) """
        x, y, w, h = map(int, rect)
        h_img, w_img = image.shape[:2]
        x = max(0, x); y = max(0, y)
        w = min(w, w_img - x); h = min(h, h_img - y)
        if w <= 0 or h <= 0: return None
        roi = image[y:y+h, x:x+w]
        return np.mean(roi, axis=(0, 1))

    @staticmethod
    def get_color_profile(image, p1, p2, width_px):
        """ 提取 p1 到 p2 的 RGB 剖面 """
        x1, y1 = p1
        x2, y2 = p2
        dx, dy = x2 - x1, y2 - y1
        length = np.hypot(dx, dy)
        num_samples = int(np.ceil(length))
        if num_samples < 2: num_samples = 2
        
        x_coords = np.linspace(x1, x2, num_samples).astype(np.float32)
        y_coords = np.linspace(y1, y2, num_samples).astype(np.float32)
        
        map_x = x_coords.reshape(1, -1)
        map_y = y_coords.reshape(1, -1)
        
        profile = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return profile[0]

    @staticmethod
    def measure_color_edge_adaptive(profile_bgr, bg_color):
        """ 智适应颜色边缘检测 """
        if profile_bgr is None or len(profile_bgr) < 5: return None
        
        # 计算距离曲线
        diff = profile_bgr.astype(np.float32) - bg_color.astype(np.float32)
        dist_curve = np.linalg.norm(diff, axis=1)
        
        # 智适应检查：动态范围过小则忽略
        dynamic_range = np.max(dist_curve) - np.min(dist_curve)
        if dynamic_range < 20.0: return None
            
        # 找最大梯度
        kernel = np.array([0.25, 0.5, 0.25])
        smooth_dist = np.convolve(dist_curve, kernel, mode='same')
        grads = np.gradient(smooth_dist)
        abs_grads = np.abs(grads)
        
        valid_grads = abs_grads[2:-2]
        if len(valid_grads) == 0: return None
        
        best_idx_local = np.argmax(valid_grads)
        best_idx = best_idx_local + 2
        
        # 亚像素插值
        y1 = abs_grads[best_idx-1]
        y2 = abs_grads[best_idx]
        y3 = abs_grads[best_idx+1]
        denom = y1 - 2*y2 + y3
        delta = (y1 - y3) / (2 * denom) if denom != 0 else 0.0
            
        return float(best_idx) + delta

    @staticmethod
    def fit_circle_robust(points, outlier_threshold=2.0):
        """ 鲁棒圆拟合 """
        if points is None or len(points) < 3:
            return {"success": False, "msg": "点数不足3"}
        pts = np.array(points)
        
        def _fit(p):
            x, y = p[:, 0], p[:, 1]
            A = np.column_stack((2*x, 2*y, np.ones_like(x)))
            b = x**2 + y**2
            try:
                res, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                cx, cy = res[0], res[1]
                r = np.sqrt(res[2] + cx**2 + cy**2)
                return (cx, cy), r
            except: return (None, None), None

        (cx, cy), r = _fit(pts)
        if cx is None: return {"success": False, "msg": "拟合失败"}

        dists = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
        valid_pts = pts[np.abs(dists - r) < outlier_threshold]

        if len(valid_pts) < 3: return {"success": False, "msg": "有效点不足"}

        (fcx, fcy), fr = _fit(valid_pts)
        final_dists = np.hypot(valid_pts[:, 0] - fcx, valid_pts[:, 1] - fcy)
        circularity = final_dists.max() - final_dists.min()

        return {
            "success": True, "center": (fcx, fcy), "radius": fr,
            "circularity": float(circularity), "valid_points": valid_pts,
            "outlier_count": len(pts) - len(valid_pts)
        }

    @staticmethod
    def fit_circle_ls(points):
        """ 非鲁棒最小二乘圆拟合（公平对比用） """
        if points is None or len(points) < 3:
            return {"success": False, "msg": "点数不足3"}
        pts = np.array(points)
        x, y = pts[:, 0], pts[:, 1]
        A = np.column_stack((2*x, 2*y, np.ones_like(x)))
        b = x**2 + y**2
        try:
            res, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            cx, cy = res[0], res[1]
            r = np.sqrt(res[2] + cx**2 + cy**2)
            dists = np.hypot(x - cx, y - cy)
            circularity = dists.max() - dists.min()
            return {
                "success": True,
                "center": (cx, cy),
                "radius": r,
                "circularity": float(circularity),
                "valid_points": pts
            }
        except:
            return {"success": False, "msg": "拟合失败"}


    @staticmethod
    def fit_line_ls(points):
        if points is None or len(points) < 2:
            return {"success": False, "msg": "点数不足"}
        pts = np.array(points).astype(np.float32)
        vx, vy, x0, y0 = cv2.fitLine(
            pts, cv2.DIST_L2, 0, 0.01, 0.01
        ).flatten()
        norm = np.hypot(vx, vy)
        ux, uy = vx/norm, vy/norm

        diffs = pts - np.array([x0, y0])
        signed_dists = diffs[:, 0] * uy - diffs[:, 1] * ux
        straightness = signed_dists.max() - signed_dists.min()

        return {
            "success": True,
            "line_point": (x0, y0),
            "line_dir": (ux, uy),
            "straightness": float(straightness),
            "valid_points": pts
        }


    @staticmethod
    def fit_line_robust(points, outlier_threshold=2.0):
        """ 鲁棒直线拟合 """
        if points is None or len(points) < 2:
            return {"success": False, "msg": "点数不足"}
        pts = np.array(points)

        def _fit(p):
            vx, vy, x0, y0 = cv2.fitLine(p.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).flatten()
            return (x0, y0), (vx, vy)

        p0, d = _fit(pts)
        vx, vy = d
        norm = np.hypot(vx, vy)
        ux, uy = vx/norm, vy/norm # 单位方向
        
        # 计算距离
        diffs = pts - np.array(p0)
        dists = np.abs(diffs[:, 0] * uy - diffs[:, 1] * ux)
        
        valid_pts = pts[dists < outlier_threshold]
        
        if len(valid_pts) < 2: return {"success": False, "msg": "有效点不足"}
        
        p_final, d_final = _fit(valid_pts)
        vf_x, vf_y = d_final
        n_f = np.hypot(vf_x, vf_y)
        uf_x, uf_y = vf_x/n_f, vf_y/n_f
        
        diffs_f = valid_pts - np.array(p_final)
        signed_dists = diffs_f[:, 0] * uf_y - diffs_f[:, 1] * uf_x
        straightness = np.max(signed_dists) - np.min(signed_dists)
        
        return {
            "success": True, "line_point": p_final, "line_dir": (uf_x, uf_y),
            "straightness": straightness, "valid_points": valid_pts,
            "outlier_count": len(pts) - len(valid_pts)
        }

# ==============================================================================
# 2. Interactive View (修改：加入箭头绘制，传递完整坐标)
# ==============================================================================

class InteractiveView(QGraphicsView):
    point_selected = pyqtSignal(float, float)
    direction_defined = pyqtSignal(object) # 此时传递 (p1, p2)
    bg_sampled = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHint(QPainter.Antialiasing)
        self.setMouseTracking(True)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.mode = "VIEW"
        self.temp_items = []
        self.result_items = []
        self._drag_start = None
        self._drag_item = None
        self._drag_arrow_head = None # 新增：箭头头部

    def load_image(self, cv_img):
        self._scene.clear()
        self.temp_items = []; self.result_items = []
        h, w = cv_img.shape[:2]
        q_img = QImage(cv_img.data, w, h, 3*w, QImage.Format_RGB888).rgbSwapped()
        self._scene.addItem(QGraphicsPixmapItem(QPixmap.fromImage(q_img)))

    def clear_overlays(self):
        for it in self.temp_items: self._scene.removeItem(it)
        self.temp_items = []

    def clear_results(self):
        for it in self.result_items: self._scene.removeItem(it)
        self.result_items = []

    def draw_results(self, valid_points, shape_params, roi_info, outliers=None):
        # ... (保持原样，通过精心构造 roi_info 即可复用) ...
        if roi_info:
            if len(roi_info) == 3 and not isinstance(roi_info[0], str):
                # 圆形 ROI
                (cx, cy), r_in, r_out = roi_info
                pen_roi = QPen(Qt.yellow, 1, Qt.DashLine)
                roi1 = QGraphicsEllipseItem(cx-r_in, cy-r_in, r_in*2, r_in*2); roi1.setPen(pen_roi)
                roi2 = QGraphicsEllipseItem(cx-r_out, cy-r_out, r_out*2, r_out*2); roi2.setPen(pen_roi)
                self._scene.addItem(roi1); self._scene.addItem(roi2)
                self.result_items.extend([roi1, roi2])
            elif len(roi_info) == 5 and roi_info[0] == "line":
                # 直线 ROI (矩形)
                _, p_s, p_e, w, nv = roi_info
                offset = nv * (w / 2.0)
                c1 = p_s - offset; c2 = p_s + offset
                c3 = p_e + offset; c4 = p_e - offset
                poly = QPolygonF([QPointF(c1[0],c1[1]), QPointF(c2[0],c2[1]), 
                                  QPointF(c3[0],c3[1]), QPointF(c4[0],c4[1])])
                roi_item = QGraphicsPolygonItem(poly)
                roi_item.setPen(QPen(Qt.yellow, 1, Qt.DashLine))
                self._scene.addItem(roi_item)
                self.result_items.append(roi_item)

        def draw_cross(pts, color):
            pen = QPen(color, 1)
            for pt in pts:
                px, py = pt
                s = 3
                l1 = QGraphicsLineItem(px-s, py-s, px+s, py+s); l1.setPen(pen)
                l2 = QGraphicsLineItem(px-s, py+s, px+s, py-s); l2.setPen(pen)
                self._scene.addItem(l1); self._scene.addItem(l2)
                self.result_items.extend([l1, l2])

        draw_cross(valid_points, Qt.red)
        if outliers:
            draw_cross(outliers, Qt.blue)

        if shape_params:
            if len(shape_params) == 2 and isinstance(shape_params[1], (float, int, np.float32, np.float64)):
                (cx, cy), r = shape_params
                c = QGraphicsEllipseItem(cx-r, cy-r, r*2, r*2)
                c.setPen(QPen(Qt.cyan, 2))
                self._scene.addItem(c)
                self.result_items.append(c)
                l1 = QGraphicsLineItem(cx-10, cy, cx+10, cy); l1.setPen(QPen(Qt.red, 2))
                l2 = QGraphicsLineItem(cx, cy-10, cx, cy+10); l2.setPen(QPen(Qt.red, 2))
                self._scene.addItem(l1); self._scene.addItem(l2)
                self.result_items.extend([l1, l2])
            else:
                (x0, y0), (vx, vy) = shape_params
                L = 3000
                p1 = (x0 - vx*L, y0 - vy*L)
                p2 = (x0 + vx*L, y0 + vy*L)
                li = QGraphicsLineItem(p1[0], p1[1], p2[0], p2[1])
                li.setPen(QPen(Qt.cyan, 2))
                self._scene.addItem(li)
                self.result_items.append(li)

    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 1.0/1.15
        self.scale(factor, factor)

    def mousePressEvent(self, event):
        sp = self.mapToScene(event.pos())
        if event.button() == Qt.LeftButton:
            if self.mode == "PICK":
                self.temp_items.append(self._scene.addEllipse(sp.x()-3, sp.y()-3, 6, 6, QPen(Qt.yellow), QBrush(Qt.yellow)))
                self.point_selected.emit(sp.x(), sp.y())
                return
            elif self.mode == "DIRECTION":
                self._drag_start = sp
                self._drag_item = self._scene.addLine(sp.x(), sp.y(), sp.x(), sp.y(), QPen(Qt.magenta, 2))
                # 创建箭头头部
                self._drag_arrow_head = QGraphicsPolygonItem()
                self._drag_arrow_head.setPen(QPen(Qt.magenta, 2))
                self._drag_arrow_head.setBrush(QBrush(Qt.magenta))
                self._scene.addItem(self._drag_arrow_head)
                return
            elif self.mode == "SAMPLE_BG":
                self._drag_start = sp
                self._drag_item = self._scene.addRect(sp.x(), sp.y(), 0, 0, QPen(Qt.green, 2))
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_start:
            sp = self.mapToScene(event.pos())
            if self.mode == "DIRECTION":
                line = QLineF(self._drag_start, sp)
                self._drag_item.setLine(line)
                
                # 更新箭头几何形状
                if line.length() > 0:
                    angle = np.arctan2(sp.y() - self._drag_start.y(), sp.x() - self._drag_start.x())
                    arrow_size = 10
                    p_end = QPointF(sp.x(), sp.y())
                    p1 = QPointF(p_end.x() - arrow_size * np.cos(angle - np.pi/6), 
                                 p_end.y() - arrow_size * np.sin(angle - np.pi/6))
                    p2 = QPointF(p_end.x() - arrow_size * np.cos(angle + np.pi/6), 
                                 p_end.y() - arrow_size * np.sin(angle + np.pi/6))
                    self._drag_arrow_head.setPolygon(QPolygonF([p_end, p1, p2]))
                return
            elif self.mode == "SAMPLE_BG":
                rect = QRectF(self._drag_start, sp).normalized()
                self._drag_item.setRect(rect)
                return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._drag_start:
            sp = self.mapToScene(event.pos())
            if self.mode == "DIRECTION":
                p1 = (self._drag_start.x(), self._drag_start.y())
                p2 = (sp.x(), sp.y())
                length = np.hypot(p2[0]-p1[0], p2[1]-p1[1])
                
                if length > 5:
                    # 此时传递点坐标，而非向量和长度
                    self.direction_defined.emit((p1, p2))
                    self.temp_items.append(self._drag_item)
                    self.temp_items.append(self._drag_arrow_head)
                else:
                    self._scene.removeItem(self._drag_item)
                    self._scene.removeItem(self._drag_arrow_head)
                self._drag_arrow_head = None
                
            elif self.mode == "SAMPLE_BG":
                rect = QRectF(self._drag_start, sp).normalized()
                if rect.width() > 2 and rect.height() > 2:
                    self.bg_sampled.emit((rect.x(), rect.y(), rect.width(), rect.height()))
                    self._drag_item.setPen(QPen(Qt.green, 1, Qt.DashLine))
                    self.temp_items.append(self._drag_item)
                else:
                    self._scene.removeItem(self._drag_item)
            self._drag_start = None
            self._drag_item = None
            return
        super().mouseReleaseEvent(event)

# ==============================================================================
# 3. Main Window (修改：ROI 计算逻辑)
# ==============================================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("智能边缘测量")
        self.resize(1300, 850)
        
        self.cv_image = None
        self.points = []
        self.bg_color = None
        self.scan_config = {"roi_points": None} # 修改配置结构
        
        self.setup_ui()

    def setup_ui(self):
        main = QWidget(); self.setCentralWidget(main)
        h = QHBoxLayout(main)
        panel = QWidget(); panel.setFixedWidth(340)
        vbox = QVBoxLayout(panel)

        vbox.addWidget(QLabel("<b>1. 图像 / Image</b>"))
        btn_load = QPushButton("打开图片"); btn_load.clicked.connect(self.load_image)
        vbox.addWidget(btn_load)

        vbox.addSpacing(10)
        vbox.addWidget(QLabel("<b>2. 模式 / Mode</b>"))
        bg_layout = QHBoxLayout()
        self.rb_circle = QRadioButton("圆形测量"); self.rb_circle.setChecked(True)
        self.rb_line = QRadioButton("直线测量")
        self.rb_circle.toggled.connect(self.reset_state)
        self.rb_line.toggled.connect(self.reset_state)
        bg_layout.addWidget(self.rb_circle); bg_layout.addWidget(self.rb_line)
        vbox.addLayout(bg_layout)

        vbox.addSpacing(10)
        vbox.addWidget(QLabel("<b>3. 背景采样 / Background</b>"))
        self.btn_bg = QPushButton("框选背景 (绿色背光)")
        self.btn_bg.clicked.connect(self.start_sample_bg)
        vbox.addWidget(self.btn_bg)
        self.lbl_bg_status = QLabel("未采样")
        self.lbl_bg_status.setStyleSheet("background-color: #eee; border:1px solid gray; padding:2px")
        vbox.addWidget(self.lbl_bg_status)

        vbox.addSpacing(10)
        grp = QGroupBox("参数")
        gl = QVBoxLayout(grp)
        row_step = QHBoxLayout(); row_step.addWidget(QLabel("步长(°或px):"))
        self.spin_step = QDoubleSpinBox(); self.spin_step.setRange(0.1, 50.0); self.spin_step.setValue(1.0)
        row_step.addWidget(self.spin_step); gl.addLayout(row_step)
        row_out = QHBoxLayout(); row_out.addWidget(QLabel("剔除阈值(px):"))
        self.spin_outlier = QDoubleSpinBox(); self.spin_outlier.setRange(0.1, 100.0); self.spin_outlier.setValue(3.0)
        row_out.addWidget(self.spin_outlier); gl.addLayout(row_out)
        row_scale = QHBoxLayout(); row_scale.addWidget(QLabel("比例 (mm/px):"))
        self.spin_scale = QDoubleSpinBox(); self.spin_scale.setDecimals(6); self.spin_scale.setValue(0.0100)
        row_scale.addWidget(self.spin_scale); gl.addLayout(row_scale)
        vbox.addWidget(grp)

        vbox.addSpacing(10)
        vbox.addWidget(QLabel("<b>4. 流程 / Flow</b>"))
        self.btn_pick = QPushButton("Step A: 选点")
        self.btn_pick.clicked.connect(self.start_pick_points)
        vbox.addWidget(self.btn_pick)
        self.lbl_hint_pick = QLabel("未开始")
        vbox.addWidget(self.lbl_hint_pick)

        self.btn_dir = QPushButton("Step B: 绘制扫描宽度")
        self.btn_dir.setEnabled(False)
        self.btn_dir.clicked.connect(self.start_draw_direction)
        vbox.addWidget(self.btn_dir)
        self.lbl_hint_dir = QLabel("需先选点")
        vbox.addWidget(self.lbl_hint_dir)

        self.btn_calc = QPushButton("Step C: 运行计算")
        self.btn_calc.setEnabled(False)
        self.btn_calc.setStyleSheet("background-color:#d0f0c0; font-weight:bold; height:40px")
        self.btn_calc.clicked.connect(self.run_measurement)
        vbox.addWidget(self.btn_calc)

        vbox.addSpacing(6)
        self.lbl_result = QLabel("结果显示区")
        self.lbl_result.setFrameShape(QLabel.Panel)
        self.lbl_result.setWordWrap(True)
        vbox.addWidget(self.lbl_result)
        vbox.addStretch()
        h.addWidget(panel)

        self.view = InteractiveView()
        self.view.point_selected.connect(self.on_point_picked)
        self.view.direction_defined.connect(self.on_direction_set)
        self.view.bg_sampled.connect(self.on_bg_sampled)
        h.addWidget(self.view)

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.bmp *.tif)")
        if not path: return
        self.cv_image = cv2.imread(path)
        if self.cv_image is None: return
        self.view.load_image(self.cv_image)
        self.reset_state()
        self.lbl_bg_status.setText("请框选背景颜色")

    def reset_state(self):
        self.points = []
        self.bg_color = None
        self.scan_config = {"roi_points": None}
        self.view.clear_overlays(); self.view.clear_results()
        self.view.mode = "VIEW"
        self.btn_dir.setEnabled(False); self.btn_calc.setEnabled(False)
        self.lbl_hint_pick.setText("等待选点...")
        self.lbl_result.setText("")

    def start_sample_bg(self):
        if self.cv_image is None: return
        self.view.mode = "SAMPLE_BG"
        QMessageBox.information(self, "提示", "请在图像【背景区域】拉一个矩形框。")

    def on_bg_sampled(self, rect):
        mean_c = MeasureUtils.get_region_mean_color(self.cv_image, rect)
        if mean_c is not None:
            self.bg_color = mean_c
            b, g, r = int(mean_c[0]), int(mean_c[1]), int(mean_c[2])
            self.lbl_bg_status.setText(f"R:{r} G:{g} B:{b}")
            self.lbl_bg_status.setStyleSheet(f"background-color: rgb({r},{g},{b}); color: {'white' if g<128 else 'black'}")
            self.view.mode = "VIEW"
        else:
            self.lbl_bg_status.setText("采样无效")

    def start_pick_points(self):
        if self.cv_image is None: return
        self.points = []
        self.view.clear_overlays(); self.view.clear_results()
        self.view.mode = "PICK"
        target = 3 if self.rb_circle.isChecked() else 2
        self.lbl_hint_pick.setText(f"请点击 {target} 个点...")

    def on_point_picked(self, x, y):
        self.points.append((x, y))
        target = 3 if self.rb_circle.isChecked() else 2
        self.lbl_hint_pick.setText(f"已选 {len(self.points)} / {target}")
        
        if len(self.points) >= target:
            self.view.mode = "VIEW"
            self.lbl_hint_pick.setText("选点完成")
            self.btn_dir.setEnabled(True)
            self.lbl_hint_dir.setText("请绘制扫描宽度")
            if self.rb_circle.isChecked():
                res = MeasureUtils.fit_circle_robust(self.points)
                if res["success"]:
                    (cx, cy), r = res["center"], res["radius"]
                    itm = QGraphicsEllipseItem(cx-r, cy-r, r*2, r*2)
                    itm.setPen(QPen(Qt.yellow, 1, Qt.DashLine))
                    self.view._scene.addItem(itm); self.view.temp_items.append(itm)
            else:
                p1, p2 = self.points[0], self.points[-1]
                itm = QGraphicsLineItem(p1[0], p1[1], p2[0], p2[1])
                itm.setPen(QPen(Qt.yellow, 1, Qt.DashLine))
                self.view._scene.addItem(itm); self.view.temp_items.append(itm)

    def start_draw_direction(self):
        self.view.mode = "DIRECTION"
        QMessageBox.information(self, "提示", "请绘制一个箭头定义扫描区域。\n箭头的起点和终点将直接决定扫描的【范围】和【方向】。")

    def on_direction_set(self, points):
        # points 是 (p1, p2)
        p1, p2 = points
        length = np.hypot(p1[0]-p2[0], p1[1]-p2[1])
        
        self.scan_config["roi_points"] = points
        self.view.mode = "VIEW"
        self.lbl_hint_dir.setText(f"范围已定 ({length:.1f}px)")
        if self.bg_color is not None:
            self.btn_calc.setEnabled(True)
        else:
            QMessageBox.warning(self, "警告", "请先完成 Step 2: 采样背景颜色")

    def run_measurement(self):
        if self.bg_color is None: return
        self.view.clear_results()
        if self.rb_circle.isChecked():
            self.measure_circle_color()
        else:
            self.measure_line_color()

    def measure_circle_color(self):
        res_base = MeasureUtils.fit_circle_robust(self.points)
        if not res_base["success"]: return
        (cx, cy), r_base = res_base["center"], res_base["radius"]
        
        # === 修改点：基于起终点距离直接确定内外径 ===
        p_scan_start, p_scan_end = self.scan_config["roi_points"]
        
        # 计算扫描起点和终点相对于圆心的距离
        r_start_scan = np.hypot(p_scan_start[0] - cx, p_scan_start[1] - cy)
        r_end_scan = np.hypot(p_scan_end[0] - cx, p_scan_end[1] - cy)
        
        # 扫描宽度即为两径之差
        scan_width = abs(r_end_scan - r_start_scan)
        if scan_width < 1.0: scan_width = 1.0
        
        angle_step = self.spin_step.value()
        
        detected_pts = []
        num_spokes = int(360.0 / angle_step)
        if num_spokes < 10: num_spokes = 10
        
        for i in range(num_spokes):
            angle_deg = i * angle_step
            ang = np.deg2rad(angle_deg)
            ca, sa = np.cos(ang), np.sin(ang)
            
            # 直接使用定义的半径进行扫描，方向隐含在 start -> end 中
            p_s = (cx + r_start_scan*ca, cy + r_start_scan*sa)
            p_e = (cx + r_end_scan*ca, cy + r_end_scan*sa)
            
            prof_bgr = MeasureUtils.get_color_profile(self.cv_image, p_s, p_e, scan_width)
            idx = MeasureUtils.measure_color_edge_adaptive(prof_bgr, self.bg_color)
            if idx is not None:
                ratio = idx / float(len(prof_bgr)-1)
                px = p_s[0] + (p_e[0]-p_s[0]) * ratio
                py = p_s[1] + (p_e[1]-p_s[1]) * ratio
                detected_pts.append((px, py))

        if len(detected_pts) < 10:
            self.lbl_result.setText("点数不足")
            return

        final_res = MeasureUtils.fit_circle_robust(detected_pts, self.spin_outlier.value())
        
        # 构造用于显示的 roi_info
        # 为了兼容 draw_results 的显示，传入 min 和 max 半径
        r_min = min(r_start_scan, r_end_scan)
        r_max = max(r_start_scan, r_end_scan)
        roi_info = ((cx, cy), r_min, r_max)
        
        if final_res["success"]:
            valid_pts = final_res["valid_points"]
            valid_set = set((float(p[0]), float(p[1])) for p in valid_pts)
            valid_list = []; outlier_list = []
            
            for pt in detected_pts:
                if (float(pt[0]), float(pt[1])) in valid_set:
                    valid_list.append(pt)
                else:
                    outlier_list.append(pt)
            
            circle_par = (final_res["center"], final_res["radius"])
            self.view.draw_results(valid_list, circle_par, roi_info, outliers=outlier_list)
            
            scale = self.spin_scale.value()
            phys_d = final_res["radius"] * 2.0 * scale
            phys_circ = final_res["circularity"] * scale
            
            self.lbl_result.setText(
                f"【测量成功】\n"
                f"总检测点: {len(detected_pts)}\n"
                f"有效点数: {len(valid_list)} (红色)\n"
                f"离群点数: {len(outlier_list)} (蓝色)\n"
                f"物理直径: {phys_d:.4f} mm\n"
                f"圆度误差: {phys_circ:.4f} mm\n"
                f"圆心: ({final_res['center'][0]:.1f}, {final_res['center'][1]:.1f})"
            )
        else:
            self.lbl_result.setText(f"拟合失败: {final_res.get('msg')}")

    def measure_line_color(self):
        if len(self.points) < 2: return
        p_start_base = np.array(self.points[0]) # Step A 的起点
        p_end_base = np.array(self.points[-1])   # Step A 的终点
        
        # 标称直线的方向和法向
        vec = p_end_base - p_start_base
        length = np.linalg.norm(vec)
        if length < 1: return
        unit_v = vec / length
        norm_v = np.array([-unit_v[1], unit_v[0]]) # 法向量
        
        # === 修改点：基于起终点投影确定扫描范围 ===
        p_scan_start, p_scan_end = self.scan_config["roi_points"]
        p_scan_start_arr = np.array(p_scan_start)
        p_scan_end_arr = np.array(p_scan_end)
        
        # 计算 Step B 的起点和终点相对于 Step A 起点在法线方向上的投影距离
        # offset = (P - P_base) dot Norm
        offset_s = np.dot(p_scan_start_arr - p_start_base, norm_v)
        offset_e = np.dot(p_scan_end_arr - p_start_base, norm_v)
        
        # 扫描宽度
        scan_width = abs(offset_e - offset_s)
        
        pixel_step = self.spin_step.value()
        if pixel_step < 0.5: pixel_step = 0.5
        
        detected_pts = []
        num_steps = int(length / pixel_step)
        
        for i in range(num_steps + 1):
            current_dist = i * pixel_step
            center_k = p_start_base + unit_v * current_dist # 标称直线上的当前点
            
            # 基于投影偏移量确定的起点和终点
            p_s = center_k + norm_v * offset_s
            p_e = center_k + norm_v * offset_e
            
            prof_bgr = MeasureUtils.get_color_profile(self.cv_image, p_s, p_e, scan_width)
            idx = MeasureUtils.measure_color_edge_adaptive(prof_bgr, self.bg_color)
            if idx is not None:
                ratio = idx / float(len(prof_bgr)-1)
                px = p_s[0] + (p_e[0]-p_s[0]) * ratio
                py = p_s[1] + (p_e[1]-p_s[1]) * ratio
                detected_pts.append((px, py))

        if len(detected_pts) < 2:
             self.lbl_result.setText("点数不足")
             return

        final_res = MeasureUtils.fit_line_robust(detected_pts, self.spin_outlier.value())
        
        # === 构造 roi_info 以适配 draw_results ===
        # draw_results 逻辑假定对称宽度：offset = nv * (w/2)。
        # 我们需要构造一个假的“中心线”和“宽度”，使得视觉上的方框与我们实际扫描区域一致。
        # 实际范围是 [offset_s, offset_e] (相对于 base line)
        mid_offset = (offset_s + offset_e) / 2.0
        # 偏移后的中心线端点
        visual_p_s = p_start_base + norm_v * mid_offset
        visual_p_e = p_end_base + norm_v * mid_offset
        # 传入的总宽度
        visual_width = abs(offset_e - offset_s)
        
        roi_info = ("line", visual_p_s, visual_p_e, visual_width, norm_v)
        
        if final_res["success"]:
            valid_pts = final_res["valid_points"]
            valid_set = set((float(p[0]), float(p[1])) for p in valid_pts)
            valid_list = []; outlier_list = []
            for pt in detected_pts:
                if (float(pt[0]), float(pt[1])) in valid_set: valid_list.append(pt)
                else: outlier_list.append(pt)
            
            line_res = (final_res["line_point"], final_res["line_dir"])
            self.view.draw_results(valid_list, line_res, roi_info, outliers=outlier_list)
            
            scale = self.spin_scale.value()
            phys_s = final_res["straightness"] * scale
            
            # 计算长度
            vx, vy = final_res["line_dir"]
            projections = valid_pts[:, 0] * vx + valid_pts[:, 1] * vy
            pixel_len = np.max(projections) - np.min(projections)
            phys_len = pixel_len * scale

            self.lbl_result.setText(
                f"【直线测量成功】\n"
                f"总检测点: {len(detected_pts)}\n"
                f"有效点数: {len(valid_list)} (红色)\n"
                f"离群点数: {len(outlier_list)} (蓝色)\n"
                f"物理长度: {phys_len:.4f} mm\n"
                f"直线度: {phys_s:.4f} mm"
            )
        else:
            self.lbl_result.setText(f"拟合失败: {final_res.get('msg')}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())