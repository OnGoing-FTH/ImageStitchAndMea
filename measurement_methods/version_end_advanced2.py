'''
圆的检测已经达到比较好的效果了
'''

import sys
import cv2
import numpy as np
import time
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QSpinBox, QGraphicsPolygonItem,
                             QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
                             QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsRectItem,
                             QFileDialog, QDoubleSpinBox, QGroupBox, QRadioButton, QMessageBox)
from PyQt5.QtCore import Qt, pyqtSignal, QPointF, QLineF, QRectF
from PyQt5.QtGui import QPixmap, QImage, QPen, QColor, QBrush, QPolygonF, QPainter

# ==============================================================================
# 1. Measure Utilities (算法核心：颜色距离 + 鲁棒拟合)
# ==============================================================================

class MeasureUtils:
    @staticmethod
    def get_region_mean_color(image, rect):
        """ 计算图像中指定矩形区域的平均颜色 (B, G, R) """
        x, y, w, h = map(int, rect)
        h_img, w_img = image.shape[:2]
        # 边界保护
        x = max(0, x); y = max(0, y)
        w = min(w, w_img - x); h = min(h, h_img - y)
        
        if w <= 0 or h <= 0: return None
        
        roi = image[y:y+h, x:x+w]
        # 计算均值
        mean_bgr = np.mean(roi, axis=(0, 1))
        return mean_bgr 

    @staticmethod
    def get_color_profile(image, p1, p2, width_px):
        """
        提取 p1 到 p2 的 RGB 剖面
        返回: (N, 3) 的 numpy 数组
        """
        x1, y1 = p1
        x2, y2 = p2
        dx, dy = x2 - x1, y2 - y1
        length = np.hypot(dx, dy)
        
        # 采样点数 = 长度 (保证分辨率)
        num_samples = int(np.ceil(length))
        if num_samples < 2: num_samples = 2
        
        x_coords = np.linspace(x1, x2, num_samples).astype(np.float32)
        y_coords = np.linspace(y1, y2, num_samples).astype(np.float32)
        
        map_x = x_coords.reshape(1, -1)
        map_y = y_coords.reshape(1, -1)
        
        # 使用 remap 提取，支持多通道
        # 结果 shape: (1, N, 3) -> squeeze -> (N, 3)
        profile = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return profile[0]

    @staticmethod
    def measure_color_edge_adaptive(profile_bgr, bg_color):
        """
        智适应颜色边缘检测
        1. 计算每个点到背景色的欧氏距离
        2. 寻找距离变化率(梯度)最大的点
        3. 检查动态范围以过滤无效扫描
        """
        if profile_bgr is None or len(profile_bgr) < 5: return None
        
        # 1. 计算距离曲线 Distance Curve
        # profile: (N, 3), bg: (3,)
        diff = profile_bgr.astype(np.float32) - bg_color.astype(np.float32)
        dist_curve = np.linalg.norm(diff, axis=1) # (N,)
        
        # 2. 智适应检查：如果整条线的颜色差异很小（全是背景或全是工件），则放弃
        dynamic_range = np.max(dist_curve) - np.min(dist_curve)
        # 阈值 20 是经验值，表示RGB空间内的显著差异
        if dynamic_range < 20.0:
            return None
            
        # 3. 计算梯度
        # 平滑一下减少噪点
        kernel = np.array([0.25, 0.5, 0.25])
        smooth_dist = np.convolve(dist_curve, kernel, mode='same')
        grads = np.gradient(smooth_dist)
        abs_grads = np.abs(grads)
        
        # 4. 找最大梯度点
        # 去掉头尾各2个像素防止边界效应
        valid_grads = abs_grads[2:-2]
        if len(valid_grads) == 0: return None
        
        # 注意：argmax 返回的是 valid_grads 的索引，需要加回偏移量 2
        best_idx_local = np.argmax(valid_grads)
        best_idx = best_idx_local + 2
        
        # 5. 亚像素插值
        y1 = abs_grads[best_idx-1]
        y2 = abs_grads[best_idx]
        y3 = abs_grads[best_idx+1]
        
        denom = y1 - 2*y2 + y3
        delta = 0.0
        if denom != 0:
            delta = (y1 - y3) / (2 * denom)
            
        return float(best_idx) + delta

    @staticmethod
    def fit_circle_robust(points, outlier_threshold=2.0):
        """
        鲁棒圆拟合：拟合 -> 剔除 -> 重拟合 -> 计算统计量
        """
        if points is None or len(points) < 3:
            return {"success": False, "msg": "点数不足3"}
        
        pts = np.array(points)
        
        # 内部函数：最小二乘圆拟合
        def _fit(p):
            x, y = p[:, 0], p[:, 1]
            A = np.column_stack((2*x, 2*y, np.ones_like(x)))
            b = x**2 + y**2
            try:
                res, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                cx, cy = res[0], res[1]
                r = np.sqrt(res[2] + cx**2 + cy**2)
                return (cx, cy), r
            except:
                return (None, None), None

        # 1. 初拟合
        (cx, cy), r = _fit(pts)
        if cx is None:
            return {"success": False, "msg": "初拟合失败"}

        # 2. 离群点剔除
        dists = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
        errors = np.abs(dists - r)
        valid_mask = errors < outlier_threshold
        valid_pts = pts[valid_mask]

        if len(valid_pts) < 3:
            return {"success": False, "msg": "剔除后有效点不足"}

        # 3. 重拟合
        (fcx, fcy), fr = _fit(valid_pts)
        
        # 4. 计算圆度 (最大半径 - 最小半径)
        final_dists = np.hypot(valid_pts[:, 0] - fcx, valid_pts[:, 1] - fcy)
        circularity = final_dists.max() - final_dists.min()

        return {
            "success": True,
            "center": (fcx, fcy),
            "radius": fr,
            "circularity": float(circularity),
            "valid_points": valid_pts,
            "outlier_count": len(pts) - len(valid_pts)
        }
    
    @staticmethod
    def fit_line_robust(points, outlier_threshold=2.0):
        """
        鲁棒直线拟合
        返回: {success, line_point, line_dir, straightness, valid_points, outlier_count}
        """
        if points is None or len(points) < 2:
            return {"success": False, "msg": "点数不足"}
        
        pts = np.array(points)

        # 内部拟合函数
        def _fit(p):
            # distType=DIST_L2, param=0, reps=0.01, aeps=0.01
            vx, vy, x0, y0 = cv2.fitLine(p.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).flatten()
            return (x0, y0), (vx, vy)

        # 1. 初拟合
        p0, d = _fit(pts)
        vx, vy = d
        
        # 2. 计算点到直线的距离
        # 距离公式 d = |(P - P0) x u| (二维叉乘模长)
        # 归一化向量
        norm = np.hypot(vx, vy)
        ux, uy = vx/norm, vy/norm
        
        diffs = pts - np.array(p0)
        # 二维叉乘: x1*y2 - x2*y1
        # 这里 u 是方向向量，diff 是 P-P0
        # dist = diff_x * uy - diff_y * ux (注意正负号，取绝对值)
        dists = np.abs(diffs[:, 0] * uy - diffs[:, 1] * ux)
        
        # 3. 过滤
        valid_mask = dists < outlier_threshold
        valid_pts = pts[valid_mask]
        
        if len(valid_pts) < 2:
            return {"success": False, "msg": "剔除后有效点不足"}
            
        # 4. 重拟合
        p_final, d_final = _fit(valid_pts)
        
        # 5. 计算直线度 (有效点到直线的最大距离 - 最小距离，或者单纯最大偏差)
        # 这里计算 range
        vf_x, vf_y = d_final
        n_f = np.hypot(vf_x, vf_y)
        uf_x, uf_y = vf_x/n_f, vf_y/n_f
        
        diffs_f = valid_pts - np.array(p_final)
        # 有符号距离
        signed_dists = diffs_f[:, 0] * uf_y - diffs_f[:, 1] * uf_x
        straightness = np.max(signed_dists) - np.min(signed_dists)
        
        return {
            "success": True,
            "line_point": p_final,
            "line_dir": (uf_x, uf_y),
            "straightness": straightness,
            "valid_points": valid_pts,
            "outlier_count": len(pts) - len(valid_pts)
        }

# ==============================================================================
# 2. Interactive View (交互画板：支持选点、画线、采样背景、结果绘制)
# ==============================================================================

class InteractiveView(QGraphicsView):
    point_selected = pyqtSignal(float, float)
    direction_defined = pyqtSignal(float, object)
    bg_sampled = pyqtSignal(object) # rect (x,y,w,h)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHint(QPainter.Antialiasing)
        self.setMouseTracking(True)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        
        self.mode = "VIEW"  # VIEW, PICK, DIRECTION, SAMPLE_BG
        self.temp_items = []
        self.result_items = []
        
        self._drag_start = None
        self._drag_item = None

    def load_image(self, cv_img):
        self._scene.clear()
        self.temp_items = []
        self.result_items = []
        h, w, c = cv_img.shape
        # 显示彩色图
        q_img = QImage(cv_img.data, w, h, 3*w, QImage.Format_RGB888).rgbSwapped()
        self._pixmap_item = QGraphicsPixmapItem(QPixmap.fromImage(q_img))
        self._scene.addItem(self._pixmap_item)

    def clear_overlays(self):
        for it in self.temp_items:
            try: self._scene.removeItem(it)
            except: pass
        self.temp_items = []

    def clear_results(self):
        for it in self.result_items:
            try: self._scene.removeItem(it)
            except: pass
        self.result_items = []

    def draw_results(self, valid_points, shape_params, roi_info, outliers=None):
        """
        绘制结果
        :param valid_points: 有效点列表 (红叉)
        :param circle_params: 拟合圆参数
        :param roi_info: ROI信息
        :param outliers: 离群点列表 (蓝叉) [新增参数]
        """
        # 1. 绘制 ROI 范围 (黄色虚线)
        if roi_info:
            if len(roi_info) == 3 and not isinstance(roi_info[0], str):
                # --- 圆形 ROI ---
                (cx, cy), r_in, r_out = roi_info
                pen_roi = QPen(Qt.yellow, 1, Qt.DashLine)
                roi1 = QGraphicsEllipseItem(cx-r_in, cy-r_in, r_in*2, r_in*2); roi1.setPen(pen_roi)
                roi2 = QGraphicsEllipseItem(cx-r_out, cy-r_out, r_out*2, r_out*2); roi2.setPen(pen_roi)
                self._scene.addItem(roi1); self._scene.addItem(roi2)
                self.result_items.extend([roi1, roi2])
            elif len(roi_info) == 5 and roi_info[0] == "line":
                # --- 直线 ROI (矩形框) ---
                _, p_s, p_e, w, nv = roi_info
                # 构造矩形的四个角点
                # p_s +/- w/2 * nv
                # p_e +/- w/2 * nv
                offset = nv * (w / 2.0)
                c1 = p_s - offset
                c2 = p_s + offset
                c3 = p_e + offset
                c4 = p_e - offset
                
                poly = QPolygonF([QPointF(c1[0],c1[1]), QPointF(c2[0],c2[1]), 
                                  QPointF(c3[0],c3[1]), QPointF(c4[0],c4[1])])
                
                roi_item = QGraphicsPolygonItem(poly)
                roi_item.setPen(QPen(Qt.yellow, 1, Qt.DashLine))
                self._scene.addItem(roi_item)
                self.result_items.append(roi_item)

        # 2. 绘制有效点 (红色小叉号)
        pen_valid = QPen(Qt.red, 1)
        for pt in valid_points:
            px, py = pt
            s = 2
            l1 = QGraphicsLineItem(px-s, py-s, px+s, py+s); l1.setPen(pen_valid)
            l2 = QGraphicsLineItem(px-s, py+s, px+s, py-s); l2.setPen(pen_valid)
            self._scene.addItem(l1); self._scene.addItem(l2)
            self.result_items.extend([l1, l2])

        # 3. 绘制离群点 (蓝色小叉号) -> 新增部分
        if outliers:
            pen_out = QPen(Qt.blue, 1)
            for pt in outliers:
                px, py = pt
                s = 2
                l1 = QGraphicsLineItem(px-s, py-s, px+s, py+s); l1.setPen(pen_out)
                l2 = QGraphicsLineItem(px-s, py+s, px+s, py-s); l2.setPen(pen_out)
                self._scene.addItem(l1); self._scene.addItem(l2)
                self.result_items.extend([l1, l2])

        # 4. 绘制拟合圆 (青色)
        # 3. 绘制拟合结果 (圆或线)
        if shape_params:
            if len(shape_params) == 2 and isinstance(shape_params[1], (float, int, np.float32, np.float64)):
                # --- 圆 ---
                (cx, cy), r = shape_params
                c = QGraphicsEllipseItem(cx-r, cy-r, r*2, r*2)
                c.setPen(QPen(Qt.cyan, 2))
                self._scene.addItem(c)
                self.result_items.append(c)
                # 圆心略...
            else:
                # --- 线 ---
                # ((x0, y0), (vx, vy))
                (x0, y0), (vx, vy) = shape_params
                # 画一条足够长的线
                L = 2000 
                p1 = (x0 - vx*L, y0 - vy*L)
                p2 = (x0 + vx*L, y0 + vy*L)
                line = QGraphicsLineItem(p1[0], p1[1], p2[0], p2[1])
                line.setPen(QPen(Qt.cyan, 2))
                self._scene.addItem(line)
                self.result_items.append(line)

    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 1.0/1.15
        self.scale(factor, factor)

    def mousePressEvent(self, event):
        sp = self.mapToScene(event.pos())
        if event.button() == Qt.LeftButton:
            if self.mode == "PICK":
                self.temp_items.append(self._scene.addEllipse(sp.x()-3, sp.y()-3, 6, 6, QPen(Qt.yellow), QBrush(Qt.yellow)))
                self.point_selected.emit(sp.x(), sp.y())
            elif self.mode == "DIRECTION":
                self._drag_start = sp
                self._drag_item = self._scene.addLine(sp.x(), sp.y(), sp.x(), sp.y(), QPen(Qt.magenta, 2))
            elif self.mode == "SAMPLE_BG":
                self._drag_start = sp
                self._drag_item = self._scene.addRect(sp.x(), sp.y(), 0, 0, QPen(Qt.green, 2))
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_start:
            sp = self.mapToScene(event.pos())
            if self.mode == "DIRECTION":
                self._drag_item.setLine(QLineF(self._drag_start, sp))
            elif self.mode == "SAMPLE_BG":
                rect = QRectF(self._drag_start, sp).normalized()
                self._drag_item.setRect(rect)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._drag_start:
            sp = self.mapToScene(event.pos())
            
            if self.mode == "DIRECTION":
                length = np.hypot(sp.x()-self._drag_start.x(), sp.y()-self._drag_start.y())
                if length > 5:
                    p1 = (self._drag_start.x(), self._drag_start.y())
                    p2 = (sp.x(), sp.y())
                    self.direction_defined.emit(length, (p1, p2))
                    self.temp_items.append(self._drag_item)
                else:
                    self._scene.removeItem(self._drag_item)
                    
            elif self.mode == "SAMPLE_BG":
                rect = QRectF(self._drag_start, sp).normalized()
                if rect.width() > 2 and rect.height() > 2:
                    self.bg_sampled.emit((rect.x(), rect.y(), rect.width(), rect.height()))
                    # 变成虚线表示采样框
                    self._drag_item.setPen(QPen(Qt.green, 1, Qt.DashLine))
                    self.temp_items.append(self._drag_item)
                else:
                    self._scene.removeItem(self._drag_item)
            
            self._drag_start = None
            self._drag_item = None
        super().mouseReleaseEvent(event)

# ==============================================================================
# 3. Main Window 
# ==============================================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("智能边缘测量 (颜色距离波形法)")
        self.resize(1300, 850)
        
        self.cv_image = None   # 保存彩色图
        self.points = []       # 定位点
        self.bg_color = None   # 背景均值
        self.scan_config = {"width": 40.0, "vector": None}
        
        self.setup_ui()

    def setup_ui(self):
        main = QWidget()
        self.setCentralWidget(main)
        h = QHBoxLayout(main)

        # Left panel
        panel = QWidget(); panel.setFixedWidth(340)
        vbox = QVBoxLayout(panel)

        # 1. 图像加载
        vbox.addWidget(QLabel("<b>1. 图像加载 / Image</b>"))
        btn_load = QPushButton("打开图片 / Open Image")
        btn_load.clicked.connect(self.load_image)
        vbox.addWidget(btn_load)

        #1-2 模式选择
        vbox.addSpacing(5)
        self.rb_circle = QRadioButton("圆形测量"); self.rb_circle.setChecked(True)
        self.rb_line = QRadioButton("直线测量")
        # 连接信号，切换模式时重置数据
        self.rb_circle.toggled.connect(self.reset_state)
        self.rb_line.toggled.connect(self.reset_state)
        
        bg_layout = QHBoxLayout()
        bg_layout.addWidget(self.rb_circle); bg_layout.addWidget(self.rb_line)
        vbox.addLayout(bg_layout)

        # 2. 采样背景 (新增)
        vbox.addSpacing(10)
        vbox.addWidget(QLabel("<b>2. 背景设置 / Background</b>"))
        self.btn_bg = QPushButton("框选背景区域 (Sample BG)")
        self.btn_bg.clicked.connect(self.start_sample_bg)
        vbox.addWidget(self.btn_bg)
        
        self.lbl_bg_status = QLabel("未采样")
        self.lbl_bg_status.setStyleSheet("background-color: #eee; border:1px solid gray; padding:2px")
        vbox.addWidget(self.lbl_bg_status)

        # 3. 参数设置
        vbox.addSpacing(10)
        vbox.addWidget(QLabel("<b>3. 参数设置 / Params</b>"))
        
        # 角度步长 (新增)
        row_step = QHBoxLayout(); row_step.addWidget(QLabel("角度步长(°):"))
        self.spin_step = QDoubleSpinBox(); self.spin_step.setRange(0.1, 10.0); self.spin_step.setValue(1.0)
        row_step.addWidget(self.spin_step); vbox.addLayout(row_step)

        # 剔除阈值
        row_out = QHBoxLayout(); row_out.addWidget(QLabel("剔除阈值(px):"))
        self.spin_outlier = QDoubleSpinBox(); self.spin_outlier.setRange(0.1, 100.0); self.spin_outlier.setValue(3.0)
        row_out.addWidget(self.spin_outlier); vbox.addLayout(row_out)
        
        # 比例
        row_scale = QHBoxLayout(); row_scale.addWidget(QLabel("比例 (mm/px):"))
        self.spin_scale = QDoubleSpinBox(); self.spin_scale.setDecimals(4); self.spin_scale.setValue(0.0100)
        row_scale.addWidget(self.spin_scale); vbox.addLayout(row_scale)

        # 4. 操作流程
        vbox.addSpacing(10)
        vbox.addWidget(QLabel("<b>4. 操作流程 / Flow</b>"))
        self.btn_pick = QPushButton("Step A: 选取3点定圆心")
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

        self.btn_calc = QPushButton("Step C: 运行计算 (Color)")
        self.btn_calc.setEnabled(False)
        self.btn_calc.setStyleSheet("background-color:#d0f0c0; font-weight:bold; height:40px")
        self.btn_calc.clicked.connect(self.run_measurement)
        vbox.addWidget(self.btn_calc)

        vbox.addSpacing(6)
        self.lbl_result = QLabel("结果显示区域")
        self.lbl_result.setFrameShape(QLabel.Panel)
        self.lbl_result.setWordWrap(True)
        vbox.addWidget(self.lbl_result)
        vbox.addStretch()
        h.addWidget(panel)

        # Right view
        self.view = InteractiveView()
        self.view.point_selected.connect(self.on_point_picked)
        self.view.direction_defined.connect(self.on_direction_set)
        self.view.bg_sampled.connect(self.on_bg_sampled)
        h.addWidget(self.view)

    # -----------------------
    # Logic
    # -----------------------
    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.bmp *.tif)")
        if not path: return
        self.cv_image = cv2.imread(path) # BGR
        if self.cv_image is None: return
        
        self.view.load_image(self.cv_image)
        self.reset_state()
        self.lbl_bg_status.setText("请框选背景颜色")

    def reset_state(self):
        self.points = []
        self.bg_color = None
        self.scan_config = {"width": 40.0, "vector": None}
        self.view.clear_overlays(); self.view.clear_results()
        self.view.mode = "VIEW"
        self.btn_dir.setEnabled(False); self.btn_calc.setEnabled(False)
        self.lbl_hint_pick.setText("等待选点...")
        self.lbl_result.setText("")

    # --- 背景采样 ---
    def start_sample_bg(self):
        if self.cv_image is None: return
        self.view.mode = "SAMPLE_BG"
        QMessageBox.information(self, "提示", "请在背景区域拉取矩形框。")

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

    # --- 选点 ---
    def start_pick_points(self):
        if self.cv_image is None: return
        self.points = []
        self.view.clear_overlays(); self.view.clear_results()
        self.view.mode = "PICK"
        
        if self.rb_circle.isChecked():
            self.lbl_hint_pick.setText("请点击 3 个点 (定圆)...")
        else:
            self.lbl_hint_pick.setText("请点击 2 个点 (定线)...")


    # 判断结束条件
    def on_point_picked(self, x, y):
        self.points.append((x, y))
        
        target_pts = 3 if self.rb_circle.isChecked() else 2
        self.lbl_hint_pick.setText(f"已选 {len(self.points)} / {target_pts}")
        
        if len(self.points) >= target_pts:
            self.view.mode = "VIEW"
            self.lbl_hint_pick.setText("选点完成")
            self.btn_dir.setEnabled(True)
            self.lbl_hint_dir.setText("请绘制扫描宽度")
            
            # 预绘制参考
            if self.rb_circle.isChecked():
                res = MeasureUtils.fit_circle_robust(self.points)
                if res["success"]:
                    (cx, cy), r = res["center"], res["radius"]
                    itm = QGraphicsEllipseItem(cx-r, cy-r, r*2, r*2)
                    itm.setPen(QPen(Qt.yellow, 1, Qt.DashLine))
                    self.view._scene.addItem(itm); self.view.temp_items.append(itm)
            else:
                # 直线参考
                p1, p2 = self.points[0], self.points[-1]
                itm = QGraphicsLineItem(p1[0], p1[1], p2[0], p2[1])
                itm.setPen(QPen(Qt.yellow, 1, Qt.DashLine))
                self.view._scene.addItem(itm); self.view.temp_items.append(itm)

    # --- 扫描方向/宽度 ---
    def start_draw_direction(self):
        self.view.mode = "DIRECTION"
        QMessageBox.information(self, "提示", "请拖拽一条线段定义扫描宽度。\n方向建议：从背景指向工件（用于辅助判断方向，实际使用智适应）。")

    def on_direction_set(self, length, vector):
        self.scan_config["width"] = length
        self.scan_config["vector"] = vector
        self.view.mode = "VIEW"
        self.lbl_hint_dir.setText(f"宽度: {length:.1f}px")
        
        if self.bg_color is not None:
            self.btn_calc.setEnabled(True)
        else:
            QMessageBox.warning(self, "警告", "请先完成 Step 2: 采样背景颜色")

    # --- 核心计算 ---
    def measure_circle_color(self):
        if self.bg_color is None: return
        self.view.clear_results()
        
        if self.rb_circle.isChecked():
            self.measure_circle_color()
        else:
            self.measure_line_color()
        self.view.clear_results()
        
        # 1. 获取基础圆
        res_base = MeasureUtils.fit_circle_robust(self.points)
        if not res_base["success"]: 
            self.lbl_result.setText("初始圆拟合失败")
            return
        (cx, cy), r_base = res_base["center"], res_base["radius"]
        
        # 2. 准备扫描参数
        scan_width = self.scan_config["width"]
        angle_step = self.spin_step.value()
        
        # 计算方向 (内->外 还是 外->内)
        # 根据用户画线的起点终点距离圆心的远近来判断
        (vx1, vy1), (vx2, vy2) = self.scan_config["vector"]
        d1 = np.hypot(vx1-cx, vy1-cy)
        d2 = np.hypot(vx2-cx, vy2-cy)
        is_outward = d2 > d1
        
        detected_pts = []
        num_spokes = int(360.0 / angle_step)
        if num_spokes < 10: num_spokes = 10
        
        # 3. 循环扫描 (不使用极坐标Unroll，使用原来的Spoke模式)
        for i in range(num_spokes):
            angle_deg = i * angle_step
            ang = np.deg2rad(angle_deg)
            ca, sa = np.cos(ang), np.sin(ang)
            
            # 定义扫描线段
            r_in = r_base - scan_width / 2.0
            r_out = r_base + scan_width / 2.0
            
            # 无论扫描方向如何设定，我们这里始终构建物理坐标
            # is_outward 决定的是 profile 的起点是内还是外
            if is_outward:
                p_s = (cx + r_in*ca, cy + r_in*sa)
                p_e = (cx + r_out*ca, cy + r_out*sa)
            else:
                p_s = (cx + r_out*ca, cy + r_out*sa)
                p_e = (cx + r_in*ca, cy + r_in*sa)
            
            # 获取【彩色剖面】
            prof_bgr = MeasureUtils.get_color_profile(self.cv_image, p_s, p_e, scan_width)
            
            # 【颜色距离边缘检测】
            idx = MeasureUtils.measure_color_edge_adaptive(prof_bgr, self.bg_color)
            
            if idx is not None:
                # 还原坐标
                ratio = idx / float(len(prof_bgr)-1)
                px = p_s[0] + (p_e[0]-p_s[0]) * ratio
                py = p_s[1] + (p_e[1]-p_s[1]) * ratio
                detected_pts.append((px, py))

        if len(detected_pts) < 10:
            self.lbl_result.setText(f"检测到的点太少 ({len(detected_pts)})，请检查背景颜色或扫描宽度")
            return

        # 4. 最终鲁棒拟合
        outlier_thresh = self.spin_outlier.value()
        final_res = MeasureUtils.fit_circle_robust(detected_pts, outlier_thresh)
        
        # 5. 绘制结果
        # ROI 信息用于画黄圈
        roi_info = ((cx, cy), r_base - scan_width/2, r_base + scan_width/2)
        
        if final_res["success"]:
            # === 修改开始：分离有效点和离群点 ===
            
            # 1. 获取有效点 (Numpy Array) 并转为 Set 方便比对
            valid_pts_np = final_res["valid_points"]
            # 注意：坐标比对需要处理浮点精度，这里简单转为 tuple
            valid_set = set((float(p[0]), float(p[1])) for p in valid_pts_np)
            
            valid_pts_list = []
            outlier_pts_list = []
            
            # 2. 遍历所有检测点，分类
            for pt in detected_pts:
                # pt 是 tuple (x, y)
                pt_key = (float(pt[0]), float(pt[1]))
                if pt_key in valid_set:
                    valid_pts_list.append(pt)
                else:
                    outlier_pts_list.append(pt)
            
            # 3. 准备拟合圆参数
            circle_par = (final_res["center"], final_res["radius"])
            
            # 4. 调用绘图 (传入有效点 和 离群点)
            self.view.draw_results(valid_pts_list, circle_par, roi_info, outliers=outlier_pts_list)
            
            # === 修改结束 ===
            
            # 结果显示
            scale = self.spin_scale.value()
            phys_d = final_res["radius"] * 2.0 * scale
            phys_circ = final_res["circularity"] * scale
            
            self.lbl_result.setText(
                f"【测量成功】\n"
                f"总检测点: {len(detected_pts)}\n"
                f"有效点数: {len(valid_pts_list)} (红色)\n"
                f"离群点数: {len(outlier_pts_list)} (蓝色)\n"
                f"物理直径: {phys_d:.4f} mm\n"
                f"圆度误差: {phys_circ:.4f} mm\n"
                f"圆心: ({final_res['center'][0]:.1f}, {final_res['center'][1]:.1f})"
            )
        else:
            self.lbl_result.setText(f"拟合失败: {final_res.get('msg')}")
    def measure_line_color(self):
        # 1. 获取基准线
        if len(self.points) < 2: return
        p_start = np.array(self.points[0])
        p_end = np.array(self.points[-1])
        
        # 2. 准备参数
        scan_width = self.scan_config["width"]
        # 直线模式下，Step 代表“像素步长”
        pixel_step = self.spin_step.value() 
        if pixel_step < 0.5: pixel_step = 0.5
        
        # 计算基准线方向向量
        vec = p_end - p_start
        length = np.linalg.norm(vec)
        if length < 1: return
        
        unit_v = vec / length  # 切向
        norm_v = np.array([-unit_v[1], unit_v[0]]) # 法向
        
        detected_pts = []
        
        # 3. 循环扫描 (沿着直线走)
        # num_steps = 长度 / 步长
        num_steps = int(length / pixel_step)
        
        for i in range(num_steps + 1):
            # 当前中心点
            current_dist = i * pixel_step
            if current_dist > length: break
            
            center_k = p_start + unit_v * current_dist
            
            # 定义扫描线段 (沿法向展开)
            # 扫描方向由 Step B 决定吗？
            # 实际上颜色距离法是“无方向”的（看距离变化），但为了获得profile，我们需要一个线段
            # 我们以 center_k 为中心，沿法向向两侧延伸
            p_s = center_k - norm_v * (scan_width / 2.0)
            p_e = center_k + norm_v * (scan_width / 2.0)
            
            # 获取剖面
            prof_bgr = MeasureUtils.get_color_profile(self.cv_image, p_s, p_e, scan_width)
            
            # 检测边缘
            idx = MeasureUtils.measure_color_edge_adaptive(prof_bgr, self.bg_color)
            
            if idx is not None:
                # 还原坐标
                ratio = idx / float(len(prof_bgr)-1)
                px = p_s[0] + (p_e[0]-p_s[0]) * ratio
                py = p_s[1] + (p_e[1]-p_s[1]) * ratio
                detected_pts.append((px, py))
                
        if len(detected_pts) < 2:
             self.lbl_result.setText("检测到的点不足")
             return

        # 4. 鲁棒直线拟合
        outlier_thresh = self.spin_outlier.value()
        final_res = MeasureUtils.fit_line_robust(detected_pts, outlier_thresh)
        
        # 5. 绘制与显示
        # 构建 ROI Info 用于画图: (p_start, p_end, width, normal)
        roi_info = ("line", p_start, p_end, scan_width, norm_v)
        
        if final_res["success"]:
            valid_pts = final_res["valid_points"]
            valid_set = set((float(p[0]), float(p[1])) for p in valid_pts)
            
            valid_list = []
            outlier_list = []
            for pt in detected_pts:
                if (float(pt[0]), float(pt[1])) in valid_set:
                    valid_list.append(pt)
                else:
                    outlier_list.append(pt)
            
            # 结果参数
            line_res = (final_res["line_point"], final_res["line_dir"])
            
            self.view.draw_results(valid_list, line_res, roi_info, outliers=outlier_list)
            
            scale = self.spin_scale.value()
            phys_straight = final_res["straightness"] * scale
            
            self.lbl_result.setText(
                f"【直线测量成功】\n"
                f"总检测点: {len(detected_pts)}\n"
                f"有效点数: {len(valid_list)} (红色)\n"
                f"离群点数: {len(outlier_list)} (蓝色)\n"
                f"直线度: {phys_straight:.4f} mm"
            )
        else:
            self.lbl_result.setText(f"拟合失败: {final_res.get('msg')}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())