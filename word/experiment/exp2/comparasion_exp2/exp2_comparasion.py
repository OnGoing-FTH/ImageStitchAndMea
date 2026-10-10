import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QSpinBox,
                             QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
                             QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsRectItem,
                             QGraphicsPolygonItem, QComboBox,
                             QFileDialog, QDoubleSpinBox, QGroupBox, QRadioButton, QMessageBox)
from PyQt5.QtCore import Qt, pyqtSignal, QPointF, QLineF, QRectF
from PyQt5.QtGui import QPixmap, QImage, QPen, QBrush, QPainter, QPolygonF




from edge_methods.canny_edge import CannyEdge
from edge_methods.sobel_edge import SobelEdge
from edge_methods.pidinet_edge import PiDiNetEdge
from edge_methods.gray_projection import GrayProjectionEdge
from edge_methods.zernike_edge import ZernikeEdge
from edge_methods.color_vector import ColorVectorEdge

from fit.fit_ls import FitLS
from fit.fit_robust import FitRobust
from fit.fit_average import FitAverage


# ==============================================================================
# 1. Utilities & View
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
    def fit_circle_initial(points):
        """ UI交互用的粗略拟合，用于辅助生成 ROI """
        if points is None or len(points) < 3: return {"success": False}
        pts = np.array(points)
        x, y = pts[:, 0], pts[:, 1]
        A = np.column_stack((2*x, 2*y, np.ones_like(x)))
        b = x**2 + y**2
        try:
            res, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            cx, cy = res[0], res[1]
            r = np.sqrt(res[2] + cx**2 + cy**2)
            return {"success": True, "center": (cx, cy), "radius": r}
        except:
            return {"success": False}

class InteractiveView(QGraphicsView):
    point_selected = pyqtSignal(float, float)
    direction_defined = pyqtSignal(object) 
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
        self._drag_arrow_head = None

    def load_image(self, cv_img):
        self._scene.clear()
        self.temp_items = []; self.result_items = []
        h, w = cv_img.shape[:2]
        # BGR转RGB
        q_img = QImage(cv_img.data, w, h, 3*w, QImage.Format_RGB888).rgbSwapped()
        self._scene.addItem(QGraphicsPixmapItem(QPixmap.fromImage(q_img)))

    def clear_overlays(self):
        for it in self.temp_items: self._scene.removeItem(it)
        self.temp_items = []

    def clear_results(self):
        for it in self.result_items: self._scene.removeItem(it)
        self.result_items = []

    def draw_results(self, valid_points, shape_params, roi_info, outliers=None):
        # 1. 绘制 ROI 范围
        if roi_info:
            if len(roi_info) == 3 and not isinstance(roi_info[0], str):
                # 圆形 ROI: ((cx, cy), r_min, r_max)
                (cx, cy), r_in, r_out = roi_info
                pen_roi = QPen(Qt.yellow, 1, Qt.DashLine)
                roi1 = QGraphicsEllipseItem(cx-r_in, cy-r_in, r_in*2, r_in*2); roi1.setPen(pen_roi)
                roi2 = QGraphicsEllipseItem(cx-r_out, cy-r_out, r_out*2, r_out*2); roi2.setPen(pen_roi)
                self._scene.addItem(roi1); self._scene.addItem(roi2)
                self.result_items.extend([roi1, roi2])
            elif len(roi_info) == 6 and roi_info[0] == "line":
                # 直线 ROI: ("line", p_s, p_e, dist_min, dist_max, nv)
                # p_s, p_e 是基准线段的起终点
                # dist_min, dist_max 是沿法线方向的偏移量
                _, p_s, p_e, d_min, d_max, nv = roi_info
                
                # 计算四个角点用于绘制矩形框
                c1 = p_s + nv * d_min
                c2 = p_s + nv * d_max
                c3 = p_e + nv * d_max
                c4 = p_e + nv * d_min
                
                poly = QPolygonF([QPointF(c1[0],c1[1]), QPointF(c2[0],c2[1]), 
                                  QPointF(c3[0],c3[1]), QPointF(c4[0],c4[1])])
                roi_item = QGraphicsPolygonItem(poly)
                roi_item.setPen(QPen(Qt.yellow, 1, Qt.DashLine))
                self._scene.addItem(roi_item)
                self.result_items.append(roi_item)

        # 2. 绘制检测点
        def draw_cross(pts, color):
            pen = QPen(color, 1)
            for pt in pts:
                px, py = pt
                s = 2
                l1 = QGraphicsLineItem(px-s, py-s, px+s, py+s); l1.setPen(pen)
                l2 = QGraphicsLineItem(px-s, py+s, px+s, py-s); l2.setPen(pen)
                self._scene.addItem(l1); self._scene.addItem(l2)
                self.result_items.extend([l1, l2])

        if valid_points is not None: draw_cross(valid_points, Qt.red)
        if outliers is not None: draw_cross(outliers, Qt.blue)

        # 3. 绘制拟合结果
        if shape_params:
            if len(shape_params) == 2 and isinstance(shape_params[1], (float, int, np.float32, np.float64)):
                # 圆
                (cx, cy), r = shape_params
                c = QGraphicsEllipseItem(cx-r, cy-r, r*2, r*2)
                c.setPen(QPen(Qt.cyan, 2))
                self._scene.addItem(c)
                self.result_items.append(c)
                # 中心
                l1 = QGraphicsLineItem(cx-10, cy, cx+10, cy); l1.setPen(QPen(Qt.red, 2))
                l2 = QGraphicsLineItem(cx, cy-10, cx, cy+10); l2.setPen(QPen(Qt.red, 2))
                self._scene.addItem(l1); self._scene.addItem(l2)
                self.result_items.extend([l1, l2])
            else:
                # 线
                (x0, y0), (vx, vy) = shape_params
                L = 3000
                p1 = (x0 - vx*L, y0 - vy*L)
                p2 = (x0 + vx*L, y0 + vy*L)
                li = QGraphicsLineItem(p1[0], p1[1], p2[0], p2[1])
                li.setPen(QPen(Qt.cyan, 2))
                self._scene.addItem(li)
                self.result_items.append(li)

    # 鼠标交互逻辑保持不变
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
# 2. Main Window
# ==============================================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("边缘检测算法对比实验 (Comparison Exp 2)")
        self.resize(1300, 850)
        
        self.cv_image = None
        self.points = []
        self.bg_color = None
        self.scan_config = {"roi_points": None}

        # 初始化算法组合 (Strategy Pattern)
        try:
            # 实例化边缘检测器
            canny = CannyEdge()
            sobel = SobelEdge()
            pidinet = PiDiNetEdge()
            gray = GrayProjectionEdge()
            zernike = ZernikeEdge()
            color = ColorVectorEdge()

            # 实例化拟合器
            ls_fit = FitLS()
            robust_fit = FitRobust()
            avg_fit = FitAverage()

            self.algorithms = {
                "Canny + LS":       (canny, ls_fit),
                "Canny+Nofit":    (canny, avg_fit),

                "Sobel + LS":       (sobel, ls_fit),
                "Sobel + Nofit":    (sobel, avg_fit),

                "PiDiNet + LS":     (pidinet, ls_fit),
                "PiDiNet + Nofit":     (pidinet, avg_fit),


                "GrayProj + LS":    (gray, ls_fit),
                "GrayProj + Nofit": (gray, avg_fit),


                "Zernike + LS":     (zernike, ls_fit),
                "Zernike + Nofit":  (zernike, avg_fit),



                "Color + LS":       (color, ls_fit),
                "Color + Robust":   (color, robust_fit),
                "Color + Nofit":   (color, avg_fit),
            }
        except Exception as e:
            QMessageBox.critical(self, "初始化错误", f"算法类初始化失败: {str(e)}")
            self.algorithms = {}

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
        vbox.addWidget(QLabel("<b>2. 实验设置 / Settings</b>"))
        bg_layout = QHBoxLayout()
        self.rb_circle = QRadioButton("圆形测量"); self.rb_circle.setChecked(True)
        self.rb_line = QRadioButton("直线测量")
        self.rb_circle.toggled.connect(self.reset_state)
        self.rb_line.toggled.connect(self.reset_state)
        bg_layout.addWidget(self.rb_circle); bg_layout.addWidget(self.rb_line)
        vbox.addLayout(bg_layout)

        vbox.addWidget(QLabel("选择算法组合:"))
        self.combo_algo = QComboBox()
        self.combo_algo.addItems(self.algorithms.keys())
        idx = self.combo_algo.findText("Color + Robust")
        if idx >= 0: self.combo_algo.setCurrentIndex(idx)
        vbox.addWidget(self.combo_algo)

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
        
        row_out = QHBoxLayout(); row_out.addWidget(QLabel("拟合阈值(px):"))
        self.spin_outlier = QDoubleSpinBox(); self.spin_outlier.setRange(0.1, 100.0); self.spin_outlier.setValue(3.0)
        row_out.addWidget(self.spin_outlier); gl.addLayout(row_out)
        
        row_scale = QHBoxLayout(); row_scale.addWidget(QLabel("比例 (mm/px):"))
        self.spin_scale = QDoubleSpinBox(); self.spin_scale.setDecimals(6); self.spin_scale.setValue(0.0100)
        row_scale.addWidget(self.spin_scale); gl.addLayout(row_scale)
        vbox.addWidget(grp)

        vbox.addSpacing(10)
        vbox.addWidget(QLabel("<b>4. 流程 / Flow</b>"))
        self.btn_pick = QPushButton("Step A: 选点 (粗定位)")
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
            # 显示粗略几何
            if self.rb_circle.isChecked():
                res = MeasureUtils.fit_circle_initial(self.points)
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
        QMessageBox.information(self, "提示", "请绘制一个箭头定义扫描区域。\n箭头的【长度】决定扫描宽度，【方向】决定扫描方向。")

    def on_direction_set(self, points):
        p1, p2 = points
        length = np.hypot(p1[0]-p2[0], p1[1]-p2[1])
        self.scan_config["roi_points"] = points
        self.view.mode = "VIEW"
        self.lbl_hint_dir.setText(f"范围已定 ({length:.1f}px)")
        if self.cv_image is not None:
            self.btn_calc.setEnabled(True)

    def run_measurement(self):
        algo_name = self.combo_algo.currentText()
        if algo_name not in self.algorithms: return

        edge_method, fit_method = self.algorithms[algo_name]
        
        self.view.clear_results()
        
        if self.rb_circle.isChecked():
            self.measure_circle_complete(edge_method, fit_method)
        else:
            self.measure_line_complete(edge_method, fit_method)

    # ==========================================================================
    # 核心逻辑：计算 ROI -> 调用算法.extract -> 调用拟合
    # ==========================================================================
    
    def measure_circle_complete(self, edge_extractor, fitter):
        """ 圆形测量流程：准备 ROI -> extract -> fit """
        # 1. 准备几何参数 (圆心)
        res_base = MeasureUtils.fit_circle_initial(self.points)
        if not res_base["success"]: 
            self.lbl_result.setText("粗略拟合失败，无法确定 ROI")
            return
        (cx, cy), r_base = res_base["center"], res_base["radius"]
        
        # 2. 准备 ROI 参数 (内外半径)
        # Step B 绘制的箭头两端到圆心的距离，定义了扫描的环形区域
        p_scan_start, p_scan_end = self.scan_config["roi_points"]
        r1 = np.hypot(p_scan_start[0] - cx, p_scan_start[1] - cy)
        r2 = np.hypot(p_scan_end[0] - cx, p_scan_end[1] - cy)
        r_in = min(r1, r2)
        r_out = max(r1, r2)
        
        # 构造传递给 extract 的 roi_info: ((cx, cy), r_in, r_out)
        roi_info = ((cx, cy), r_in, r_out)
        
        # 3. 调用算法提取点集
        # 注意：此处不再循环，循环逻辑包含在 ColorVectorEdge.extract 内部，或 Canny 的全局过滤中
        step_val = self.spin_step.value()
        try:
            detected_pts = edge_extractor.extract(
                self.cv_image, 
                roi_info, 
                step=step_val, 
                bg_color=self.bg_color
            )
        except Exception as e:
            self.lbl_result.setText(f"提取过程出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return

        if not detected_pts or len(detected_pts) < 5:
            self.lbl_result.setText(f"检测点数不足 ({len(detected_pts) if detected_pts else 0})")
            return

        # 4. 拟合
        self.perform_fitting(fitter, detected_pts, roi_info, is_circle=True)

    def measure_line_complete(self, edge_extractor, fitter):
        """ 直线测量流程：准备 ROI -> extract -> fit """
        if len(self.points) < 2: return
        
        # 1. 准备基准线
        p_start_base = np.array(self.points[0])
        p_end_base = np.array(self.points[-1])
        vec = p_end_base - p_start_base
        length = np.linalg.norm(vec)
        if length < 1: return
        unit_v = vec / length
        norm_v = np.array([-unit_v[1], unit_v[0]]) # 法向量
        
        # 2. 准备 ROI 参数 (沿法向量的扫描宽度)
        p_scan_start, p_scan_end = self.scan_config["roi_points"]
        p_scan_start_arr = np.array(p_scan_start)
        p_scan_end_arr = np.array(p_scan_end)
        
        # 计算投影距离
        offset_s = np.dot(p_scan_start_arr - p_start_base, norm_v)
        offset_e = np.dot(p_scan_end_arr - p_start_base, norm_v)
        dist_min = min(offset_s, offset_e)
        dist_max = max(offset_s, offset_e)
        
        # 构造传递给 extract 的 roi_info
        # 格式约定: ("line", p_base_start, p_base_end, dist_min, dist_max, norm_vec)
        roi_info = ("line", p_start_base, p_end_base, dist_min, dist_max, norm_v)
        
        # 3. 调用算法提取点集
        step_val = self.spin_step.value()
        # 4
        
        try:
            detected_pts = edge_extractor.extract(
                self.cv_image, 
                roi_info, 
                step=step_val, 
                bg_color=self.bg_color
            )

            self.perform_fitting(fitter, detected_pts, roi_info, is_circle=False)
        except Exception as e:
            print("\n perform_fitting 崩溃了")
            print(f"错误类型: {type(e).__name__}")
            print(f"错误详情: {str(e)}")
            import traceback
            traceback.print_exc()
            # 强制更新标签显示真实错误
            self.lbl_result.setText(f"逻辑错误: {str(e)}")

        # if not detected_pts or len(detected_pts) < 2:
        #      self.lbl_result.setText(f"检测点数不足 ({len(detected_pts) if detected_pts else 0})")
        #      return

        # 4. 拟合
        self.perform_fitting(fitter, detected_pts, roi_info, is_circle=False)

    def perform_fitting(self, fitter, detected_pts, roi_info, is_circle):
        """ 通用拟合与显示逻辑 (复活离群点显示版) """
        try:
            # 兼容 Robust (带阈值) 和 LS (不带阈值)
            if isinstance(fitter, FitRobust):
                threshold = self.spin_outlier.value()
                if is_circle:
                    final_res = fitter.fit_circle(detected_pts, outlier_threshold=threshold)
                else:
                    final_res = fitter.fit_line(detected_pts, outlier_threshold=threshold)
            else:
                if is_circle:
                    final_res = fitter.fit_circle(detected_pts)
                else:
                    final_res = fitter.fit_line(detected_pts)
        except Exception as e:
            self.lbl_result.setText(f"拟合计算错误: {e}")
            import traceback
            traceback.print_exc()
            return

        if final_res["success"]:
            # 1. 获取最原始的有效点 (优先用拟合结果)
            raw_pts = final_res.get("valid_points", detected_pts)
            
            # 2. 强制转为 List，确保计算长度时不报错
            if isinstance(raw_pts, np.ndarray):
                valid_list = raw_pts.tolist()
            else:
                valid_list = raw_pts

            # 3. 空值兜底
            if not valid_list:
                valid_list = detected_pts # 如果空了，就认为全是有效点

            # ========================================================
            # 🔴 复活离群点计算 (使用模糊匹配，防止精度Bug)
            # ========================================================
            outlier_list = []
            
            # 定义一个生成 Key 的函数，保留3位小数，忽略微小误差
            def get_key(pt):
                return (round(float(pt[0]), 3), round(float(pt[1]), 3))

            # 制作有效点的“指纹”集合
            valid_keys = set(get_key(p) for p in valid_list)

            # 遍历所有点，如果指纹不在有效集合里，就是离群点
            for pt in detected_pts:
                if get_key(pt) not in valid_keys:
                    outlier_list.append(pt)
            # ========================================================

            # 绘制
            if is_circle:
                shape_par = (final_res["center"], final_res["radius"])
            else:
                shape_par = (final_res["line_point"], final_res["line_dir"])
            
            # 这里传入了 outlier_list，蓝叉叉就会回来了
            self.view.draw_results(valid_list, shape_par, roi_info, outliers=outlier_list)
            
            # 显示文本结果
            scale = self.spin_scale.value()
            if is_circle:
                phys_d = final_res["radius"] * 2.0 * scale
                phys_err = final_res.get("circularity", 0.0) * scale
                msg = (f"【{self.combo_algo.currentText()}】\n"
                       f"总点数: {len(detected_pts)}\n"
                       f"有效点: {len(valid_list)} | 离群: {len(outlier_list)}\n"
                       f"直径: {phys_d:.4f} mm\n"
                       f"圆度: {phys_err:.4f} mm")
            else:
                phys_err = final_res.get("straightness", 0.0) * scale
                # 计算长度 (投影范围)
                vx, vy = final_res["line_dir"]
                
                # 安全防护
                if len(valid_list) < 2:
                    msg = (f"【{self.combo_algo.currentText()}】\n"
                        f"总点数: {len(detected_pts)}\n"
                        f"有效点不足，无法计算长度\n"
                        f"直线度: {phys_err:.4f} mm")
                    self.lbl_result.setText(msg)
                    return

                v_pts = np.array(valid_list)
                proj = v_pts[:, 0] * vx + v_pts[:, 1] * vy
                length_mm = (np.max(proj) - np.min(proj)) * scale
                msg = (f"【{self.combo_algo.currentText()}】\n"
                       f"总点数: {len(detected_pts)}\n"
                       f"有效点: {len(valid_list)} | 离群: {len(outlier_list)}\n"
                       f"长度: {length_mm:.4f} mm\n"
                       f"直线度: {phys_err:.4f} mm")
            
            self.lbl_result.setText(msg)
        else:
            self.lbl_result.setText(f"拟合失败: {final_res.get('msg')}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())