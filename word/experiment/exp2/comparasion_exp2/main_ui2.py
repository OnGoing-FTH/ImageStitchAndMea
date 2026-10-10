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

# ==============================================================================
# 0. 导入外部算法模块 (根据你的目录结构)
# ==============================================================================

from edge_methods.canny_edge import CannyEdge
from edge_methods.sobel_edge import SobelEdge
# from edge_methods.pidinet_edge import PiDiNetEdge
from edge_methods.gray_projection import GrayProjectionEdge
from edge_methods.zernike_edge import ZernikeEdge
from edge_methods.color_vector import ColorVectorEdge

from fit.fit_ls import fit_circle_ls, fit_line_ls
from fit.fit_robust import fit_circle_robust, fit_line_robust


# ==============================================================================
# 1. Utilities & View (UI 组件)
# ==============================================================================

class MeasureUtils:
    """ 
    保留原本的辅助函数，主要用于UI交互时的背景采样。
    核心算法现在委托给 edge_methods 和 fit 模块。
    """
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
    
    # 这里的 fit_circle_robust 仅用于 Step A 的初始交互辅助（粗定位），
    # 最终计算使用 fit 模块中的类。
    @staticmethod
    def fit_circle_initial(points):
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
        # BGR to RGB
        q_img = QImage(cv_img.data, w, h, 3*w, QImage.Format_RGB888).rgbSwapped()
        self._scene.addItem(QGraphicsPixmapItem(QPixmap.fromImage(q_img)))

    def clear_overlays(self):
        for it in self.temp_items: self._scene.removeItem(it)
        self.temp_items = []

    def clear_results(self):
        for it in self.result_items: self._scene.removeItem(it)
        self.result_items = []

    def draw_results(self, valid_points, shape_params, roi_info, outliers=None):
        # 绘制 ROI 范围
        if roi_info:
            if len(roi_info) == 3 and not isinstance(roi_info[0], str):
                # 圆形 ROI: ((cx, cy), r_min, r_max)
                (cx, cy), r_in, r_out = roi_info
                pen_roi = QPen(Qt.yellow, 1, Qt.DashLine)
                roi1 = QGraphicsEllipseItem(cx-r_in, cy-r_in, r_in*2, r_in*2); roi1.setPen(pen_roi)
                roi2 = QGraphicsEllipseItem(cx-r_out, cy-r_out, r_out*2, r_out*2); roi2.setPen(pen_roi)
                self._scene.addItem(roi1); self._scene.addItem(roi2)
                self.result_items.extend([roi1, roi2])
            elif len(roi_info) == 5 and roi_info[0] == "line":
                # 直线 ROI: ("line", p_s, p_e, w, nv)
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

        # 绘制点
        def draw_cross(pts, color):
            pen = QPen(color, 1)
            for pt in pts:
                px, py = pt
                s = 2
                l1 = QGraphicsLineItem(px-s, py-s, px+s, py+s); l1.setPen(pen)
                l2 = QGraphicsLineItem(px-s, py+s, px+s, py-s); l2.setPen(pen)
                self._scene.addItem(l1); self._scene.addItem(l2)
                self.result_items.extend([l1, l2])

        draw_cross(valid_points, Qt.red)
        if outliers:
            draw_cross(outliers, Qt.blue)

        # 绘制拟合结果 (圆或线)
        if shape_params:
            if len(shape_params) == 2 and isinstance(shape_params[1], (float, int, np.float32, np.float64)):
                # 圆: ((cx, cy), r)
                (cx, cy), r = shape_params
                c = QGraphicsEllipseItem(cx-r, cy-r, r*2, r*2)
                c.setPen(QPen(Qt.cyan, 2))
                self._scene.addItem(c)
                self.result_items.append(c)
                # 十字中心
                l1 = QGraphicsLineItem(cx-10, cy, cx+10, cy); l1.setPen(QPen(Qt.red, 2))
                l2 = QGraphicsLineItem(cx, cy-10, cx, cy+10); l2.setPen(QPen(Qt.red, 2))
                self._scene.addItem(l1); self._scene.addItem(l2)
                self.result_items.extend([l1, l2])
            else:
                # 线: ((x0, y0), (vx, vy))
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

        # -----------------------------------------------------------
        # 初始化算法组合 (Strategy Pattern)
        # -----------------------------------------------------------
        try:
            # 实例化边缘检测器
            canny = CannyEdge()
            sobel = SobelEdge()
            # pidinet = PiDiNetEdge()
            gray = GrayProjectionEdge()
            zernike = ZernikeEdge()
            color = ColorVectorEdge()
            

            # 实例化拟合器
            ls_fit = fit_circle_ls()
            robust_fit = fit_circle_robust()

            self.algorithms = {
                "Canny + LS":       (canny, ls_fit),
                "Sobel + LS":       (sobel, ls_fit),
                # "PiDiNet + LS":     (pidinet, ls_fit),
                "GrayProj + LS":    (gray, ls_fit),
                "Zernike + LS":     (zernike, ls_fit),
                "Color + LS":       (color, ls_fit),
                "Color + Robust":   (color, robust_fit)
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

        # 算法选择下拉框
        vbox.addWidget(QLabel("选择算法组合:"))
        self.combo_algo = QComboBox()
        self.combo_algo.addItems(self.algorithms.keys())
        # 默认选中 Color + Robust 以匹配之前体验
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
        
        # 拟合阈值对于 Robust 有用，对于 LS 可能不用但保留作为统一接口参数
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
        QMessageBox.information(self, "提示", "请在图像【背景区域】拉一个矩形框。\n(部分算法如Canny可能忽略此步，但为保证一致性请执行)")

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
            # 使用粗略拟合仅用于UI显示，不用于最终计算
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
        QMessageBox.information(self, "提示", "请绘制一个箭头定义扫描区域。\n箭头的【长度】决定扫描宽度，【方向】决定扫描方向(从内到外或反之)。")

    def on_direction_set(self, points):
        p1, p2 = points
        length = np.hypot(p1[0]-p2[0], p1[1]-p2[1])
        self.scan_config["roi_points"] = points
        self.view.mode = "VIEW"
        self.lbl_hint_dir.setText(f"范围已定 ({length:.1f}px)")
        
        # 只要选了区域和图片即可允许计算，bg_color 缺失时算法内部处理
        if self.cv_image is not None:
            self.btn_calc.setEnabled(True)

    def run_measurement(self):
        algo_name = self.combo_algo.currentText()
        if algo_name not in self.algorithms: return

        edge_method, fit_method = self.algorithms[algo_name]
        
        # 传递背景颜色 (某些算法如 ColorVector 需要)
        # 我们假设 edge_method.detect 支持 bg_color 参数，或者在初始化时不需要
        # 为了通用性，我们将在循环中调用 .detect(...) 
        
        self.view.clear_results()
        
        if self.rb_circle.isChecked():
            self.measure_circle_generic(edge_method, fit_method)
        else:
            self.measure_line_generic(edge_method, fit_method)

    # ==========================================================================
    # 核心测量逻辑 (通用化)
    # ==========================================================================
    
    def measure_circle_generic(self, edge_detector, fitter):
        # 1. 获取粗略圆心 (仅用于生成扫描线)
        res_base = MeasureUtils.fit_circle_initial(self.points)
        if not res_base["success"]: 
            self.lbl_result.setText("粗略拟合失败，无法生成扫描路径")
            return
        (cx, cy), r_base = res_base["center"], res_base["radius"]
        
        # 2. 确定扫描参数 (保持 ROI 一致性)
        p_scan_start, p_scan_end = self.scan_config["roi_points"]
        r_start_scan = np.hypot(p_scan_start[0] - cx, p_scan_start[1] - cy)
        r_end_scan = np.hypot(p_scan_end[0] - cx, p_scan_end[1] - cy)
        
        scan_width = abs(r_end_scan - r_start_scan)
        if scan_width < 1.0: scan_width = 1.0
        
        angle_step = self.spin_step.value()
        detected_pts = []
        num_spokes = int(360.0 / angle_step)
        if num_spokes < 10: num_spokes = 10
        
        # 3. 扫描循环
        for i in range(num_spokes):
            angle_deg = i * angle_step
            ang = np.deg2rad(angle_deg)
            ca, sa = np.cos(ang), np.sin(ang)
            
            p_s = (cx + r_start_scan*ca, cy + r_start_scan*sa)
            p_e = (cx + r_end_scan*ca, cy + r_end_scan*sa)
            
            # --- 调用具体的边缘检测算法 ---
            # 假设接口: detect(image, p_start, p_end, scan_width, bg_color=None) -> (x, y) or None
            try:
                pt = edge_detector.detect(self.cv_image, p_s, p_e, scan_width, bg_color=self.bg_color)
                if pt is not None:
                    detected_pts.append(pt)
            except Exception as e:
                print(f"Detection error at angle {angle_deg}: {e}")
                continue

        if len(detected_pts) < 10:
            self.lbl_result.setText(f"检测点数不足 ({len(detected_pts)})")
            return

        # 4. 拟合
        # 假设接口: fit_circle(points) -> dict
        # 如果是 Robust 拟合，可能需要 outlier_threshold 参数，这里尝试注入参数
        # 我们可以检查 fitter 是否有 threshold 属性或者直接传参
        try:
            if hasattr(fitter, 'fit_circle'):
                # 尝试传递阈值，如果 fitter 不支持 robust 参数，需在 fitter 内部处理或使用 **kwargs
                # 这里假设你的 fit_robust.py 的 fit_circle 接收 threshold，而 fit_ls 忽略
                # 为了安全，我们假设 fitter 是标准化包装过的
                
                # 特殊处理：如果 fitter 是 FitRobust 实例，我们传入 spinbox 的值
                if isinstance(fitter, fit_circle_robust):
                    final_res = fitter.fit_circle(detected_pts, outlier_threshold=self.spin_outlier.value())
                else:
                    final_res = fitter.fit_circle(detected_pts)
            else:
                self.lbl_result.setText("拟合器缺少 fit_circle 方法")
                return
        except Exception as e:
            self.lbl_result.setText(f"拟合过程出错: {e}")
            return

        # 5. 结果显示
        r_min = min(r_start_scan, r_end_scan)
        r_max = max(r_start_scan, r_end_scan)
        roi_info = ((cx, cy), r_min, r_max)
        
        self.display_result_circle(final_res, detected_pts, roi_info)

    def measure_line_generic(self, edge_detector, fitter):
        if len(self.points) < 2: return
        p_start_base = np.array(self.points[0])
        p_end_base = np.array(self.points[-1])
        
        vec = p_end_base - p_start_base
        length = np.linalg.norm(vec)
        if length < 1: return
        unit_v = vec / length
        norm_v = np.array([-unit_v[1], unit_v[0]])
        
        p_scan_start, p_scan_end = self.scan_config["roi_points"]
        p_scan_start_arr = np.array(p_scan_start)
        p_scan_end_arr = np.array(p_scan_end)
        
        offset_s = np.dot(p_scan_start_arr - p_start_base, norm_v)
        offset_e = np.dot(p_scan_end_arr - p_start_base, norm_v)
        scan_width = abs(offset_e - offset_s)
        
        pixel_step = self.spin_step.value()
        if pixel_step < 0.5: pixel_step = 0.5
        
        detected_pts = []
        num_steps = int(length / pixel_step)
        
        for i in range(num_steps + 1):
            current_dist = i * pixel_step
            center_k = p_start_base + unit_v * current_dist
            
            p_s = center_k + norm_v * offset_s
            p_e = center_k + norm_v * offset_e
            
            try:
                pt = edge_detector.detect(self.cv_image, p_s, p_e, scan_width, bg_color=self.bg_color)
                if pt is not None:
                    detected_pts.append(pt)
            except Exception as e:
                continue

        if len(detected_pts) < 2:
             self.lbl_result.setText(f"检测点数不足 ({len(detected_pts)})")
             return

        try:
            if isinstance(fitter, fit_circle_robust):
                final_res = fitter.fit_line(detected_pts, outlier_threshold=self.spin_outlier.value())
            else:
                final_res = fitter.fit_line(detected_pts)
        except Exception as e:
            self.lbl_result.setText(f"拟合过程出错: {e}")
            return

        mid_offset = (offset_s + offset_e) / 2.0
        visual_p_s = p_start_base + norm_v * mid_offset
        visual_p_e = p_end_base + norm_v * mid_offset
        visual_width = abs(offset_e - offset_s)
        roi_info = ("line", visual_p_s, visual_p_e, visual_width, norm_v)
        
        self.display_result_line(final_res, detected_pts, roi_info)

    # ==========================================================================
    # 结果展示 Helper
    # ==========================================================================

    def display_result_circle(self, final_res, all_pts, roi_info):
        if final_res["success"]:
            # 分离有效点和离群点
            valid_pts = final_res.get("valid_points", all_pts) # LS可能不返回valid_points，默认为all
            
            # 如果拟合算法没返回 valid_points (比如普通 LS), 则所有点都是红色
            # 如果有返回 (Robust), 则区分颜色
            if "valid_points" in final_res:
                valid_set = set((float(p[0]), float(p[1])) for p in valid_pts)
                valid_list = []; outlier_list = []
                for pt in all_pts:
                    if (float(pt[0]), float(pt[1])) in valid_set:
                        valid_list.append(pt)
                    else:
                        outlier_list.append(pt)
            else:
                valid_list = all_pts
                outlier_list = []

            circle_par = (final_res["center"], final_res["radius"])
            self.view.draw_results(valid_list, circle_par, roi_info, outliers=outlier_list)
            
            scale = self.spin_scale.value()
            phys_d = final_res["radius"] * 2.0 * scale
            phys_circ = final_res.get("circularity", 0.0) * scale
            
            self.lbl_result.setText(
                f"【{self.combo_algo.currentText()}】\n"
                f"总检测点: {len(all_pts)}\n"
                f"有效点: {len(valid_list)} | 离群点: {len(outlier_list)}\n"
                f"物理直径: {phys_d:.4f} mm\n"
                f"圆度误差: {phys_circ:.4f} mm\n"
                f"圆心: ({final_res['center'][0]:.1f}, {final_res['center'][1]:.1f})"
            )
        else:
            self.lbl_result.setText(f"拟合失败: {final_res.get('msg')}")

    def display_result_line(self, final_res, all_pts, roi_info):
        if final_res["success"]:
            valid_pts = final_res.get("valid_points", all_pts)
            
            if "valid_points" in final_res:
                valid_set = set((float(p[0]), float(p[1])) for p in valid_pts)
                valid_list = []; outlier_list = []
                for pt in all_pts:
                    if (float(pt[0]), float(pt[1])) in valid_set: valid_list.append(pt)
                    else: outlier_list.append(pt)
            else:
                valid_list = all_pts; outlier_list = []
            
            line_res = (final_res["line_point"], final_res["line_dir"])
            self.view.draw_results(valid_list, line_res, roi_info, outliers=outlier_list)
            
            scale = self.spin_scale.value()
            phys_s = final_res.get("straightness", 0.0) * scale
            
            # 计算长度
            vx, vy = final_res["line_dir"]
            # 使用 numpy 确保操作正确
            v_pts_arr = np.array(valid_list)
            if len(v_pts_arr) > 0:
                projections = v_pts_arr[:, 0] * vx + v_pts_arr[:, 1] * vy
                pixel_len = np.max(projections) - np.min(projections)
            else:
                pixel_len = 0
            phys_len = pixel_len * scale

            self.lbl_result.setText(
                f"【{self.combo_algo.currentText()}】\n"
                f"总检测点: {len(all_pts)}\n"
                f"有效点: {len(valid_list)} | 离群点: {len(outlier_list)}\n"
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