# 融合代码1和代码3的版本
# smart_edge_metrology.py
# 合并版：UI(from code1) + 鲁棒测量(from code2) + 直线度（基于采集点的鲁棒拟合，选项 B）
import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QSpinBox,
                             QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
                             QGraphicsEllipseItem, QGraphicsLineItem, QFileDialog,
                             QDoubleSpinBox, QGroupBox, QRadioButton, QMessageBox)
from PyQt5.QtCore import Qt, pyqtSignal, QPointF, QLineF
from PyQt5.QtGui import QPixmap, QImage, QPen, QColor, QBrush, QPolygonF, QPainter

# ==============================================================================
# 1. Measure Utilities (融合并扩展)
# ==============================================================================

class MeasureUtils:
    @staticmethod
    def get_profile_subpix(image, p1, p2, width_px):
        """
        从 p1 到 p2 提取一维剖面（双线性插值）
        返回 1D numpy 数组
        """
        x1, y1 = p1
        x2, y2 = p2
        dx, dy = x2 - x1, y2 - y1
        length = np.hypot(dx, dy)
        num_samples = int(np.ceil(length))
        if num_samples < 2:
            num_samples = 2
        x_coords = np.linspace(x1, x2, num_samples).astype(np.float32)
        y_coords = np.linspace(y1, y2, num_samples).astype(np.float32)
        map_x = x_coords.reshape(1, -1)
        map_y = y_coords.reshape(1, -1)
        profile = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return profile[0, :]

    @staticmethod
    def get_projection_profile(image, p_start, p_end, width_px, thickness=3):
        """
        投影波形（沿中线多条平行线平均），返回一维 profile（长度 = ceil(width_px)）
        thickness: 在扫描方向垂直的小厚度（用于稳定）
        """
        x1, y1 = p_start
        x2, y2 = p_end
        dx, dy = x2 - x1, y2 - y1
        length = np.hypot(dx, dy)
        if length < 1e-6:
            return np.zeros(int(np.ceil(width_px)))
        ux, uy = dx / length, dy / length
        nx, ny = -uy, ux  # 法向（垂直于主方向）
        num_samples = int(np.ceil(width_px))
        if num_samples < 2:
            num_samples = 2
        offsets = np.linspace(-(thickness-1)/2, (thickness-1)/2, thickness)
        accumulated = None
        for off in offsets:
            sx = x1 + nx * off
            sy = y1 + ny * off
            ex = x2 + nx * off
            ey = y2 + ny * off
            x_coords = np.linspace(sx, ex, num_samples).astype(np.float32)
            y_coords = np.linspace(sy, ey, num_samples).astype(np.float32)
            map_x = x_coords.reshape(1, -1)
            map_y = y_coords.reshape(1, -1)
            prof = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)[0, :]
            if accumulated is None:
                accumulated = prof
            else:
                accumulated += prof
        return accumulated / float(len(offsets))

    @staticmethod
    def measure_1d_edge_directional(profile, threshold=5.0):
        """
        投影波形法在1D上寻找第一个显著边缘（方向由用户起终点决定）
        返回 sub-pixel index 或 None
        """
        if profile is None or len(profile) < 5:
            return None
        kernel = np.array([0.25, 0.5, 0.25])
        smooth = np.convolve(profile, kernel, mode='same')
        grads = np.gradient(smooth)
        absg = np.abs(grads)
        candidates = []
        for i in range(2, len(absg)-2):
            g = absg[i]
            if g > threshold and g >= absg[i-1] and g >= absg[i+1]:
                candidates.append(i)
        if not candidates:
            return None
        best = candidates[0]
        # 亚像素抛物线拟合
        y1 = absg[best-1]; y2 = absg[best]; y3 = absg[best+1]
        denom = (y1 - 2*y2 + y3)
        if denom == 0:
            return float(best)
        delta = (y1 - y3) / (2 * denom)
        return best + delta

    @staticmethod
    def fit_circle_basic(points):
        if points is None or len(points) < 3:
            return (None, None)
        pts = np.array(points)
        X = pts[:, 0].reshape(-1, 1)
        Y = pts[:, 1].reshape(-1, 1)
        A = np.hstack((2*X, 2*Y, np.ones_like(X)))
        B = X**2 + Y**2
        try:
            res, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
            cx, cy, cterm = res[0][0], res[1][0], res[2][0]
            radius = np.sqrt(cterm + cx**2 + cy**2)
            return (cx, cy), radius
        except Exception:
            return (None, None)

    @staticmethod
    def filter_and_eval_circle(points, outlier_threshold=1.0):
        """
        鲁棒圆度：初拟合 -> 误差过滤 -> 重拟合 -> 计算圆度 (px)
        返回 dict: success, center, radius, circularity, valid_points, outlier_count 或失败信息
        """
        pts = np.array(points)
        if pts.shape[0] < 3:
            return {"success": False, "msg": "点数不足3"}
        (cx, cy), r = MeasureUtils.fit_circle_basic(pts)
        if cx is None:
            return {"success": False, "msg": "初拟合失败"}
        dists = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
        errors = np.abs(dists - r)
        mask = errors < outlier_threshold
        valid = pts[mask]
        if valid.shape[0] < 3:
            return {"success": False, "msg": "过滤后有效点不足3，请增大阈值"}
        (fcx, fcy), fr = MeasureUtils.fit_circle_basic(valid)
        new_dists = np.hypot(valid[:, 0] - fcx, valid[:, 1] - fcy)
        circularity = new_dists.max() - new_dists.min()
        return {
            "success": True,
            "center": (fcx, fcy),
            "radius": fr,
            "circularity": float(circularity),
            "valid_points": valid,
            "outlier_count": int(pts.shape[0] - valid.shape[0])
        }

    @staticmethod
    def fit_line(points):
        """ 使用 cv2.fitLine 返回: (point_on_line), (direction_vector) """
        pts = np.array(points)
        if pts.shape[0] < 2:
            return None, None
        vx, vy, x0, y0 = cv2.fitLine(pts.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        return (float(x0), float(y0)), (float(vx), float(vy))

    @staticmethod
    def filter_and_eval_line(points, outlier_threshold=1.0):
        """
        鲁棒直线：基于采集点（你选择的 B 策略）
        流程：
          1. 初拟合直线
          2. 计算点到直线的有符号垂直距离
          3. 根据 outlier_threshold 过滤（|dist - median_dist| < outlier_threshold 或直接 |dist| < threshold）
          4. 重拟合并计算直线度 = max(dist) - min(dist)
        返回 dict 包含 success, line (point,dvec), straightness(px), valid_points, outlier_count
        """
        pts = np.array(points)
        if pts.shape[0] < 2:
            return {"success": False, "msg": "点数不足2"}
        p0, d = MeasureUtils.fit_line(pts)
        if p0 is None:
            return {"success": False, "msg": "拟合失败"}
        vx, vy = d
        # 单位方向
        norm = np.hypot(vx, vy)
        if norm < 1e-8:
            return {"success": False, "msg": "拟合方向长度异常"}
        ux, uy = vx / norm, vy / norm
        # 计算点到线的有符号垂直距离
        #  d_signed = cross( (pt - p0) , u )  (since cross in 2D gives scalar)
        diffs = pts - np.array(p0)
        d_signed = diffs[:, 0] * (-uy) + diffs[:, 1] * (ux)   # equivalent to cross with unit dir
        # use absolute distances for filtering but preserve sign for straightness range
        absd = np.abs(d_signed)
        # filter by threshold
        mask = absd < outlier_threshold
        valid = pts[mask]
        if valid.shape[0] < 2:
            return {"success": False, "msg": "过滤后有效点不足2，增大阈值"}
        # 重拟合
        p0f, df = MeasureUtils.fit_line(valid)
        if p0f is None:
            return {"success": False, "msg": "重拟合失败"}
        vx2, vy2 = df
        norm2 = np.hypot(vx2, vy2)
        ux2, uy2 = vx2 / norm2, vy2 / norm2
        diffs2 = valid - np.array(p0f)
        dsigned2 = diffs2[:, 0] * (-uy2) + diffs2[:, 1] * (ux2)
        straightness = float(np.max(dsigned2) - np.min(dsigned2))  # px
        return {
            "success": True,
            "line_point": p0f,
            "line_dir": (ux2, uy2),
            "straightness": straightness,
            "valid_points": valid,
            "outlier_count": int(pts.shape[0] - valid.shape[0])
        }

# ==============================================================================
# 2. Interactive View (结合两份代码的交互体验)
# ==============================================================================

class InteractiveView(QGraphicsView):
    point_selected = pyqtSignal(float, float)
    direction_defined = pyqtSignal(float, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHint(QPainter.Antialiasing)
        self.setMouseTracking(True)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item = None
        self.mode = "VIEW"  # VIEW, PICK, DIRECTION
        self.temp_items = []
        self.result_items = []
        self._drag_start = None
        self._drag_line_item = None
        self._drag_arrow_item = None

    def load_image(self, cv_img):
        self._scene.clear()
        self.temp_items = []
        self.result_items = []
        h, w, c = cv_img.shape
        q_img = QImage(cv_img.data, w, h, 3*w, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(q_img)
        self._pixmap_item = QGraphicsPixmapItem(pixmap)
        self._scene.addItem(self._pixmap_item)
        self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)

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

    def _get_arrow_polygon(self, p_start, p_end):
        dx = p_start.x() - p_end.x()
        dy = p_start.y() - p_end.y()
        length = np.hypot(dx, dy)
        if length < 1e-6:
            return QPolygonF()
        arrow_size = min(15, length * 0.3)
        angle = np.arctan2(dy, dx)
        p1 = QPointF(p_end.x() + arrow_size * np.cos(angle + 0.5),
                     p_end.y() + arrow_size * np.sin(angle + 0.5))
        p2 = QPointF(p_end.x() + arrow_size * np.cos(angle - 0.5),
                     p_end.y() + arrow_size * np.sin(angle - 0.5))
        return QPolygonF([p_end, p1, p2])

    def draw_scan_line(self, p1, p2):
        line = QGraphicsLineItem(p1[0], p1[1], p2[0], p2[1])
        pen = QPen(QColor(200,200,200,120), 1)
        line.setPen(pen)
        self._scene.addItem(line)
        self.result_items.append(line)

    def draw_detected_point(self, x, y, is_valid=True):
        r = 3 if is_valid else 2
        color = Qt.yellow if is_valid else Qt.gray
        dot = QGraphicsEllipseItem(x-r, y-r, r*2, r*2)
        dot.setPen(QPen(color, 1))
        dot.setBrush(QBrush(color))
        self._scene.addItem(dot)
        self.result_items.append(dot)

    def draw_fitted_circle(self, center, radius):
        cx, cy = center
        circ = QGraphicsEllipseItem(cx-radius, cy-radius, radius*2, radius*2)
        circ.setPen(QPen(Qt.blue, 2))
        self._scene.addItem(circ)
        self.result_items.append(circ)
        # draw center cross
        cl = 8
        l1 = QGraphicsLineItem(cx-cl, cy, cx+cl, cy)
        l2 = QGraphicsLineItem(cx, cy-cl, cx, cy+cl)
        l1.setPen(QPen(Qt.yellow,2)); l2.setPen(QPen(Qt.yellow,2))
        self._scene.addItem(l1); self._scene.addItem(l2)
        self.result_items.extend([l1,l2])

    def draw_fitted_line(self, line_point, line_dir, length=2000):
        x0, y0 = line_point
        ux, uy = line_dir
        # produce a long segment for visualization
        p1 = (x0 - ux*length, y0 - uy*length)
        p2 = (x0 + ux*length, y0 + uy*length)
        li = QGraphicsLineItem(p1[0], p1[1], p2[0], p2[1])
        li.setPen(QPen(Qt.blue, 2))
        self._scene.addItem(li)
        self.result_items.append(li)

    # Events
    def wheelEvent(self, event):
        zoom_in = 1.15
        zoom_out = 1.0/zoom_in
        factor = zoom_in if event.angleDelta().y() > 0 else zoom_out
        old = self.mapToScene(event.pos())
        self.scale(factor, factor)
        new = self.mapToScene(event.pos())
        delta = new - old
        self.translate(delta.x(), delta.y())

    def mousePressEvent(self, event):
        sp = self.mapToScene(event.pos())
        if self.mode == "PICK" and event.button() == Qt.LeftButton:
            r = 4
            dot = self._scene.addEllipse(sp.x()-r, sp.y()-r, r*2, r*2, QPen(Qt.red), QBrush(Qt.red))
            self.temp_items.append(dot)
            self.point_selected.emit(sp.x(), sp.y())
        elif self.mode == "DIRECTION" and event.button() == Qt.LeftButton:
            self._drag_start = sp
            self._drag_line_item = QGraphicsLineItem(sp.x(), sp.y(), sp.x(), sp.y())
            pen = QPen(Qt.cyan, 2)
            pen.setCosmetic(True)
            self._drag_line_item.setPen(pen)
            self._scene.addItem(self._drag_line_item)
            self._drag_arrow_item = self._scene.addPolygon(QPolygonF(), QPen(Qt.cyan), QBrush(Qt.cyan))
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        sp = self.mapToScene(event.pos())
        if self.mode == "DIRECTION" and self._drag_start:
            self._drag_line_item.setLine(QLineF(self._drag_start, sp))
            poly = self._get_arrow_polygon(self._drag_start, sp)
            self._drag_arrow_item.setPolygon(poly)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        sp = self.mapToScene(event.pos())
        if self.mode == "DIRECTION" and self._drag_start:
            p1 = (self._drag_start.x(), self._drag_start.y())
            p2 = (sp.x(), sp.y())
            length = np.hypot(p2[0]-p1[0], p2[1]-p1[1])
            if length > 2:
                self.temp_items.append(self._drag_line_item)
                self.temp_items.append(self._drag_arrow_item)
                self.direction_defined.emit(length, (p1, p2))
            else:
                try:
                    self._scene.removeItem(self._drag_line_item)
                    self._scene.removeItem(self._drag_arrow_item)
                except: pass
            self._drag_start = None
            self._drag_line_item = None
            self._drag_arrow_item = None

# ==============================================================================
# 3. Main Window 
# ==============================================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("边缘检测demo")
        self.resize(1300, 850)
        self.cv_image = None
        self.gray_image = None
        self.points = []
        self.scan_config = {"width": 40.0, "vector": None}
        self.setup_ui()

    def setup_ui(self):
        main = QWidget()
        self.setCentralWidget(main)
        h = QHBoxLayout(main)

        # Left panel
        panel = QWidget(); panel.setFixedWidth(320)
        vbox = QVBoxLayout(panel)

        # Image load
        vbox.addWidget(QLabel("<b>1. 图像加载 / Image</b>"))
        btn_load = QPushButton("打开图片 / Open Image")
        btn_load.clicked.connect(self.load_image)
        vbox.addWidget(btn_load)

        # Mode selection
        vbox.addSpacing(6)
        vbox.addWidget(QLabel("<b>2. 测量模式 / Mode</b>"))
        self.rb_circle = QRadioButton("圆形测量 (Circle)")
        self.rb_line = QRadioButton("直线测量 (Line)")
        self.rb_circle.setChecked(True)
        vbox.addWidget(self.rb_circle); vbox.addWidget(self.rb_line)

        # Parameters
        vbox.addSpacing(6)
        row0 = QHBoxLayout(); row0.addWidget(QLabel("定点数:"))
        self.spin_pts = QSpinBox(); self.spin_pts.setRange(2, 40); self.spin_pts.setValue(3)
        row0.addWidget(self.spin_pts); vbox.addLayout(row0)

        row_scale = QHBoxLayout(); row_scale.addWidget(QLabel("比例 (mm/px):"))
        self.spin_scale = QDoubleSpinBox(); self.spin_scale.setDecimals(4); self.spin_scale.setValue(0.0100)
        row_scale.addWidget(self.spin_scale); vbox.addLayout(row_scale)

        row_thresh = QHBoxLayout(); row_thresh.addWidget(QLabel("梯度阈值:"))
        self.spin_thresh = QDoubleSpinBox(); self.spin_thresh.setRange(0.1, 200.0); self.spin_thresh.setValue(10.0)
        row_thresh.addWidget(self.spin_thresh); vbox.addLayout(row_thresh)

        # Outlier threshold (from code2)
        row_out = QHBoxLayout(); row_out.addWidget(QLabel("剔除阈值(px):"))
        self.spin_outlier = QDoubleSpinBox(); self.spin_outlier.setRange(0.1, 100.0); self.spin_outlier.setValue(2.0)
        row_out.addWidget(self.spin_outlier); vbox.addLayout(row_out)
        vbox.addWidget(QLabel("<small style='color:gray'>超出该像素偏差将被视为异常点并剔除（鲁棒拟合）</small>"))

        vbox.addSpacing(10)
        vbox.addWidget(QLabel("<b>3. 操作流程 / Flow</b>"))
        self.btn_pick = QPushButton("Step A: 选取轮廓点 / Pick Points")
        self.btn_pick.clicked.connect(self.start_pick_points)
        vbox.addWidget(self.btn_pick)

        self.lbl_hint_pick = QLabel("未开始")
        self.lbl_hint_pick.setStyleSheet("color: gray")
        vbox.addWidget(self.lbl_hint_pick)

        self.btn_dir = QPushButton("Step B: 绘制扫描方向 / Draw Direction")
        self.btn_dir.setEnabled(False)
        self.btn_dir.clicked.connect(self.start_draw_direction)
        vbox.addWidget(self.btn_dir)

        self.lbl_hint_dir = QLabel("需先选点")
        self.lbl_hint_dir.setStyleSheet("color: gray")
        vbox.addWidget(self.lbl_hint_dir)

        self.btn_calc = QPushButton("Step C: 计算并显示 / Compute")
        self.btn_calc.setEnabled(False)
        self.btn_calc.setStyleSheet("background-color:#d0f0c0; font-weight:bold; height:36px")
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
        h.addWidget(self.view)

    # -----------------------
    # Image & state helpers
    # -----------------------
    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.bmp *.tif)")
        if not path:
            return
        self.cv_image = cv2.imread(path)
        if self.cv_image is None:
            QMessageBox.warning(self, "错误", "无法加载图片")
            return
        self.gray_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)
        self.view.load_image(self.cv_image)
        self.reset_state()

    def reset_state(self):
        self.points = []
        self.scan_config = {"width": 40.0, "vector": None}
        self.view.clear_overlays(); self.view.clear_results()
        self.view.mode = "VIEW"
        self.btn_dir.setEnabled(False); self.btn_calc.setEnabled(False)
        self.lbl_hint_pick.setText("请点击图片选点...")
        self.lbl_hint_dir.setText("等待选点完成")
        self.lbl_result.setText("")

    # -----------------------
    # Step A: pick points
    # -----------------------
    def start_pick_points(self):
        if self.cv_image is None:
            QMessageBox.information(self, "提示", "请先加载图像")
            return
        self.points = []
        self.view.clear_overlays(); self.view.clear_results()
        self.view.mode = "PICK"
        self.lbl_hint_pick.setText("正在选点... (滚轮缩放)")
        self.btn_dir.setEnabled(False); self.btn_calc.setEnabled(False)

    def on_point_picked(self, x, y):
        self.points.append((x, y))
        target = self.spin_pts.value() if self.rb_circle.isChecked() else 2
        self.lbl_hint_pick.setText(f"已选 {len(self.points)} / {target}")
        # 自动结束
        if len(self.points) >= target:
            self.view.mode = "VIEW"
            self.lbl_hint_pick.setText("选点完成!")
            self.btn_dir.setEnabled(True)
            self.lbl_hint_dir.setText("请点击下方按钮开始绘制方向")
            # preliminary draw
            if self.rb_circle.isChecked() and len(self.points) >= 3:
                c, r = MeasureUtils.fit_circle_basic(self.points)
                if c and r:
                    circ = QGraphicsEllipseItem(c[0]-r, c[1]-r, r*2, r*2)
                    circ.setPen(QPen(Qt.yellow, 1, Qt.DashLine))
                    self.view._scene.addItem(circ); self.view.temp_items.append(circ)
            elif self.rb_line.isChecked() and len(self.points) >= 2:
                p1, p2 = self.points[0], self.points[-1]
                li = QGraphicsLineItem(p1[0], p1[1], p2[0], p2[1])
                li.setPen(QPen(Qt.yellow, 1, Qt.DashLine))
                self.view._scene.addItem(li); self.view.temp_items.append(li)

    # -----------------------
    # Step B: draw direction (used by scanning routines)
    # -----------------------
    def start_draw_direction(self):
        if len(self.points) == 0:
            QMessageBox.information(self, "提示", "请先选点")
            return
        self.view.mode = "DIRECTION"
        self.lbl_hint_dir.setText("在图上拖拽: 起点(背景)->终点(目标)")
        QMessageBox.information(self, "提示", "请在图像边缘处拖拽一条线段。\n\n1. 线段长度 = 搜索宽度(ROI)\n2. 箭头方向 = 梯度搜索方向\n建议：从背景指向目标边缘（背景->目标）")

    def on_direction_set(self, length, vector):
        self.scan_config["width"] = length
        self.scan_config["vector"] = vector
        self.view.mode = "VIEW"
        self.lbl_hint_dir.setText(f"方向已定: 宽度={length:.1f}px")
        self.btn_calc.setEnabled(True)

    # -----------------------
    # Step C: measurement
    # -----------------------
    def run_measurement(self):
        self.view.clear_results()
        if self.rb_circle.isChecked():
            self.measure_circle()
        else:
            self.measure_line()

    def measure_circle(self):
        # 基于用户选点 + 扫描方向进行投影测边并使用 filter_and_eval_circle
        if len(self.points) < 3:
            self.lbl_result.setText("请至少选取3个点用于圆测量")
            return
        (icx, icy), radius = MeasureUtils.fit_circle_basic(self.points)
        if icx is None:
            self.lbl_result.setText("初拟合失败")
            return
        if self.scan_config["vector"] is None:
            self.lbl_result.setText("请先绘制扫描方向 (Step B)")
            return
        (vx1, vy1), (vx2, vy2) = self.scan_config["vector"]
        scan_width = self.scan_config["width"]
        # 判断方向（向心/离心）
        mid_x, mid_y = (vx1+vx2)/2.0, (vy1+vy2)/2.0
        is_outward = np.dot([mid_x-icx, mid_y-icy], [vx2-vx1, vy2-vy1]) > 0

        detected_pts = []
        num_spokes = 360
        grad_thresh = self.spin_thresh.value()
        # 更节省时间：可选减少角度数，但默认 360  这里可以加超参数该成圆弧
        for i in range(num_spokes):
            ang = np.deg2rad(i)
            ca, sa = np.cos(ang), np.sin(ang)
            r_in = radius - scan_width/2.0
            r_out = radius + scan_width/2.0
            if is_outward:
                p_s = (icx + r_in*ca, icy + r_in*sa)
                p_e = (icx + r_out*ca, icy + r_out*sa)
            else:
                p_s = (icx + r_out*ca, icy + r_out*sa)
                p_e = (icx + r_in*ca, icy + r_in*sa)
            # optional: 每隔若干角度绘制扫描线用于调试
            if i % 30 == 0:
                self.view.draw_scan_line(p_s, p_e)
            prof = MeasureUtils.get_projection_profile(self.gray_image, p_s, p_e, scan_width, thickness=3)
            idx = MeasureUtils.measure_1d_edge_directional(prof, grad_thresh)
            if idx is not None:
                ratio = idx / float(len(prof)-1)
                px = p_s[0] + (p_e[0]-p_s[0]) * ratio
                py = p_s[1] + (p_e[1]-p_s[1]) * ratio
                detected_pts.append((px, py))
        # 调用鲁棒圆度计算
        outlier_px = self.spin_outlier.value()
        result = MeasureUtils.filter_and_eval_circle(detected_pts, outlier_px)
        if result and result.get("success"):
            fcx, fcy = result["center"]
            fr = result["radius"]
            # 绘制有效点
            valid_set = set([(float(pt[0]), float(pt[1])) for pt in result["valid_points"]])
            for pt in detected_pts:
                is_valid = (float(pt[0]), float(pt[1])) in valid_set
                if is_valid:
                    self.view.draw_detected_point(pt[0], pt[1], True)
            self.view.draw_fitted_circle((fcx, fcy), fr)
            scale = self.spin_scale.value()
            phys_d = fr * 2.0 * scale
            phys_circ = result["circularity"] * scale
            self.lbl_result.setText(
                f"【圆测量成功】\n"
                f"原始点数: {len(detected_pts)}\n"
                f"有效点数: {len(result['valid_points'])} (剔除 {result['outlier_count']})\n"
                f"物理直径: {phys_d:.4f} mm\n"
                f"物理圆度: {phys_circ:.4f} mm\n"
                f"圆心(Px): ({fcx:.1f}, {fcy:.1f})"
            )
        else:
            msg = result.get("msg", "未找到边缘") if result else "未找到边缘"
            self.lbl_result.setText(f"测量失败: {msg}")

    #     )
    def measure_line(self):
        # 基本检查
        if len(self.points) < 2:
            self.lbl_result.setText("请至少选取2个点用于直线测量")
            return
        if self.scan_config["vector"] is None:
            self.lbl_result.setText("请先绘制扫描方向 (Step B)")
            return
        
        p_start, p_end = np.array(self.points[0]), np.array(self.points[-1])
        scan_width = self.scan_config["width"]
        grad_thresh = self.spin_thresh.value()
        outlier_px = self.spin_outlier.value()

        # 计算方向向量和长度
        vec = p_end - p_start
        length = np.linalg.norm(vec)
        if length < 1e-6:
            self.lbl_result.setText("选取点距离过近")
            return
        unit_vec = vec / length
        normal_vec = np.array([-unit_vec[1], unit_vec[0]])  # 法向量

        # 端点长度扩张5%
        expansion_ratio = 0.05
        length_expanded = length * (1 + 2 * expansion_ratio)
        center = (p_start + p_end) / 2
        new_start = center - unit_vec * length_expanded / 2
        new_end = center + unit_vec * length_expanded / 2

        # 扫描线数量（按像素采样）
        num_lines = int(np.ceil(length_expanded))
        if num_lines < 2:
            num_lines = 2

        detected_pts = []
        self.view.clear_results()

        for i in range(num_lines):
            # 当前扫描线起止点，沿扩张后的方向插值
            cx = new_start[0] + unit_vec[0] * i
            cy = new_start[1] + unit_vec[1] * i
            p_s = (cx + normal_vec[0] * (-scan_width / 2), cy + normal_vec[1] * (-scan_width / 2))
            p_e = (cx + normal_vec[0] * (scan_width / 2), cy + normal_vec[1] * (scan_width / 2))

            if i % 10 == 0:
                self.view.draw_scan_line(p_s, p_e)

            prof = MeasureUtils.get_projection_profile(self.gray_image, p_s, p_e, scan_width, thickness=3)
            idx = MeasureUtils.measure_1d_edge_directional(prof, grad_thresh)
            if idx is not None:
                ratio = idx / float(len(prof) - 1)
                px = p_s[0] + (p_e[0] - p_s[0]) * ratio
                py = p_s[1] + (p_e[1] - p_s[1]) * ratio
                detected_pts.append((px, py))
                self.view.draw_detected_point(px, py, is_valid=False)

        if len(detected_pts) < 2:
            self.lbl_result.setText("未找到足够边缘点")
            return

        # 异常点剔除和拟合
        result = MeasureUtils.filter_and_eval_line(detected_pts, outlier_px)
        if not result or not result.get("success"):
            msg = result.get("msg", "拟合失败") if result else "拟合失败"
            self.lbl_result.setText(f"测量失败: {msg}")
            return

        valid_pts = result["valid_points"]
        valid_set = set([(float(pt[0]), float(pt[1])) for pt in valid_pts])

        # 更新点颜色
        for pt in detected_pts:
            is_valid = (float(pt[0]), float(pt[1])) in valid_set
            self.view.draw_detected_point(pt[0], pt[1], is_valid)

        # 计算最远两点距离（测量长度）
        valid_np = np.array(valid_pts)
        max_dist = 0
        pt1 = pt2 = None
        for i in range(len(valid_np)):
            for j in range(i + 1, len(valid_np)):
                dist = np.linalg.norm(valid_np[i] - valid_np[j])
                if dist > max_dist:
                    max_dist = dist
                    pt1, pt2 = valid_np[i], valid_np[j]

        # 绘制拟合直线
        self.view.draw_fitted_line(result["line_point"], result["line_dir"], length=4000)

        # 显示结果
        scale = self.spin_scale.value()
        straight_px = result["straightness"]
        straight_mm = straight_px * scale
        length_mm = max_dist * scale

        self.lbl_result.setText(
            f"【直线测量成功】\n"
            f"扫描线数: {num_lines}\n"
            f"检测边缘点数: {len(detected_pts)}\n"
            f"有效点数: {len(valid_pts)} (剔除 {result['outlier_count']})\n"
            f"线段物理长度: {length_mm:.3f}mm\n"
            f"物理直线度: {straight_mm:.4f} mm\n"
        )



# ==============================================================================
# 4. Run App
# ==============================================================================

if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
