'''
如果不拟合（即不通过最小二乘法找到圆心），是无法准确计算圆度和物理尺寸的。因为圆度的定义是基于某个圆心计算出的半径差
），如果圆心不准，圆度值会巨大且无意义。
工业上标准的做法不是“不拟合”，而是**“鲁棒拟合” (Robust Fitting)**。即：
先拟合一次，找到大致圆心。
过滤：根据你设定的“超参数”（误差阈值），剔除那些距离圆心过远或过近的噪点。
重拟合：用剩下的干净点，再算一次精准的圆心、直径和圆度。
这种方法完全可行，且是解决“偏离点”的标准方案。
'''

import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QSpinBox, 
                             QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, 
                             QGraphicsEllipseItem, QGraphicsLineItem, QFileDialog, 
                             QDoubleSpinBox, QGroupBox, QRadioButton, QMessageBox)
from PyQt5.QtCore import Qt, pyqtSignal, QPointF, QRectF, QLineF
from PyQt5.QtGui import QPixmap, QImage, QPen, QColor, QBrush, QPainter, QPolygonF

# ==============================================================================
# 1. 核心算法层 (增加过滤与圆度计算)
# ==============================================================================

class MeasureUtils:
    @staticmethod
    def get_projection_profile(image, p_start, p_end, width_px, thickness=3):
        """ (保持不变) 投影波形提取 """
        x1, y1 = p_start
        x2, y2 = p_end
        dx, dy = x2 - x1, y2 - y1
        length = np.sqrt(dx**2 + dy**2)
        if length < 1e-6: return np.zeros(int(width_px))
        
        ux, uy = dx/length, dy/length
        nx, ny = -uy, ux
        
        num_samples = int(np.ceil(width_px))
        if num_samples < 2: num_samples = 2
        
        offsets = np.linspace(-(thickness-1)/2, (thickness-1)/2, thickness)
        accumulated_profile = None
        
        for off in offsets:
            curr_start = (x1 + nx * off, y1 + ny * off)
            curr_end   = (x2 + nx * off, y2 + ny * off)
            
            x_coords = np.linspace(curr_start[0], curr_end[0], num_samples).astype(np.float32)
            y_coords = np.linspace(curr_start[1], curr_end[1], num_samples).astype(np.float32)
            
            map_x = x_coords.reshape(1, -1)
            map_y = y_coords.reshape(1, -1)
            
            line_profile = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            if accumulated_profile is None:
                accumulated_profile = line_profile[0, :]
            else:
                accumulated_profile += line_profile[0, :]
                
        return accumulated_profile / thickness

    @staticmethod
    def measure_1d_edge_directional(profile, threshold=5.0):
        """ (保持不变) 寻找第一个显著边缘 """
        if len(profile) < 5: return None
        kernel = np.array([0.25, 0.5, 0.25])
        smooth_profile = np.convolve(profile, kernel, mode='same')
        gradients = np.gradient(smooth_profile)
        abs_grads = np.abs(gradients)
        
        candidates = []
        for i in range(2, len(abs_grads) - 2):
            g = abs_grads[i]
            if g > threshold:
                if g >= abs_grads[i-1] and g >= abs_grads[i+1]:
                    candidates.append(i)
        
        if not candidates: return None
        best_idx = candidates[0] # 取第一个遇到的边缘
        
        y1 = abs_grads[best_idx - 1]
        y2 = abs_grads[best_idx]
        y3 = abs_grads[best_idx + 1]
        denom = (y1 - 2*y2 + y3)
        if denom == 0: return float(best_idx)
        delta = (y1 - y3) / (2 * denom)
        return best_idx + delta

    @staticmethod
    def fit_circle_basic(points):
        """ 基础拟合：只返回圆心和半径 """
        if len(points) < 3: return None, None
        pts = np.array(points)
        X = pts[:, 0].reshape(-1, 1)
        Y = pts[:, 1].reshape(-1, 1)
        A = np.hstack((2 * X, 2 * Y, np.ones_like(X)))
        B = X**2 + Y**2
        try:
            res, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
            cx, cy, c_term = res[0][0], res[1][0], res[2][0]
            radius = np.sqrt(c_term + cx**2 + cy**2)
            return (cx, cy), radius
        except:
            return None, None

    # --- 新增：带过滤和圆度计算的高级方法 ---
    @staticmethod
    def filter_and_eval_circle(points, outlier_threshold=1.0):
        """
        鲁棒测量逻辑：
        1. 初次拟合
        2. 计算所有点到圆心的距离误差
        3. 剔除误差 > outlier_threshold 的点
        4. 使用剩余点重拟合
        5. 计算圆度 (Max Radius - Min Radius)
        """
        pts = np.array(points)
        if len(pts) < 3: return None
        
        # 1. 初次拟合
        (cx, cy), radius = MeasureUtils.fit_circle_basic(pts)
        if cx is None: return None
        
        # 2. 计算每个点到圆心的距离
        # dists = sqrt((x-cx)^2 + (y-cy)^2)
        dists = np.sqrt((pts[:, 0] - cx)**2 + (pts[:, 1] - cy)**2)
        
        # 3. 计算偏差并过滤
        errors = np.abs(dists - radius)
        valid_mask = errors < outlier_threshold
        
        valid_points = pts[valid_mask]
        
        if len(valid_points) < 3:
            return {
                "success": False, 
                "msg": "过滤后有效点不足3个，请增大阈值"
            }
        
        # 4. 重拟合 (为了获得更准确的圆心，排除噪点影响)
        (final_cx, final_cy), final_radius = MeasureUtils.fit_circle_basic(valid_points)
        
        # 5. 计算圆度 (Circularity)
        # 圆度通常定义为：包容实际轮廓的两个同心圆的半径差
        # 在这里我们基于重拟合后的圆心计算所有有效点的半径
        new_dists = np.sqrt((valid_points[:, 0] - final_cx)**2 + (valid_points[:, 1] - final_cy)**2)
        max_r = np.max(new_dists)
        min_r = np.min(new_dists)
        circularity = max_r - min_r
        
        return {
            "success": True,
            "center": (final_cx, final_cy),
            "radius": final_radius,
            "circularity": circularity,
            "valid_points": valid_points, # 返回有效点用于绘图
            "outlier_count": len(pts) - len(valid_points)
        }

    @staticmethod
    def fit_line(points):
        """ 最小二乘法拟合直线 """
        if len(points) < 2: return None, None
        pts = np.array(points)
        vx, vy, x, y = cv2.fitLine(pts.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01)
        return (x[0], y[0]), (vx[0], vy[0])

# ==============================================================================
# 2. 交互视图层 (保持不变，省略部分重复代码)
# ==============================================================================

class InteractiveView(QGraphicsView):
    point_selected = pyqtSignal(float, float)     
    direction_defined = pyqtSignal(float, object) 

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setRenderHint(QPainter.Antialiasing)
        self.setMouseTracking(True)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item = None
        self.mode = "VIEW" 
        self.temp_items = []
        self.result_items = []
        self._drag_start = None
        self._drag_line_item = None

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

    def clear_temp(self):
        for item in self.temp_items: self._scene.removeItem(item)
        self.temp_items = []

    def clear_results(self):
        for item in self.result_items: self._scene.removeItem(item)
        self.result_items = []

    def draw_scan_line(self, p1, p2):
        line = QGraphicsLineItem(p1[0], p1[1], p2[0], p2[1])
        line.setPen(QPen(QColor(200, 200, 200, 100), 1))
        self._scene.addItem(line)
        self.result_items.append(line)

    def draw_detected_point(self, x, y, is_valid=True):
        r = 3 if is_valid else 2
        color = Qt.green if is_valid else Qt.gray
        dot = QGraphicsEllipseItem(x-r, y-r, r*2, r*2)
        dot.setPen(QPen(color, 1))
        dot.setBrush(QBrush(color))
        self._scene.addItem(dot)
        self.result_items.append(dot)

    def draw_fitted_result(self, center, radius):
        cx, cy = center
        circle = QGraphicsEllipseItem(cx-radius, cy-radius, radius*2, radius*2)
        circle.setPen(QPen(Qt.blue, 2))
        self._scene.addItem(circle)
        self.result_items.append(circle)
        
        # 绘制圆心
        cross_len = 10
        l1 = QGraphicsLineItem(cx-cross_len, cy, cx+cross_len, cy)
        l2 = QGraphicsLineItem(cx, cy-cross_len, cx, cy+cross_len)
        l1.setPen(QPen(Qt.yellow, 2)); l2.setPen(QPen(Qt.yellow, 2))
        self._scene.addItem(l1); self._scene.addItem(l2)
        self.result_items.extend([l1, l2])

    # ... (WheelEvent, MouseEvents 保持原样) ...
    def wheelEvent(self, event):
        zoom_in = 1.15
        zoom_out = 1.0 / zoom_in
        factor = zoom_in if event.angleDelta().y() > 0 else zoom_out
        self.scale(factor, factor)
    
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
            self._drag_line_item.setPen(QPen(Qt.cyan, 2, Qt.SolidLine))
            self._scene.addItem(self._drag_line_item)
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        if self.mode == "DIRECTION" and self._drag_start:
            self._drag_line_item.setLine(QLineF(self._drag_start, self.mapToScene(event.pos())))

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        sp = self.mapToScene(event.pos())
        if self.mode == "DIRECTION" and self._drag_start:
            p1 = (self._drag_start.x(), self._drag_start.y())
            p2 = (sp.x(), sp.y())
            length = np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
            if length > 2:
                # 简单画个箭头
                arrow_poly = QPolygonF([QPointF(*p2), QPointF(p2[0]-5, p2[1]-5), QPointF(p2[0]+5, p2[1]-5)]) # 简化箭头
                self.temp_items.append(self._drag_line_item)
                self.direction_defined.emit(length, (p1, p2))
            else:
                self._scene.removeItem(self._drag_line_item)
            self._drag_start = None

# ==============================================================================
# 3. 主界面逻辑 (添加剔除参数输入)
# ==============================================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("高精度边缘测量 (剔除异常点版)")
        self.resize(1300, 850)
        
        self.cv_image = None
        self.gray_image = None
        self.points = []
        self.scan_config = {"width": 40.0, "vector": None}
        
        self.setup_ui()

    def setup_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        
        # --- 左侧面板 ---
        panel = QWidget()
        panel.setFixedWidth(280)
        vbox = QVBoxLayout(panel)
        
        btn_load = QPushButton("加载图像")
        btn_load.clicked.connect(self.load_image)
        vbox.addWidget(btn_load)
        
        self.rb_circle = QRadioButton("圆形测量"); self.rb_circle.setChecked(True)
        self.rb_line = QRadioButton("直线测量")
        vbox.addWidget(self.rb_circle); vbox.addWidget(self.rb_line)

        # 参数区
        grp = QGroupBox("参数设置")
        form = QVBoxLayout()
        
        # 初始点数
        row0 = QHBoxLayout(); row0.addWidget(QLabel("初始点数:")); 
        self.spin_pts = QSpinBox(); self.spin_pts.setValue(3); row0.addWidget(self.spin_pts)
        form.addLayout(row0)
        
        # 剔除阈值 (关键新增)
        row1 = QHBoxLayout(); row1.addWidget(QLabel("剔除阈值(px):"))
        self.spin_outlier = QDoubleSpinBox(); self.spin_outlier.setValue(2.0) # 默认允许2像素误差
        self.spin_outlier.setRange(0.1, 50.0); self.spin_outlier.setSingleStep(0.5)
        row1.addWidget(self.spin_outlier)
        form.addLayout(row1)
        form.addWidget(QLabel("<small style='color:gray'>超过此距离的点将被视为噪点剔除</small>"))

        # 梯度阈值
        row2 = QHBoxLayout(); row2.addWidget(QLabel("梯度阈值:"))
        self.spin_thresh = QDoubleSpinBox(); self.spin_thresh.setValue(5.0); row2.addWidget(self.spin_thresh)
        form.addLayout(row2)
        
        # 物理比例
        row3 = QHBoxLayout(); row3.addWidget(QLabel("比例(mm/px):"))
        self.spin_scale = QDoubleSpinBox(); self.spin_scale.setDecimals(4); self.spin_scale.setValue(0.01); row3.addWidget(self.spin_scale)
        form.addLayout(row3)

        grp.setLayout(form)
        vbox.addWidget(grp)

        # 按钮
        self.btn_pick = QPushButton("Step A: 选点")
        self.btn_pick.clicked.connect(self.start_pick_points)
        vbox.addWidget(self.btn_pick)
        
        self.btn_dir = QPushButton("Step B: 绘制方向")
        self.btn_dir.setEnabled(False)
        self.btn_dir.clicked.connect(self.start_draw_direction)
        vbox.addWidget(self.btn_dir)
        
        self.btn_calc = QPushButton("Step C: 计算")
        self.btn_calc.setStyleSheet("background-color: #d0f0c0; height: 40px; font-weight: bold")
        self.btn_calc.setEnabled(False)
        self.btn_calc.clicked.connect(self.run_measurement)
        vbox.addWidget(self.btn_calc)
        
        self.lbl_result = QLabel("结果...")
        self.lbl_result.setFrameShape(QLabel.Panel)
        self.lbl_result.setWordWrap(True)
        vbox.addWidget(self.lbl_result)
        vbox.addStretch()
        
        self.view = InteractiveView()
        self.view.point_selected.connect(self.on_point_picked)
        self.view.direction_defined.connect(self.on_direction_set)
        
        layout.addWidget(panel)
        layout.addWidget(self.view)

    # ... (Load Image, Pick Points, Draw Direction 逻辑与之前相同，省略重复) ...
    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open", "", "Images (*.png *.jpg *.bmp)")
        if path:
            self.cv_image = cv2.imread(path)
            self.gray_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)
            self.view.load_image(self.cv_image)
            self.points = []
            self.view.clear_temp(); self.view.clear_results()
            self.btn_calc.setEnabled(False); self.btn_dir.setEnabled(False)

    def start_pick_points(self):
        self.points = []
        self.view.clear_temp(); self.view.clear_results()
        self.view.mode = "PICK"

    def on_point_picked(self, x, y):
        self.points.append((x, y))
        target = self.spin_pts.value() if self.rb_circle.isChecked() else 2
        if len(self.points) >= target:
            self.view.mode = "VIEW"
            self.btn_dir.setEnabled(True)
            # 画初始猜测
            if self.rb_circle.isChecked():
                (cx, cy), r = MeasureUtils.fit_circle_basic(self.points)
                if cx: 
                    it = QGraphicsEllipseItem(cx-r, cy-r, r*2, r*2)
                    it.setPen(QPen(Qt.yellow, 1, Qt.DashLine))
                    self.view._scene.addItem(it); self.view.temp_items.append(it)

    def start_draw_direction(self):
        self.view.mode = "DIRECTION"

    def on_direction_set(self, length, vector):
        self.scan_config["width"] = length
        self.scan_config["vector"] = vector
        self.view.mode = "VIEW"
        self.btn_calc.setEnabled(True)

    def run_measurement(self):
        self.view.clear_results()
        if self.rb_circle.isChecked():
            self.measure_circle()
        else:
            self.lbl_result.setText("直线逻辑同理，此处略。")

    def measure_circle(self):
        # 1. 基础几何
        (icx, icy), radius = MeasureUtils.fit_circle_basic(self.points)
        if icx is None: return
        
        (vx1, vy1), (vx2, vy2) = self.scan_config["vector"]
        scan_width = self.scan_config["width"]
        
        # 判断方向
        mid_x, mid_y = (vx1+vx2)/2, (vy1+vy2)/2
        is_outward = np.dot([mid_x-icx, mid_y-icy], [vx2-vx1, vy2-vy1]) > 0
        
        detected_pts = []
        num_spokes = 360 # 扫描密度
        
        for i in range(num_spokes):
            angle = np.deg2rad(i)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            
            r_in, r_out = radius - scan_width/2, radius + scan_width/2
            if is_outward: p_s, p_e = (icx+r_in*cos_a, icy+r_in*sin_a), (icx+r_out*cos_a, icy+r_out*sin_a)
            else:          p_s, p_e = (icx+r_out*cos_a, icy+r_out*sin_a), (icx+r_in*cos_a, icy+r_in*sin_a)
            
            # 只有当需要调试时才画扫描线，否则太乱
            if i % 20 == 0: self.view.draw_scan_line(p_s, p_e)
            
            prof = MeasureUtils.get_projection_profile(self.gray_image, p_s, p_e, scan_width)
            idx = MeasureUtils.measure_1d_edge_directional(prof, self.spin_thresh.value())
            
            if idx is not None:
                ratio = idx / (len(prof)-1)
                px = p_s[0] + (p_e[0]-p_s[0])*ratio
                py = p_s[1] + (p_e[1]-p_s[1])*ratio
                detected_pts.append((px, py))

        # 2. 核心：调用带过滤的计算逻辑
        outlier_thresh = self.spin_outlier.value()
        result = MeasureUtils.filter_and_eval_circle(detected_pts, outlier_thresh)
        
        if result and result["success"]:
            fcx, fcy = result["center"]
            fr = result["radius"]
            
            # 仅绘制有效点
            valid_pts_set = set([(pt[0], pt[1]) for pt in result["valid_points"]])
            for pt in detected_pts:
                # 如果点在有效集中，画绿色；否则(被剔除的)不画或者画灰色
                is_valid = (pt[0], pt[1]) in valid_pts_set
                if is_valid: # 只画有效点，保持界面干净
                    self.view.draw_detected_point(pt[0], pt[1], True)
            
            self.view.draw_fitted_result((fcx, fcy), fr)
            
            # 物理转换
            scale = self.spin_scale.value()
            phys_d = fr * 2 * scale
            phys_circ = result["circularity"] * scale
            
            self.lbl_result.setText(
                f"【测量成功】\n"
                f"原始点数: {len(detected_pts)}\n"
                f"有效点数: {len(result['valid_points'])} (剔除 {result['outlier_count']})\n"
                f"----------------\n"
                f"物理直径: {phys_d:.4f} mm\n"
                f"物理圆度: {phys_circ:.4f} mm\n"
                f"----------------\n"
                f"圆心(Px): ({fcx:.1f}, {fcy:.1f})"
            )
        else:
            msg = result["msg"] if result else "未找到边缘"
            self.lbl_result.setText(f"测量失败: {msg}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())