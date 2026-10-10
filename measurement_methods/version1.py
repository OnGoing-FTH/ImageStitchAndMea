'''
使用扇形roi估计和投影波形法测点，测的点有点少，而且使用拟合方式计算圆尺寸。
如果降低点阈值可能会引入过多的误差点，如果强制拟合就意义不大。


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
# 1. 核心算法层 (Math & Algorithms)
# ==============================================================================

class MeasureUtils:
    @staticmethod
    def get_profile_subpix(image, p1, p2, width_px):
        """
        提取 p1 到 p2 线段上的图像剖面 (支持亚像素采样)
        """
        # 生成采样坐标
        num_samples = int(np.ceil(width_px))
        if num_samples < 2: num_samples = 2
        
        x_coords = np.linspace(p1[0], p2[0], num_samples).astype(np.float32)
        y_coords = np.linspace(p1[1], p2[1], num_samples).astype(np.float32)
        
        # 使用 cv2.remap 进行双线性插值采样
        # map_x, map_y 必须是 float32
        #以此构建 1xN 的映射矩阵
        map_x = x_coords.reshape(1, -1)
        map_y = y_coords.reshape(1, -1)
        
        # 采样结果
        profile = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return profile[0, :] # 返回一维数组

    @staticmethod
    def measure_1d_edge_directional(profile, threshold=5.0):
        """
        投影波形法：沿数组方向寻找边缘。
        策略：寻找【第一个】梯度绝对值超过阈值的局部峰值。
        """
        if len(profile) < 5: return None
        
        # 1. 高斯平滑 (减少噪声干扰)
        kernel = np.array([0.25, 0.5, 0.25])
        smooth_profile = np.convolve(profile, kernel, mode='same')
        
        # 2. 计算梯度 (使用中心差分)
        # grad[i] = (f[i+1] - f[i-1]) / 2
        gradients = np.gradient(smooth_profile)
        abs_grads = np.abs(gradients)
        
        # 3. 寻找候选峰值
        # 简单的非极大值抑制 (NMS) 逻辑
        candidates = []
        for i in range(2, len(abs_grads) - 2):
            g = abs_grads[i]
            if g > threshold:
                # 判断是否是局部峰值
                if g >= abs_grads[i-1] and g >= abs_grads[i+1]:
                    candidates.append(i)
        
        if not candidates:
            return None
            
        # 4. 关键：方向性选择
        # 因为 profile 是按照用户画线的方向提取的 (Start -> End)
        # 所以我们直接取 candidates[0] 就是遇到的第一个边缘
        best_idx = candidates[0]
        
        # 5. 亚像素拟合 (抛物线插值)
        # 利用 best_idx 及其左右邻域的梯度值拟合抛物线 y = ax^2 + bx + c
        # 峰值位置 x = -b / (2a)
        y1 = abs_grads[best_idx - 1]
        y2 = abs_grads[best_idx]
        y3 = abs_grads[best_idx + 1]
        
        denom = (y1 - 2*y2 + y3)
        if denom == 0:
            return float(best_idx)
            
        delta = (y1 - y3) / (2 * denom)
        sub_pixel_idx = best_idx + delta
        
        return sub_pixel_idx

    @staticmethod
    def fit_circle(points):
        """ 最小二乘法拟合圆 """
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

    @staticmethod
    def fit_line(points):
        """ 最小二乘法拟合直线 (返回 point, direction_vector) """
        pts = np.array(points)
        # [vx, vy, x, y]
        vx, vy, x, y = cv2.fitLine(pts.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01)
        return (x[0], y[0]), (vx[0], vy[0])

# ==============================================================================
# 2. 交互视图层 (Interaction)
# ==============================================================================

class InteractiveView(QGraphicsView):
    point_selected = pyqtSignal(float, float)     # 选点信号
    direction_defined = pyqtSignal(float, object) # 方向确定信号 (length, vector_tuple)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setRenderHint(QPainter.Antialiasing)
        self.setMouseTracking(True) # 追踪鼠标移动
        
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item = None
        
        # 状态机
        self.mode = "VIEW" # VIEW, PICK, DIRECTION
        self.temp_items = []
        
        # 拖拽绘制方向线相关
        self._drag_start = None
        self._drag_line_item = None

    def load_image(self, cv_img):
        self._scene.clear()
        self.temp_items = []
        h, w, c = cv_img.shape
        # BGR -> RGB
        q_img = QImage(cv_img.data, w, h, 3*w, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(q_img)
        self._pixmap_item = QGraphicsPixmapItem(pixmap)
        self._scene.addItem(self._pixmap_item)
        self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)

    def clear_overlays(self):
        for item in self.temp_items:
            self._scene.removeItem(item)
        self.temp_items = []

    def wheelEvent(self, event):
        """ 鼠标中心缩放 """
        zoom_in = 1.15
        zoom_out = 1.0 / zoom_in
        factor = zoom_in if event.angleDelta().y() > 0 else zoom_out
        
        old_pos = self.mapToScene(event.pos())
        self.scale(factor, factor)
        new_pos = self.mapToScene(event.pos())
        delta = new_pos - old_pos
        self.translate(delta.x(), delta.y())

    def mousePressEvent(self, event):
        sp = self.mapToScene(event.pos())
        
        if self.mode == "PICK" and event.button() == Qt.LeftButton:
            # 绘制选取的点
            r = 4
            dot = self._scene.addEllipse(sp.x()-r, sp.y()-r, r*2, r*2, QPen(Qt.red), QBrush(Qt.red))
            self.temp_items.append(dot)
            self.point_selected.emit(sp.x(), sp.y())
            
        elif self.mode == "DIRECTION" and event.button() == Qt.LeftButton:
            self._drag_start = sp
            self._drag_line_item = QGraphicsLineItem(sp.x(), sp.y(), sp.x(), sp.y())
            pen = QPen(Qt.cyan, 2, Qt.SolidLine)
            pen.setCosmetic(True) # 线宽不随缩放变化
            self._drag_line_item.setPen(pen)
            self._scene.addItem(self._drag_line_item)
            
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        sp = self.mapToScene(event.pos())

        if self.mode == "DIRECTION" and self._drag_start is not None:
            # 更新正在拖拽的线
            line = QLineF(self._drag_start, sp)
            self._drag_line_item.setLine(line)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        sp = self.mapToScene(event.pos())

        if self.mode == "DIRECTION" and self._drag_start is not None:
            # 计算长度和方向
            p1 = (self._drag_start.x(), self._drag_start.y())
            p2 = (sp.x(), sp.y())
            
            line_len = np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
            if line_len > 2: # 防止误触
                # 绘制一个箭头头部表示方向
                arrow_len = min(10, line_len * 0.2)
                angle = np.arctan2(p1[1]-p2[1], p1[0]-p2[0])
                
                # 计算箭头两翼坐标
                p_arrow1 = QPointF(p2[0] + arrow_len * np.cos(angle + 0.5), p2[1] + arrow_len * np.sin(angle + 0.5))
                p_arrow2 = QPointF(p2[0] + arrow_len * np.cos(angle - 0.5), p2[1] + arrow_len * np.sin(angle - 0.5))
                
                # --- 修改开始: 将列表转换为 QPolygonF 对象 ---
                arrow_poly = QPolygonF([QPointF(*p2), p_arrow1, p_arrow2])
                
                head = self._scene.addPolygon(
                    arrow_poly, 
                    QPen(Qt.cyan), QBrush(Qt.cyan)
                )
                # --- 修改结束 ---

                self.temp_items.append(self._drag_line_item)
                self.temp_items.append(head)
                
                self.direction_defined.emit(line_len, (p1, p2))
            else:
                self._scene.removeItem(self._drag_line_item)
            
            self._drag_start = None
            self._drag_line_item = None

# ==============================================================================
# 3. 主界面与业务逻辑 (Main Logic)
# ==============================================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("智能边缘测量工具 (Smart Edge Metrology)")
        self.resize(1300, 850)
        
        # 数据状态
        self.cv_image = None
        self.gray_image = None
        self.points = []
        self.scan_config = {"width": 40.0, "vector": None} # vector: ((x1,y1), (x2,y2))
        
        self.setup_ui()

    def setup_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        
        # --- 左侧面板 ---
        panel = QWidget()
        panel.setFixedWidth(280)
        vbox = QVBoxLayout(panel)
        
        # 1. 图像
        vbox.addWidget(QLabel("<b>1. 图像加载</b>"))
        btn_load = QPushButton("打开图片")
        btn_load.clicked.connect(self.load_image)
        vbox.addWidget(btn_load)
        
        # 2. 模式
        vbox.addSpacing(10)
        vbox.addWidget(QLabel("<b>2. 测量设置</b>"))
        self.rb_circle = QRadioButton("圆形测量 (Circle)")
        self.rb_line = QRadioButton("直线测量 (Line)")
        self.rb_circle.setChecked(True)
        vbox.addWidget(self.rb_circle)
        vbox.addWidget(self.rb_line)
        
        # 参数
        form_layout = QHBoxLayout()
        form_layout.addWidget(QLabel("定圆点数:"))
        self.spin_pts = QSpinBox()
        self.spin_pts.setRange(3, 20); self.spin_pts.setValue(3)
        form_layout.addWidget(self.spin_pts)
        vbox.addLayout(form_layout)
        
        form_layout2 = QHBoxLayout()
        form_layout2.addWidget(QLabel("比例(mm/px):"))
        self.spin_scale = QDoubleSpinBox()
        self.spin_scale.setDecimals(4); self.spin_scale.setValue(0.0100)
        form_layout2.addWidget(self.spin_scale)
        vbox.addLayout(form_layout2)

        form_layout3 = QHBoxLayout()
        form_layout3.addWidget(QLabel("梯度阈值:"))
        self.spin_thresh = QDoubleSpinBox()
        self.spin_thresh.setRange(1.0, 100.0); self.spin_thresh.setValue(10.0)
        form_layout3.addWidget(self.spin_thresh)
        vbox.addLayout(form_layout3)
        
        # 3. 操作流
        vbox.addSpacing(20)
        vbox.addWidget(QLabel("<b>3. 操作流程</b>"))
        
        self.btn_pick = QPushButton("Step A: 选取轮廓点")
        self.btn_pick.clicked.connect(self.start_pick_points)
        vbox.addWidget(self.btn_pick)
        
        self.lbl_hint_pick = QLabel("未开始")
        self.lbl_hint_pick.setStyleSheet("color: gray")
        vbox.addWidget(self.lbl_hint_pick)

        self.btn_direction = QPushButton("Step B: 绘制扫描方向")
        self.btn_direction.setEnabled(False)
        self.btn_direction.clicked.connect(self.start_draw_direction)
        vbox.addWidget(self.btn_direction)
        
        self.lbl_hint_dir = QLabel("需先选点")
        self.lbl_hint_dir.setStyleSheet("color: gray")
        vbox.addWidget(self.lbl_hint_dir)
        
        vbox.addSpacing(10)
        self.btn_calc = QPushButton("Step C: 计算与测量")
        self.btn_calc.setStyleSheet("background-color: #d0f0c0; height: 40px; font-weight: bold;")
        self.btn_calc.setEnabled(False)
        self.btn_calc.clicked.connect(self.run_measurement)
        vbox.addWidget(self.btn_calc)
        
        # 结果显示
        self.lbl_result = QLabel("结果区域")
        self.lbl_result.setFrameShape(QLabel.Panel)
        self.lbl_result.setWordWrap(True)
        vbox.addWidget(self.lbl_result)
        
        vbox.addStretch()
        
        # --- 右侧视图 ---
        self.view = InteractiveView()
        self.view.point_selected.connect(self.on_point_picked)
        self.view.direction_defined.connect(self.on_direction_set)
        
        layout.addWidget(panel)
        layout.addWidget(self.view)

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.bmp *.tif)")
        if path:
            self.cv_image = cv2.imread(path)
            if self.cv_image is None: return
            self.gray_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)
            self.view.load_image(self.cv_image)
            self.reset_state()

    def reset_state(self):
        self.points = []
        self.scan_config = {"width": 40.0, "vector": None}
        self.view.clear_overlays()
        self.view.mode = "VIEW"
        self.btn_direction.setEnabled(False)
        self.btn_calc.setEnabled(False)
        self.lbl_hint_pick.setText("请点击图片选点...")
        self.lbl_hint_dir.setText("等待选点完成")
        self.lbl_result.setText("")

    # --- 流程 Step A: 选点 ---
    def start_pick_points(self):
        if self.cv_image is None: return
        self.points = []
        self.view.clear_overlays()
        self.view.mode = "PICK"
        self.lbl_hint_pick.setText("正在选点... (滚轮缩放)")
        self.btn_direction.setEnabled(False)
        self.btn_calc.setEnabled(False)

    def on_point_picked(self, x, y):
        self.points.append((x, y))
        target = self.spin_pts.value() if self.rb_circle.isChecked() else 2
        
        self.lbl_hint_pick.setText(f"已选 {len(self.points)} / {target}")
        
        if len(self.points) >= target:
            self.view.mode = "VIEW" # 自动结束选点
            self.lbl_hint_pick.setText("选点完成!")
            self.btn_direction.setEnabled(True)
            self.lbl_hint_dir.setText("请点击下方按钮开始绘制方向")
            
            # 绘制初步拟合结果作为参考
            self.draw_preliminary_geometry()

    def draw_preliminary_geometry(self):
        if self.rb_circle.isChecked():
            center, radius = MeasureUtils.fit_circle(self.points)
            if center:
                item = QGraphicsEllipseItem(center[0]-radius, center[1]-radius, radius*2, radius*2)
                item.setPen(QPen(Qt.yellow, 1, Qt.DashLine))
                self.view._scene.addItem(item)
                self.view.temp_items.append(item)
        else:
            # 简单的两点连线预览
            p1, p2 = self.points[0], self.points[-1]
            item = QGraphicsLineItem(p1[0], p1[1], p2[0], p2[1])
            item.setPen(QPen(Qt.yellow, 1, Qt.DashLine))
            self.view._scene.addItem(item)
            self.view.temp_items.append(item)

    # --- 流程 Step B: 绘制方向 ---
    def start_draw_direction(self):
        self.view.mode = "DIRECTION"
        self.lbl_hint_dir.setText("在图上拖拽: 起点(背景)->终点(目标)")
        QMessageBox.information(self, "提示", "请在图像边缘处拖拽一条线段。\n\n1. 线段长度 = 搜索宽度(ROI)\n2. 箭头方向 = 梯度搜索方向\n(建议：从无边缘区域指向有边缘区域)")

    def on_direction_set(self, length, vector):
        self.scan_config["width"] = length
        self.scan_config["vector"] = vector
        self.view.mode = "VIEW"
        
        self.lbl_hint_dir.setText(f"方向已定: 宽度={length:.1f}px")
        self.btn_calc.setEnabled(True)

    # --- 流程 Step C: 计算 ---
    def run_measurement(self):
        if self.rb_circle.isChecked():
            self.measure_circle()
        else:
            self.measure_line()

    def measure_circle(self):
        # 1. 获取初步几何信息
        center, radius = MeasureUtils.fit_circle(self.points)
        if center is None: return
        cx, cy = center
        
        # 2. 解析扫描方向
        # 向量 V = End - Start
        (vx1, vy1), (vx2, vy2) = self.scan_config["vector"]
        scan_width = self.scan_config["width"]
        
        # 判断是 "向心(Out->In)" 还是 "离心(In->Out)"
        # 计算向量中点相对于圆心的方向
        mid_x, mid_y = (vx1+vx2)/2, (vy1+vy2)/2
        vec_radial = np.array([mid_x - cx, mid_y - cy])
        vec_user = np.array([vx2 - vx1, vy2 - vy1])
        
        # 点积 > 0 表示同向(离心)， < 0 表示反向(向心)
        is_outward = np.dot(vec_radial, vec_user) > 0
        
        # 3. 生成所有扫描线并测量
        detected_pts = []
        num_spokes = 360 # 采样密度
        
        # 准备批量计算，不阻塞UI建议用线程，这里简单处理
        grad_thresh = self.spin_thresh.value()

        for i in range(num_spokes):
            angle_rad = np.deg2rad(i)
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)
            
            # 定义几何上的内圆和外圆半径
            r_inner = radius - scan_width / 2
            r_outer = radius + scan_width / 2
            
            # 根据用户方向，决定采样的 Start 和 End
            if is_outward:
                # 离心：Start=内, End=外
                p_start = (cx + r_inner * cos_a, cy + r_inner * sin_a)
                p_end   = (cx + r_outer * cos_a, cy + r_outer * sin_a)
            else:
                # 向心：Start=外, End=内 (通常用于 背景->目标)
                p_start = (cx + r_outer * cos_a, cy + r_outer * sin_a)
                p_end   = (cx + r_inner * cos_a, cy + r_inner * sin_a)
            
            # 提取剖面
            profile = MeasureUtils.get_profile_subpix(self.gray_image, p_start, p_end, scan_width)
            
            # 1D 寻边
            idx = MeasureUtils.measure_1d_edge_directional(profile, grad_thresh)
            
            if idx is not None:
                # 将 1D 索引映射回 2D 坐标
                # idx 是相对于 p_start 的距离比例
                ratio = idx / (len(profile) - 1)
                px = p_start[0] + (p_end[0] - p_start[0]) * ratio
                py = p_start[1] + (p_end[1] - p_start[1]) * ratio
                detected_pts.append((px, py))
                
                # 可视化点 (绿色)
                self.view._scene.addRect(px-0.5, py-0.5, 1, 1, QPen(Qt.green), QBrush(Qt.green))

        # 4. 最终拟合与输出
        if len(detected_pts) > 10:
            final_center, final_radius = MeasureUtils.fit_circle(detected_pts)
            
            # 绘制最终圆 (蓝色)
            final_item = QGraphicsEllipseItem(final_center[0]-final_radius, final_center[1]-final_radius, 
                                              final_radius*2, final_radius*2)
            final_item.setPen(QPen(Qt.blue, 2))
            self.view._scene.addItem(final_item)
            self.view.temp_items.append(final_item)
            
            # 结果文本
            scale = self.spin_scale.value()
            phys_r = final_radius * scale
            phys_d = phys_r * 2
            
            res_str = (f"【测量成功】\n"
                       f"点数: {len(detected_pts)}\n"
                       f"圆心: ({final_center[0]:.1f}, {final_center[1]:.1f})\n"
                       f"像素半径: {final_radius:.2f} px\n"
                       f"物理直径: {phys_d:.4f} mm")
        else:
            res_str = "测量失败: 未找到足够的边缘点。\n请检查方向设置或调整梯度阈值。"
            
        self.lbl_result.setText(res_str)

    def measure_line(self):
        # 1. 拟合初始直线
        (lx, ly), (nvx, nvy) = MeasureUtils.fit_line(self.points) # nvx,nvy是单位方向向量
        
        # 2. 解析扫描方向
        (vx1, vy1), (vx2, vy2) = self.scan_config["vector"]
        scan_width = self.scan_config["width"]
        grad_thresh = self.spin_thresh.value()
        
        # 用户向量
        u_vec = np.array([vx2-vx1, vy2-vy1])
        u_len = np.linalg.norm(u_vec)
        if u_len == 0: return
        u_vec_norm = u_vec / u_len
        
        # 直线法向量 (我们要沿法向量扫描)
        # 直线方向是 (nvx, nvy)，法向量是 (-nvy, nvx) 或 (nvy, -nvx)
        # 选择与用户向量点积为正的那个法向量
        normal1 = np.array([-nvy, nvx])
        if np.dot(normal1, u_vec_norm) < 0:
            normal1 = -normal1
            
        # 3. 沿直线生成多个扫描线
        # 需要确定直线的范围 (投影点)
        pts_arr = np.array(self.points)
        # 简单处理：取点集中最远的两端投影
        # 这里为了演示，直接取点集中 x 最小和最大的范围，沿直线方向步进
        # 更好的方法是旋转坐标系。这里简化处理：
        
        # 计算所有点在直线上的投影位置 t
        # P = P0 + t * V
        detected_pts = []
        
        # 既然是 Demo，我们直接在每两个点击点之间生成扫描线
        # 或者简化：假设点是按顺序排的，从 p[0] 到 p[-1] 扫描
        p_start_node = np.array(self.points[0])
        p_end_node = np.array(self.points[-1])
        
        line_vec = p_end_node - p_start_node
        line_len = np.linalg.norm(line_vec)
        steps = int(line_len) # 每像素一个扫描线
        
        step_vec = line_vec / steps
        
        scan_offset_vec = normal1 * (scan_width / 2.0)
        
        for i in range(steps):
            # 当前直线上的中心点
            curr_center = p_start_node + step_vec * i
            
            # 关键：根据用户方向确定的 Start 和 End
            # 用户向量不仅定了法线方向，还定了谁是Start谁是End
            # normal1 已经是指向 "User Vector End" 的方向了
            # 所以 Start 在 -normal1 侧，End 在 +normal1 侧?
            # 不，用户画的是 Start -> End。
            # 我们的 scan 应该是从 "User Start 侧" 走到 "User End 侧"
            
            # 构造扫描线段： Center - (Width/2)*Dir  --->  Center + (Width/2)*Dir
            # 这里的 Dir 必须与 User Vector 同向
            p_scan_start = curr_center - scan_offset_vec
            p_scan_end   = curr_center + scan_offset_vec
            
            # 提取
            profile = MeasureUtils.get_profile_subpix(self.gray_image, p_scan_start, p_scan_end, scan_width)
            
            # 寻边
            idx = MeasureUtils.measure_1d_edge_directional(profile, grad_thresh)
            
            if idx is not None:
                ratio = idx / (len(profile) - 1)
                px = p_scan_start[0] + (p_scan_end[0] - p_scan_start[0]) * ratio
                py = p_scan_start[1] + (p_scan_end[1] - p_scan_start[1]) * ratio
                detected_pts.append((px, py))
                self.view._scene.addRect(px-0.5, py-0.5, 1, 1, QPen(Qt.green), QBrush(Qt.green))
                
        # 4. 拟合直线并计算距离(略)
        if len(detected_pts) > 5:
            # Fit line again
            (fx, fy), (fvx, fvy) = MeasureUtils.fit_line(detected_pts)
            
            # 绘制一条长线代表结果
            # 取 detected_pts 的包围盒画线
            d_arr = np.array(detected_pts)
            min_x, max_x = np.min(d_arr[:,0]), np.max(d_arr[:,0])
            
            # y = y0 + (x-x0) * vy/vx
            if abs(fvx) > 1e-5:
                y_min = fy + (min_x - fx) * (fvy/fvx)
                y_max = fy + (max_x - fx) * (fvy/fvx)
                l_item = QGraphicsLineItem(min_x, y_min, max_x, y_max)
                l_item.setPen(QPen(Qt.blue, 2))
                self.view._scene.addItem(l_item)
                self.view.temp_items.append(l_item)
            
            self.lbl_result.setText(f"【直线测量成功】\n点数: {len(detected_pts)}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())