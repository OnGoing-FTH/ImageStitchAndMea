'''
import numpy as np
import cv2
from .base import EdgeExtractorBase

class ColorVectorEdge(EdgeExtractorBase):
    def extract(self, image, roi_info, step, bg_color=None):
        if bg_color is None:
            return []

        (cx, cy), r_in, r_out = roi_info
        pts = []

        num = max(int(360.0 / step), 10)
        for i in range(num):
            ang = np.deg2rad(i * step)
            ca, sa = np.cos(ang), np.sin(ang)
            p_s = (cx + r_in * ca, cy + r_in * sa)
            p_e = (cx + r_out * ca, cy + r_out * sa)

            prof = self._get_color_profile(image, p_s, p_e)
            idx = self._measure_color_edge_adaptive(prof, bg_color)
            if idx is not None:
                t = idx / (len(prof) - 1)
                x = p_s[0] + (p_e[0] - p_s[0]) * t
                y = p_s[1] + (p_e[1] - p_s[1]) * t
                pts.append((x, y))
        return pts

    # ===== 以下函数逻辑与你原代码保持一致 =====
    def _get_color_profile(self, image, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        n = int(np.hypot(x2-x1, y2-y1))
        if n < 2:
            return None
        xs = np.linspace(x1, x2, n).astype(np.float32)
        ys = np.linspace(y1, y2, n).astype(np.float32)
        prof = cv2.remap(
            image,
            xs.reshape(1, -1),
            ys.reshape(1, -1),
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )
        return prof[0]

    def _measure_color_edge_adaptive(self, profile, bg_color):
        if profile is None or len(profile) < 5:
            return None

        diff = profile.astype(np.float32) - bg_color.astype(np.float32)
        dist = np.linalg.norm(diff, axis=1)
        if dist.max() - dist.min() < 20:
            return None

        smooth = np.convolve(dist, [0.25, 0.5, 0.25], mode='same')
        g = np.abs(np.gradient(smooth))
        idx = np.argmax(g[2:-2]) + 2

        y1, y2, y3 = g[idx-1], g[idx], g[idx+1]
        denom = y1 - 2*y2 + y3
        delta = (y1 - y3) / (2 * denom) if denom != 0 else 0.0

        return float(idx) + delta

'''


import numpy as np
import cv2
from .base import EdgeExtractorBase

class ColorVectorEdge(EdgeExtractorBase):
    def extract(self, image, roi_info, step, bg_color=None):
        if bg_color is None: return []
        pts = []

        # === 分支 1: 圆形扫描 ===
        if len(roi_info) == 3 and not isinstance(roi_info[0], str):
            (cx, cy), r_in, r_out = roi_info
            
            # 自动计算扫描条数 (周长 / step)
            circumference = 2 * np.pi * ((r_in + r_out) / 2)
            num = max(int(circumference / step), 12) # 至少12条线
            
            for i in range(num):
                # 均匀分布在 0~360 度
                ang = (2 * np.pi * i) / num
                ca, sa = np.cos(ang), np.sin(ang)
                
                # 确定扫描线的起终点 (极坐标 -> 直角坐标)
                p_s = (cx + r_in * ca, cy + r_in * sa)
                p_e = (cx + r_out * ca, cy + r_out * sa)
                
                self._process_one_line(image, p_s, p_e, bg_color, pts)

        # === 分支 2: 直线扫描 ===
        elif len(roi_info) == 6 and roi_info[0] == "line":
            _, p_s, p_e, d_min, d_max, nv = roi_info
            
            # 计算基准线长度和方向
            vec = p_e - p_s
            length = np.linalg.norm(vec)
            uv = vec / length if length > 0 else vec
            
            num_steps = max(int(length / step), 2)
            
            for i in range(num_steps + 1):
                # 当前扫描线的中心点
                curr = p_s + uv * (i * step) if num_steps > 0 else p_s
                
                # 沿法线方向延伸出扫描线
                p1 = curr + nv * d_min
                p2 = curr + nv * d_max
                
                self._process_one_line(image, p1, p2, bg_color, pts)
                
        return pts

    def _process_one_line(self, image, p1, p2, bg_color, pts_list):
        # 1. 使用基类方法提取颜色剖面
        prof = self._get_profile(image, p1, p2)
        if prof is None: return

        # 2. 核心算法: 颜色距离 + 自适应梯度
        diff = prof.astype(np.float32) - bg_color.astype(np.float32)
        dist_curve = np.linalg.norm(diff, axis=1)
        
        # 简单检查: 动态范围太小说明全是背景或全是前景
        if dist_curve.max() - dist_curve.min() < 15.0: return
        
        # 梯度找边缘
        kernel = np.array([0.25, 0.5, 0.25])
        smooth = np.convolve(dist_curve, kernel, mode='same')
        grads = np.abs(np.gradient(smooth))
        
        # 避开边界 (2:-2)
        valid_grads = grads[2:-2]
        if len(valid_grads) == 0: return
        idx_local = np.argmax(valid_grads)
        idx = idx_local + 2
        
        # 🔴 关键修复：安全的亚像素插值
        if 0 < idx < len(grads) - 1:
            y1, y2, y3 = grads[idx-1], grads[idx], grads[idx+1]
            denom = y1 - 2*y2 + y3
            
            # 只有当分母足够大，且计算出的偏移量合理时，才应用插值
            if abs(denom) > 1e-4:  # 稍微提高一点分母阈值
                delta = (y1 - y3) / (2 * denom)
                # 限制偏移量在 [-0.7, 0.7] 之间
                # 如果偏移超过这个范围，说明不是真正的抛物线峰值，直接用整数索引更安全
                if abs(delta) < 0.7:
                    idx += delta
        
        # 3. 映射回全局坐标
        t = idx / (len(prof) - 1)
        px = p1[0] + (p2[0] - p1[0]) * t
        py = p1[1] + (p2[1] - p1[1]) * t
        pts_list.append((px, py))
