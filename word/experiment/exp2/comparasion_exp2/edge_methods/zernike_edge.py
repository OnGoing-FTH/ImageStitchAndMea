import numpy as np
import cv2
from .base import EdgeExtractorBase

class ZernikeEdge(EdgeExtractorBase):
    """
    Zernike Edge Extractor
    保留原始算法逻辑，仅补全直线扫描(line)分支
    """

    def extract(self, image, roi_info, step, bg_color=None):
        # 统一转灰度
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            gray = image.astype(np.float32)
            
        pts = []

        # =====================================================
        # 1. 圆环 ROI
        # =====================================================
        if len(roi_info) == 3 and not isinstance(roi_info[0], str):
            (cx, cy), r_in, r_out = roi_info
            r_mid = 0.5 * (r_in + r_out)
            circumference = 2 * np.pi * r_mid
            num = max(int(circumference / step), 12)

            for i in range(num):
                ang = 2 * np.pi * i / num
                ca, sa = np.cos(ang), np.sin(ang)
                p1 = (cx + r_in * ca, cy + r_in * sa)
                p2 = (cx + r_out * ca, cy + r_out * sa)
                
                pt = self._scanline_edge(gray, p1, p2)
                if pt is not None: pts.append(pt)

        # =====================================================
        # 2. 直线带宽扫描 (Line Scan) -> 🔴 补全了这个缺失的分支
        # =====================================================
        elif len(roi_info) == 6 and roi_info[0] == "line":
            # 解包 UI 传来的参数
            _, p_s, p_e, d_min, d_max, nv = roi_info
            
            # 计算基准线长度和方向
            vec = p_e - p_s
            length = np.linalg.norm(vec)
            if length < 1e-6: return pts # 防止除以0
            
            uv = vec / length # 单位方向向量
            num_steps = max(int(length / step), 2)
            
            for i in range(num_steps + 1):
                # 计算当前扫描线的中心点
                curr = p_s + uv * (i * step)
                
                # 沿法线方向构造扫描线 (从 d_min 到 d_max)
                p1 = curr + nv * d_min
                p2 = curr + nv * d_max
                
                pt = self._scanline_edge(gray, p1, p2)
                if pt is not None: pts.append(pt)

        # =====================================================
        # 3. 矩形 ROI (保留原有逻辑)
        # =====================================================
        elif len(roi_info) == 6 and roi_info[0] == "rect":
            _, x, y, w, h, direction = roi_info
            x, y, w, h = map(int, (x, y, w, h))

            if direction == "vertical":
                num = max(int(h / step), 1)
                for i in range(num):
                    yy = y + i * step
                    p1, p2 = (x, yy), (x + w, yy)
                    pt = self._scanline_edge(gray, p1, p2)
                    if pt is not None: pts.append(pt)

            elif direction == "horizontal":
                num = max(int(w / step), 1)
                for i in range(num):
                    xx = x + i * step
                    p1, p2 = (xx, y), (xx, y + h)
                    pt = self._scanline_edge(gray, p1, p2)
                    if pt is not None: pts.append(pt)

        return pts

    # =====================================================
    # 核心处理函数
    # =====================================================
    def _scanline_edge(self, gray, p1, p2):
        # 1. 提取剖面
        prof = self._sample_profile(gray, p1, p2)
        if prof is None: return None

        # 2. 计算边缘索引 (使用你原始的 Zernike/二阶导逻辑)
        idx = self._zernike_edge_index(prof)
        if idx is None: return None

        # 3. 映射回全局坐标
        t = idx / (len(prof) - 1)
        x = p1[0] + (p2[0] - p1[0]) * t
        y = p1[1] + (p2[1] - p1[1]) * t
        return (float(x), float(y))

    def _sample_profile(self, img, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        dist = np.hypot(x2 - x1, y2 - y1)
        n = int(dist)
        if n < 5: return None # 长度太短不处理

        xs = np.linspace(x1, x2, n, dtype=np.float32)
        ys = np.linspace(y1, y2, n, dtype=np.float32)

        prof = cv2.remap(
            img,
            xs.reshape(1, -1),
            ys.reshape(1, -1),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )
        return prof[0]

    def _zernike_edge_index(self, profile):
        """
        原始提供的算法：二阶导数峰值 + 简单的亚像素插值
        不做任何修改
        """
        g = np.gradient(profile)
        z = np.gradient(g)

        idx = int(np.argmax(np.abs(z)))
        
        # 边界检查
        if idx <= 1 or idx >= len(profile) - 2:
            return None

        y1, y2, y3 = z[idx - 1], z[idx], z[idx + 1]
        denom = y1 - 2 * y2 + y3
        
        # 防止分母为0
        if abs(denom) < 1e-6:
            return float(idx)

        delta = 0.5 * (y1 - y3) / denom
        return float(idx + delta)