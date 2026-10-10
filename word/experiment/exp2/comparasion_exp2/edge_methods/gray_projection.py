# import numpy as np
# import cv2
# from .base import EdgeExtractorBase

# class GrayProjectionEdge(EdgeExtractorBase):
#     def extract(self, image, roi_info, step, bg_color=None):
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         (cx, cy), r_in, r_out = roi_info

#         pts = []
#         num = max(int(360 / step), 10)
#         for i in range(num):
#             ang = np.deg2rad(i * step)
#             ca, sa = np.cos(ang), np.sin(ang)
#             p_s = (cx + r_in * ca, cy + r_in * sa)
#             p_e = (cx + r_out * ca, cy + r_out * sa)

#             prof = self._profile(gray, p_s, p_e)
#             if prof is None:
#                 continue

#             g = np.abs(np.gradient(prof))
#             idx = np.argmax(g)
#             if 1 <= idx < len(prof) - 1:
#                 y1, y2, y3 = g[idx-1], g[idx], g[idx+1]
#                 denom = y1 - 2*y2 + y3
#                 delta = (y1 - y3) / (2*denom) if denom != 0 else 0.0
#                 t = (idx + delta) / (len(prof)-1)
#                 x = p_s[0] + (p_e[0] - p_s[0]) * t
#                 y = p_s[1] + (p_e[1] - p_s[1]) * t
#                 pts.append((x, y))
#         return pts

#     def _profile(self, img, p1, p2):
#         x1, y1 = p1
#         x2, y2 = p2
#         n = int(np.hypot(x2-x1, y2-y1))
#         if n < 2:
#             return None
#         xs = np.linspace(x1, x2, n).astype(np.float32)
#         ys = np.linspace(y1, y2, n).astype(np.float32)
#         prof = cv2.remap(
#             img,
#             xs.reshape(1, -1),
#             ys.reshape(1, -1),
#             cv2.INTER_LINEAR,
#             borderMode=cv2.BORDER_REPLICATE
#         )
#         return prof[0]


import numpy as np
import cv2
from .base import EdgeExtractorBase


class GrayProjectionEdge(EdgeExtractorBase):
    def extract(self, image, roi_info, step, bg_color=None):
        # 扫描线方法不需要裁剪，直接转灰度供 remap 使用
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        pts = []

        # === 圆形扫描 ===
        if len(roi_info) == 3 and not isinstance(roi_info[0], str):
            (cx, cy), r_in, r_out = roi_info
            circumference = 2 * np.pi * ((r_in + r_out) / 2)
            num = max(int(circumference / step), 12)
            
            for i in range(num):
                ang = (2 * np.pi * i) / num
                ca, sa = np.cos(ang), np.sin(ang)
                p_s = (cx + r_in * ca, cy + r_in * sa)
                p_e = (cx + r_out * ca, cy + r_out * sa)
                self._process_one_line(gray, p_s, p_e, pts)

        # === 直线扫描 ===
        elif len(roi_info) == 6 and roi_info[0] == "line":
            _, p_s, p_e, d_min, d_max, nv = roi_info

            # === 与 ColorVectorEdge 完全一致的几何定义 ===
            vec = p_e - p_s
            length = np.linalg.norm(vec)
            if length < 1e-6:
                return pts

            uv = vec / length
            num_steps = max(int(length / step), 2)

            for i in range(num_steps + 1):
                # 基准线上的当前点
                curr = p_s + uv * (i * step)

                # 沿法线方向构造扫描线
                p1 = curr + nv * d_min
                p2 = curr + nv * d_max
                self._process_one_line(gray, p1, p2, pts)

        return pts

    def _process_one_line(self, gray_img, p1, p2, pts_list):
        # 使用基类方法提取剖面 (Base 里的 remap 对单通道也有效)
        prof = self._get_profile(gray_img, p1, p2)
        if prof is None or len(prof) < 3: return
        
        # 简单的一阶导数最大值
        prof = prof.flatten().astype(np.float32)
        grads = np.abs(np.gradient(prof))
        
        idx = np.argmax(grads)
        if grads[idx] < 5.0: return # 阈值过滤弱边缘
        
        # 亚像素
        if 0 < idx < len(grads) - 1:
            y1, y2, y3 = grads[idx-1], grads[idx], grads[idx+1]
            denom = y1 - 2*y2 + y3
            if abs(denom) > 1e-5:
                idx += (y1 - y3) / (2 * denom)
                
        # 坐标映射
        t = idx / (len(prof) - 1)
        px = p1[0] + (p2[0] - p1[0]) * t
        py = p1[1] + (p2[1] - p1[1]) * t
        pts_list.append((px, py))
