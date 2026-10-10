import cv2
import numpy as np
from .base import EdgeExtractorBase

class CannyEdge(EdgeExtractorBase):
    def extract(self, image, roi_info, step, bg_color=None):
        # 1. 切出包含圆环的矩形 (减少 Canny 处理的数据量)
        crop, off_x, off_y = self._get_roi_crop(image, roi_info)
        if crop is None: return []

        # 2. 在矩形上运行 Canny
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # 3. 获取所有边缘点的局部坐标
        # np.where 返回的是 (row, col) 即 (y, x)
        ys_local, xs_local = np.where(edges > 0)
        
        if len(xs_local) == 0: return []

        # 4. 转换为全局坐标 (向量化加法，极快)
        gx = xs_local + off_x
        gy = ys_local + off_y

        # 5. 严格的 ROI 几何过滤 (向量化计算)
        pts = []
        
        if len(roi_info) == 3 and not isinstance(roi_info[0], str):
            # --- 圆环逻辑 ---
            (cx, cy), r_in, r_out = roi_info
            
            # 计算所有边缘点到圆心的距离
            dists = np.hypot(gx - cx, gy - cy)
            
            # 核心：生成掩膜，只保留 r_in <= dist <= r_out 的点
            # 这一步剔除了矩形四角和内圆孔洞的所有干扰点
            mask = (dists >= r_in) & (dists <= r_out)
            
            # 应用掩膜
            valid_x = gx[mask]
            valid_y = gy[mask]
            
            # 组合结果
            pts = np.column_stack((valid_x, valid_y)).tolist()

        elif len(roi_info) == 6 and roi_info[0] == "line":
            # --- 直线逻辑 ---
            _, p_s, p_e, d_min, d_max, nv = roi_info
            p_s = np.array(p_s)
            
            # 计算所有点到基准直线的投影距离
            # 向量化点积: dot((N,2), (2,))
            vecs = np.column_stack((gx, gy)) - p_s
            dists = np.dot(vecs, nv)
            
            # 掩膜过滤
            mask = (dists >= d_min) & (dists <= d_max)
            
            valid_x = gx[mask]
            valid_y = gy[mask]
            pts = np.column_stack((valid_x, valid_y)).tolist()

        return pts