import cv2
import numpy as np
from .base import EdgeExtractorBase

class SobelEdge(EdgeExtractorBase):
    def extract(self, image, roi_info, step, bg_color=None):
        crop, off_x, off_y = self._get_roi_crop(image, roi_info)
        if crop is None: return []

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        
        # Sobel 算子
        grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0)
        grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1)
        abs_x = cv2.convertScaleAbs(grad_x)
        abs_y = cv2.convertScaleAbs(grad_y)
        grad = cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)
        
        # 二值化
        _, edges = cv2.threshold(grad, 50, 255, cv2.THRESH_BINARY)
        
        # --- 以下逻辑与 CannyEdge 完全一致 (向量化筛选) ---
        ys_local, xs_local = np.where(edges > 0)
        if len(xs_local) == 0: return []

        gx = xs_local + off_x
        gy = ys_local + off_y
        pts = []

        if len(roi_info) == 3 and not isinstance(roi_info[0], str):
            (cx, cy), r_in, r_out = roi_info
            dists = np.hypot(gx - cx, gy - cy)
            mask = (dists >= r_in) & (dists <= r_out)
            pts = np.column_stack((gx[mask], gy[mask])).tolist()
            
        elif len(roi_info) == 6 and roi_info[0] == "line":
            _, p_s, p_e, d_min, d_max, nv = roi_info
            p_s = np.array(p_s)
            dists = np.dot(np.column_stack((gx, gy)) - p_s, nv)
            mask = (dists >= d_min) & (dists <= d_max)
            pts = np.column_stack((gx[mask], gy[mask])).tolist()

        return pts