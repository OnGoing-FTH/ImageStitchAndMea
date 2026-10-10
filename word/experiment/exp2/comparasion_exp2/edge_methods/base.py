import cv2
import numpy as np

class EdgeExtractorBase:
    def extract(self, image, roi_info, step, bg_color=None):
        """
        子类实现具体的检测逻辑
        """
        raise NotImplementedError

    def _get_profile(self, image, p1, p2):
        """ 获取两点间的灰度/颜色剖面 """
        x1, y1 = p1
        x2, y2 = p2
        length = np.hypot(x2-x1, y2-y1)
        n = int(length)
        if n < 2: return None
        xs = np.linspace(x1, x2, n).astype(np.float32)
        ys = np.linspace(y1, y2, n).astype(np.float32)
        prof = cv2.remap(image, xs.reshape(1, -1), ys.reshape(1, -1), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return prof[0]

    def _get_roi_crop(self, image, roi_info, padding=5):
        """
        根据 ROI 计算最小包围盒并裁剪。
        虽然 ROI 是圆环，但 OpenCV 算子需要矩阵输入，所以必须切出包含圆环的外接矩形。
        我们在 extract 中会通过坐标掩膜 (Mask) 剔除矩形内圆环外的点。
        
        Returns:
            crop_img: 裁剪后的图像
            offset_x, offset_y: 裁剪图左上角在原图中的坐标
        """
        h, w = image.shape[:2]
        
        if len(roi_info) == 3 and not isinstance(roi_info[0], str):
            # --- 圆形 (圆环) ---
            (cx, cy), r_in, r_out = roi_info
            # 只需要包含外径的矩形
            r_max = r_out + padding
            x_min = int(cx - r_max)
            y_min = int(cy - r_max)
            x_max = int(cx + r_max)
            y_max = int(cy + r_max)
            
        elif len(roi_info) == 6 and roi_info[0] == "line":
            # --- 直线 (矩形带) ---
            _, p_s, p_e, d_min, d_max, nv = roi_info
            p_s, p_e, nv = np.array(p_s), np.array(p_e), np.array(nv)
            
            # 计算四个角点
            c1 = p_s + nv * d_min
            c2 = p_s + nv * d_max
            c3 = p_e + nv * d_max
            c4 = p_e + nv * d_min
            
            xs = [c1[0], c2[0], c3[0], c4[0]]
            ys = [c1[1], c2[1], c3[1], c4[1]]
            
            x_min = int(min(xs) - padding)
            y_min = int(min(ys) - padding)
            x_max = int(max(xs) + padding)
            y_max = int(max(ys) + padding)
        else:
            return image, 0, 0

        # 边界保护
        x_min = max(0, x_min); y_min = max(0, y_min)
        x_max = min(w, x_max); y_max = min(h, y_max)
        
        if x_max <= x_min or y_max <= y_min:
            return None, 0, 0
            
        return image[y_min:y_max, x_min:x_max], x_min, y_min