import numpy as np
import cv2

class FitLS:
    def fit_circle(self, points, **kwargs):
        """
        标准最小二乘圆拟合。
        拟合模型由【所有点】决定。
        拟合后，根据阈值统计离群点。
        """
        if points is None or len(points) < 3:
            return {"success": False, "msg": "点数不足3"}

        pts = np.asarray(points, dtype=np.float64)
        x, y = pts[:, 0], pts[:, 1]

        # 1. 使用矩阵法求解标准最小二乘圆 (A*x = b)
        # 即使有噪点，也全部参与计算
        A = np.column_stack((2*x, 2*y, np.ones_like(x)))
        b = x**2 + y**2

        try:
            res, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            cx, cy = res[0], res[1]
            r = np.sqrt(res[2] + cx**2 + cy**2)
        except:
            return {"success": False, "msg": "拟合计算失败"}

        # 2. 【事后统计】：计算点到圆的距离，判断是否离群
        # 注意：这里的筛选不会改变上面的 cx, cy, r
        dists = np.abs(np.hypot(x - cx, y - cy) - r)
        
        # 获取阈值 (如果没有传，默认给个宽松的值)
        threshold = kwargs.get('outlier_threshold', 3.0)
        
        # 生成掩膜：距离小于阈值的记为有效点
        valid_mask = dists <= threshold
        valid_pts = pts[valid_mask]
        
        # 计算圆度 (基于所有点还是有效点？通常评估只看有效点，或者看整体)
        # 这里为了符合逻辑，圆度误差应该反映有效点的贴合程度
        if len(valid_pts) > 0:
            v_dists = np.hypot(valid_pts[:,0] - cx, valid_pts[:,1] - cy)
            circularity = v_dists.max() - v_dists.min()
        else:
            circularity = 0.0

        return {
            "success": True,
            "center": (cx, cy),
            "radius": r,
            "circularity": float(circularity),
            "valid_points": valid_pts,  # UI会将不在这个列表里的点画成蓝色
            "outlier_count": len(pts) - len(valid_pts)
        }

    
    def fit_line(self, points, **kwargs):
        """
        修正后的鲁棒直线拟合
        """
        if points is None or len(points) < 2:
            return {"success": False, "msg": "点数不足"}

        pts = np.asarray(points, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[0] < 2:
            return {"success": False, "msg": "有效点不足以拟合直线"}

        if pts.size > 0:
            mask = np.isfinite(pts).all(axis=1)
            pts = pts[mask]

        if pts.shape[0] < 2:
            return {"success": False, "msg": "有效点不足以拟合直线(清洗NaN后)"}

        # 1. L2 初始拟合
        vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        n = np.hypot(vx, vy)
        if n == 0: return {"success": False, "msg": "方向向量模为0"}
        ux, uy = vx / n, vy / n

        # 2. 计算点到直线的距离
        diffs = pts - np.array([x0, y0], dtype=np.float32)
        dists = np.abs(diffs[:, 0] * (-uy) + diffs[:, 1] * ux)

        # ===============================
        # 修正部分：引入最小噪声容忍度
        # ===============================
        med = np.median(dists)
        mad = np.median(np.abs(dists - med))
        
        # 设定一个最小的 MAD 值（例如 0.5 像素）
        # 如果数据非常完美，MAD 会趋近于 0，但这不代表我们只接受 0 误差
        min_mad = 0.5 
        effective_mad = max(mad, min_mad)
        
        # 阈值 = 中位数 + 3倍的有效偏差
        # 这样即使 mad=0，阈值至少也是 med + 1.5，保证了亚像素点的存活
        threshold = med + 3.0 * effective_mad

        valid_mask = dists <= threshold
        valid_pts = pts[valid_mask]

        # 如果剔除后点太少，可能意味着初始 L2 拟合完全被远端噪点带偏了
        # 这种情况下可以考虑用 RANSAC，或者暂时返回所有点
        if len(valid_pts) < 2:
            # 兜底策略：如果鲁棒过滤杀完了所有点，说明初始拟合偏离太远
            # 但既然是 ColorVectorEdge，通常点是很准的，这里返回原始点更安全
            valid_pts = pts 

        # 3. 计算直线度
        if len(valid_pts) >= 2:
            diffs_v = valid_pts - np.array([x0, y0], dtype=np.float32)
            signed = diffs_v[:, 0] * uy - diffs_v[:, 1] * ux
            straightness = signed.max() - signed.min()
        else:
            straightness = 0.0
            
        # 1. 确保是 Numpy 数组
        if not isinstance(valid_pts, np.ndarray):
             valid_pts = np.array(valid_pts)
        valid_points_list = [tuple(map(float, p)) for p in valid_pts]

        return {
            "success": True,
            "line_point": (float(x0), float(y0)),
            "line_dir": (float(ux), float(uy)),
            "straightness": float(straightness),
            "valid_points": valid_points_list  
        }

