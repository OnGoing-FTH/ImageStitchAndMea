import numpy as np

class FitAverage:
    """
    【统计法 / 不拟合】
    不进行最小二乘优化，直接计算点集的“重心”作为圆心/中心。
    用于对比，证明 LS/Robust 拟合的必要性。
    """
    
    def fit_circle(self, points, **kwargs):
        if points is None or len(points) < 3:
            return {"success": False, "msg": "点数不足"}
        
        pts = np.asarray(points, dtype=np.float64)
        x, y = pts[:, 0], pts[:, 1]
        
        # 1. 直接取所有点的平均值作为圆心 (重心)
        # 这不是几何中心，而是坐标重心。如果是半圆，重心会偏向圆弧内部，导致半径计算错误。
        cx = np.mean(x)
        cy = np.mean(y)
        
        # 2. 计算所有点到重心的距离
        dists = np.hypot(x - cx, y - cy)
        
        # 3. 半径取平均距离
        r = np.mean(dists)
        
        # 4. 统计圆度 (极差)
        circularity = dists.max() - dists.min()
        
        # 5. 离群点判断 (同样可以使用阈值)
        threshold = kwargs.get('outlier_threshold', 3.0)
        valid_mask = np.abs(dists - r) <= threshold
        valid_pts = pts[valid_mask]
        
        return {
            "success": True,
            "center": (cx, cy),
            "radius": r,
            "circularity": float(circularity),
            "valid_points": valid_pts,
            "outlier_count": len(pts) - len(valid_pts),
            "msg": "Based on Centroid (Not Optimized)"
        }

    def fit_line(self, points, **kwargs):
        if points is None or len(points) < 2:
            return {"success": False, "msg": "点数不足"}
            
        pts = np.asarray(points, dtype=np.float64)
        
        # 1. 直线经过重心
        cx = np.mean(pts[:, 0])
        cy = np.mean(pts[:, 1])
        
        # 2. 方向直接取 首尾连线 (不通过中间点优化方向)
        # 这种方法完全依赖于起点和终点的准确性，极其不稳定
        p_start = pts[0]
        p_end = pts[-1]
        
        vec = p_end - p_start
        norm = np.hypot(vec[0], vec[1])
        
        if norm == 0:
            ux, uy = 1.0, 0.0 # 默认水平
        else:
            ux, uy = vec[0] / norm, vec[1] / norm
            
        # 3. 计算直线度
        # 距离公式 d = (p - center) x direction (2D cross product equivalent)
        # 法向量 (-uy, ux)
        diffs = pts - np.array([cx, cy])
        dists = np.abs(diffs[:, 0] * (-uy) + diffs[:, 1] * ux)
        
        straightness = dists.max() - dists.min()
        
        # 阈值筛选
        threshold = kwargs.get('outlier_threshold', 3.0)
        valid_mask = dists <= threshold
        valid_pts = pts[valid_mask]
        
        return {
            "success": True,
            "line_point": (cx, cy),
            "line_dir": (ux, uy),
            "straightness": float(straightness),
            "valid_points": valid_pts,
            "msg": "Start-End Line through Centroid"
        }