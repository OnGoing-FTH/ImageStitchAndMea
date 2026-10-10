'''
import numpy as np
import cv2

def fit_circle_robust(points, outlier_threshold=2.0):
    if points is None or len(points) < 3:
        return {"success": False, "msg": "点数不足3"}
    pts = np.array(points)

    def _fit(p):
        x, y = p[:, 0], p[:, 1]
        A = np.column_stack((2*x, 2*y, np.ones_like(x)))
        b = x**2 + y**2
        try:
            res, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            cx, cy = res[0], res[1]
            r = np.sqrt(res[2] + cx**2 + cy**2)
            return (cx, cy), r
        except:
            return (None, None), None

    (cx, cy), r = _fit(pts)
    if cx is None:
        return {"success": False, "msg": "拟合失败"}

    dists = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
    valid_pts = pts[np.abs(dists - r) < outlier_threshold]

    if len(valid_pts) < 3:
        return {"success": False, "msg": "有效点不足"}

    (fcx, fcy), fr = _fit(valid_pts)
    final_dists = np.hypot(valid_pts[:, 0] - fcx, valid_pts[:, 1] - fcy)
    circularity = final_dists.max() - final_dists.min()

    return {
        "success": True,
        "center": (fcx, fcy),
        "radius": fr,
        "circularity": float(circularity),
        "valid_points": valid_pts,
        "outlier_count": len(pts) - len(valid_pts)
    }


def fit_line_robust(points, outlier_threshold=2.0):
    if points is None or len(points) < 2:
        return {"success": False, "msg": "点数不足"}

    pts = np.array(points)

    def _fit(p):
        vx, vy, x0, y0 = cv2.fitLine(
            p.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01
        ).flatten()
        return (x0, y0), (vx, vy)

    p0, d = _fit(pts)
    vx, vy = d
    n = np.hypot(vx, vy)
    ux, uy = vx/n, vy/n

    diffs = pts - np.array(p0)
    dists = np.abs(diffs[:, 0] * uy - diffs[:, 1] * ux)
    valid_pts = pts[dists < outlier_threshold]

    if len(valid_pts) < 2:
        return {"success": False, "msg": "有效点不足"}

    p_final, d_final = _fit(valid_pts)
    vx, vy = d_final
    n = np.hypot(vx, vy)
    ux, uy = vx/n, vy/n

    diffs_f = valid_pts - np.array(p_final)
    signed = diffs_f[:, 0] * uy - diffs_f[:, 1] * ux
    straightness = signed.max() - signed.min()

    return {
        "success": True,
        "line_point": p_final,
        "line_dir": (ux, uy),
        "straightness": float(straightness),
        "valid_points": valid_pts,
        "outlier_count": len(pts) - len(valid_pts)
    }
'''


import numpy as np
import cv2

class FitRobust:
    def fit_circle(self, points, outlier_threshold=2.0, **kwargs):
        """
        鲁棒圆拟合 (剔除离群点)
        """
        if points is None or len(points) < 3:
            return {"success": False, "msg": "点数不足3"}
        pts = np.array(points)

        # 内部辅助函数
        def _fit_internal(p):
            x, y = p[:, 0], p[:, 1]
            A = np.column_stack((2*x, 2*y, np.ones_like(x)))
            b = x**2 + y**2
            try:
                res, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                cx, cy = res[0], res[1]
                r = np.sqrt(res[2] + cx**2 + cy**2)
                return (cx, cy), r
            except:
                return (None, None), None

        # 1. 初次拟合
        (cx, cy), r = _fit_internal(pts)
        if cx is None:
            return {"success": False, "msg": "初次拟合失败"}

        # 2. 剔除离群点
        dists = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
        valid_mask = np.abs(dists - r) < outlier_threshold
        valid_pts = pts[valid_mask]

        if len(valid_pts) < 3:
            return {"success": False, "msg": "剔除后有效点不足3个"}

        # 3. 二次拟合
        (fcx, fcy), fr = _fit_internal(valid_pts)
        if fcx is None:
             # 如果二次拟合失败，回退到初次结果，或报错
             return {"success": False, "msg": "二次拟合失败"}

        final_dists = np.hypot(valid_pts[:, 0] - fcx, valid_pts[:, 1] - fcy)
        circularity = final_dists.max() - final_dists.min()

        return {
            "success": True,
            "center": (fcx, fcy),
            "radius": fr,
            "circularity": float(circularity),
            "valid_points": valid_pts,
            "outlier_count": len(pts) - len(valid_pts)
        }

    def fit_line(self, points, outlier_threshold=2.0, **kwargs):
        """
        鲁棒直线拟合
        """
        if points is None or len(points) < 2:
            return {"success": False, "msg": "点数不足"}

        pts = np.array(points)

        def _fit_internal(p):
            if len(p) < 2: return None, None
            vx, vy, x0, y0 = cv2.fitLine(
                p.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01
            ).flatten()
            return (x0, y0), (vx, vy)

        # 1. 初次拟合
        res = _fit_internal(pts)
        if res[0] is None: return {"success": False, "msg": "初次拟合失败"}
        p0, d = res
        vx, vy = d
        n = np.hypot(vx, vy)
        if n == 0: return {"success": False, "msg": "向量模为0"}
        ux, uy = vx/n, vy/n

        # 2. 计算距离并剔除
        diffs = pts - np.array(p0)
        dists = np.abs(diffs[:, 0] * uy - diffs[:, 1] * ux)
        valid_pts = pts[dists < outlier_threshold]

        if len(valid_pts) < 2:
            return {"success": False, "msg": "剔除后有效点不足2个"}

        # 3. 二次拟合
        res_final = _fit_internal(valid_pts)
        if res_final[0] is None: return {"success": False, "msg": "二次拟合失败"}
        
        p_final, d_final = res_final
        vx, vy = d_final
        n = np.hypot(vx, vy)
        ux, uy = vx/n, vy/n

        diffs_f = valid_pts - np.array(p_final)
        signed = diffs_f[:, 0] * uy - diffs_f[:, 1] * ux
        straightness = signed.max() - signed.min()

        return {
            "success": True,
            "line_point": p_final,
            "line_dir": (ux, uy),
            "straightness": float(straightness),
            "valid_points": valid_pts,
            "outlier_count": len(pts) - len(valid_pts)
        }