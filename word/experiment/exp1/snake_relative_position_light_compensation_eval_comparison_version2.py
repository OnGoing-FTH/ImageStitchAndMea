"""

使用的clahe的光照模型
grid_stitch_ablation.py

功能：S形扫描拼接对比实验 (消融实验)
更新：光照模型已替换为 "Gaussian Blur + CLAHE" 以适应金属拉丝表面的移动光源场景。

实验组：
1. 去除光照, 去除纠正, SIFT
2. 去除光照, 去除纠正, ORB
3. 保留光照, 去除纠正, SIFT
4. 保留光照, 去除纠正, ORB
5. 去除光照, 保留纠正, ORB
6. 去除光照, 保留纠正, SIFT
"""

import cv2
import numpy as np
import glob
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim

# ----------------- 基础配置 -----------------
IMG_FOLDER = r"D:\paper_3\word\experiment\exp1\data129"  # 请确保路径正确
RATIO_TEST = 0.75
MIN_MATCHES = 8
OUTLIER_THRESH = 0.05
# -----------------------------------------

def parse_row_col(filename):
    basename = os.path.basename(filename)
    match = re.search(r'(\d+)[-_](\d+)', basename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None

# ---------- 1. 特征提取 (核心修改部分) ----------

def extract_features(args):
    """
    args: (path, use_light, feature_type)
    """
    path, use_light, feature_type = args
    img = cv2.imread(path)
    if img is None: return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # --- 模块：光照补偿 (已修改为 CLAHE 方案) ---
    if use_light:
        # 步骤 1: 轻微高斯模糊
        # 目的: 抑制金属表面的高频颗粒噪声。如果不做这一步，CLAHE会把颗粒放大成特征点。
        # kernel=(3,3) 是比较保守的选择，如果噪点依然多，可改为 (5,5)
        gray_blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # 步骤 2: 自适应直方图均衡化 (CLAHE)
        # 目的: 强行拉平局部光照差异，对抗“中心亮边缘暗”以及“动态高光”。
        # clipLimit=3.0: 限制对比度放大的倍数。值越大，纹理越清晰，但噪声也越大。建议 2.0 ~ 4.0。
        # tileGridSize=(8,8): 将图像切分为 8x8 的网格进行局部处理。
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        final_img = clahe.apply(gray_blurred)
    else:
        # 不使用光照补偿，直接使用原灰度图
        final_img = gray

    # --- 模块：特征描述符 ---
    if feature_type == 'SIFT':
        # SIFT 对光照变化的鲁棒性通常优于 ORB，配合 CLAHE 效果更佳
        detector = cv2.SIFT_create()
    else:
        # ORB 
        detector = cv2.ORB_create(10000) 
        
    kp, des = detector.detectAndCompute(final_img, None)
    
    return {
        'path': path,
        'shape': gray.shape[::-1], # (w, h)
        'kp': kp,
        'des': des,
        'file_rc': parse_row_col(path)
    }

def batch_extract(image_paths, use_light, feature_type):
    raw_db = []
    
    # 包装参数
    tasks = [(p, use_light, feature_type) for p in image_paths]
    
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(extract_features, t): t for t in tasks}
        for f in tqdm(as_completed(futures), total=len(futures), leave=False, desc=f"提取特征({feature_type}, Light={use_light})"):
            res = f.result()
            if res and res['file_rc']:
                raw_db.append(res)
    return raw_db

# ---------- 2. 坐标映射 (保持不变) ----------

def map_to_logical_grid(raw_db):
    if not raw_db: return {}
    all_cols = [item['file_rc'][1] for item in raw_db]
    max_col = max(all_cols)
    logical_db = {} 
    for item in raw_db:
        r, c = item['file_rc']
        grid_y = r 
        if r % 2 != 0:
            grid_x = c
        else:
            grid_x = max_col - c + 1
        item['logical_pos'] = (grid_x, grid_y)
        logical_db[(grid_x, grid_y)] = item
    return logical_db

# ---------- 3. 计算位移与构建图 (保持不变) ----------

def compute_translation(item_src, item_dst, feature_type):
    kp1, des1 = item_src['kp'], item_src['des']
    kp2, des2 = item_dst['kp'], item_dst['des']
    
    if des1 is None or des2 is None or len(des1) < MIN_MATCHES or len(des2) < MIN_MATCHES:
        return None

    if feature_type == 'SIFT':
        matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
        knn_matches = matcher.knnMatch(des1, des2, k=2)
    else:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        knn_matches = matcher.knnMatch(des1, des2, k=2)

    good = []
    for m_n in knn_matches:
        if len(m_n) != 2: continue
        m, n = m_n
        if m.distance < RATIO_TEST * n.distance:
            good.append(m)

    if len(good) < MIN_MATCHES:
        return None

    pts_src = np.float32([kp1[m.queryIdx].pt for m in good])
    pts_dst = np.float32([kp2[m.trainIdx].pt for m in good])

    diff = pts_dst - pts_src
    dx = np.median(diff[:, 0])
    dy = np.median(diff[:, 1])
    return np.array([dx, dy])

def build_graph(logical_db, feature_type, use_correction):
    raw_edges = []
    keys = sorted(logical_db.keys())
    
    # 1. 计算所有相邻边的位移
    for (lx, ly) in keys:
        # 右邻居
        if (lx + 1, ly) in logical_db:
            shift = compute_translation(logical_db[(lx,ly)], logical_db[(lx+1, ly)], feature_type)
            raw_edges.append({'src': (lx,ly), 'dst': (lx+1, ly), 'shift': shift, 'type': 'H'})
            
        # 下邻居
        if (lx, ly + 1) in logical_db:
            shift = compute_translation(logical_db[(lx,ly)], logical_db[(lx, ly+1)], feature_type)
            raw_edges.append({'src': (lx,ly), 'dst': (lx, ly+1), 'shift': shift, 'type': 'V'})

    # 2. 统计中位值 (先验知识)
    valid_h = [e['shift'] for e in raw_edges if e['type'] == 'H' and e['shift'] is not None]
    valid_v = [e['shift'] for e in raw_edges if e['type'] == 'V' and e['shift'] is not None]
    
    h_median = np.median(valid_h, axis=0) if valid_h else np.array([0., 0.])
    v_median = np.median(valid_v, axis=0) if valid_v else np.array([0., 0.])
    
    adj_list = {}
    fixed_count = 0
    
    # 3. 过滤与纠正
    for edge in raw_edges:
        target = h_median if edge['type'] == 'H' else v_median
        computed = edge['shift']
        
        use_median_val = False
        
        if computed is None:
            use_median_val = True 
        else:
            if use_correction:
                dist = np.linalg.norm(computed - target)
                ref = np.linalg.norm(target)
                if ref > 0 and (dist / ref > OUTLIER_THRESH) and dist > 10:
                    use_median_val = True
            else:
                use_median_val = False
        
        final_vec = target if use_median_val else computed
        if use_median_val and computed is not None: 
            fixed_count += 1
        
        src, dst = edge['src'], edge['dst']
        if src not in adj_list: adj_list[src] = []
        if dst not in adj_list: adj_list[dst] = []
        
        adj_list[src].append({'to': dst, 'vec': final_vec})
        adj_list[dst].append({'to': src, 'vec': -final_vec})
        
    return adj_list, fixed_count

# ---------- 4. 坐标计算与渲染 (保持不变) ----------

def calc_positions(logical_db, adj_list):
    if not logical_db: return {}
    sorted_keys = sorted(logical_db.keys())
    root = sorted_keys[0]
    positions = {root: np.array([0.0, 0.0])}
    queue = [root]
    visited = {root}
    
    while queue:
        curr = queue.pop(0)
        curr_pos = positions[curr]
        if curr in adj_list:
            for edge in adj_list[curr]:
                neighbor = edge['to']
                vec = edge['vec']
                if neighbor not in visited:
                    positions[neighbor] = curr_pos - vec
                    visited.add(neighbor)
                    queue.append(neighbor)
    return positions

def render(logical_db, positions, output_name):
    if not positions: return None, None, None

    all_x = [pos[0] for pos in positions.values()]
    all_y = [pos[1] for pos in positions.values()]
    w_img, h_img = list(logical_db.values())[0]['shape']
    
    min_x, max_x = min(all_x), max(all_x) + w_img
    min_y, max_y = min(all_y), max(all_y) + h_img
    
    cv_w = int(max_x - min_x)
    cv_h = int(max_y - min_y)
    off_x, off_y = -min_x, -min_y

    try:
        canvas = np.zeros((cv_h, cv_w, 4), dtype=np.float32)
        canvas_sq = np.zeros((cv_h, cv_w, 3), dtype=np.float32)
    except MemoryError:
        print("内存不足。")
        return None, None, None
    
    for k, pos in positions.items():
        img = cv2.imread(logical_db[k]['path'])
        if img is None: continue
        img_f = img.astype(np.float32)
        h, w = img_f.shape[:2]
        x_s = int(pos[0] + off_x)
        y_s = int(pos[1] + off_y)
        
        if x_s < 0 or y_s < 0 or x_s+w > cv_w or y_s+h > cv_h: continue
        
        canvas[y_s:y_s+h, x_s:x_s+w, :3] += img_f
        canvas[y_s:y_s+h, x_s:x_s+w, 3] += 1.0
        canvas_sq[y_s:y_s+h, x_s:x_s+w] += img_f ** 2
        
    weights = canvas[:,:,3]
    mask = weights > 0
    weights_3 = np.stack([weights[mask]] * 3, axis=1)
    
    res_f = np.zeros((cv_h, cv_w, 3), dtype=np.float32)
    res_f[mask] = canvas[mask, :3] / weights_3
    
    var_map = np.zeros((cv_h, cv_w, 3), dtype=np.float32)
    overlap_mask = weights > 1
    if np.any(overlap_mask):
        w_overlap_3 = np.stack([weights[overlap_mask]] * 3, axis=1)
        e_x2 = canvas_sq[overlap_mask] / w_overlap_3
        e_x_sq = (canvas[overlap_mask, :3] / w_overlap_3) ** 2
        var_map[overlap_mask] = np.maximum(0, e_x2 - e_x_sq)
    
    res = np.clip(res_f, 0, 255).astype(np.uint8)
    if output_name:
        cv2.imwrite(output_name, res)
    return res, var_map, weights

# ---------- 5. 评估类 (保持不变) ----------

class StitchingEvaluator:
    def __init__(self, logical_db, positions, stitched_img, var_map, weights):
        self.logical_db = logical_db
        self.positions = positions
        self.img_out = stitched_img
        self.var_map = var_map
        self.weights = weights
        all_x = [pos[0] for pos in positions.values()]
        all_y = [pos[1] for pos in positions.values()]
        self.off_x, self.off_y = -min(all_x), -min(all_y)

    def calculate_metrics(self):
        rmse_list = []
        ssim_list = []
        color_diff_list = []
        
        for k, pos in self.positions.items():
            img_src = cv2.imread(self.logical_db[k]['path'])
            h, w = img_src.shape[:2]
            x_s, y_s = int(pos[0] + self.off_x), int(pos[1] + self.off_y)
            img_roi = self.img_out[y_s:y_s+h, x_s:x_s+w]
            if img_roi.shape != img_src.shape: continue

            mse = np.mean((img_src.astype(np.float32) - img_roi.astype(np.float32)) ** 2)
            rmse_list.append(np.sqrt(mse))

            g_src = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
            g_roi = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
            s = ssim(g_src, g_roi)
            ssim_list.append(s)

            mean_src = np.mean(img_src, axis=(0,1))
            mean_roi = np.mean(img_roi, axis=(0,1))
            color_diff_list.append(np.linalg.norm(mean_src - mean_roi))

        if np.any(self.weights > 1):
            avg_overlap_var = np.mean(self.var_map[self.weights > 1])
        else:
            avg_overlap_var = 0.0

        sd_value = self._calculate_seam_difference()

        return {
            "RMSE": np.mean(rmse_list),
            "SSIM": np.mean(ssim_list),
            "Var": avg_overlap_var,
            "ColorDiff": np.mean(color_diff_list),
            "SeamDiff": sd_value
        }

    def _calculate_seam_difference(self):
        gray_out = cv2.cvtColor(self.img_out, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray_out, cv2.CV_32F)
        mask = self.weights > 1
        if np.any(mask):
            seam_grad = np.abs(laplacian[mask])
            return np.mean(seam_grad)
        return 0.0

# ---------- Main Execution Block ----------

def run_single_experiment(exp_name, use_light, use_correction, feature_type, paths):
    print(f"\n=== 运行实验: {exp_name} ===")
    print(f"    [配置] Light={use_light} | Correction={use_correction} | Feature={feature_type}")

    raw_db = batch_extract(paths, use_light, feature_type)
    logical_db = map_to_logical_grid(raw_db)
    adj, fixed_num = build_graph(logical_db, feature_type, use_correction)
    print(f"    [过程] 纠正/补全边数: {fixed_num}")
    
    pos = calc_positions(logical_db, adj)
    out_file = f"result_{exp_name}.jpg"
    final_img, var_map, weights = render(logical_db, pos, out_file)
    
    metrics = {}
    if final_img is not None:
        evaluator = StitchingEvaluator(logical_db, pos, final_img, var_map, weights)
        metrics = evaluator.calculate_metrics()
        print(f"    [结果] RMSE={metrics['RMSE']:.2f}, SSIM={metrics['SSIM']:.3f}, Var={metrics['Var']:.2f}")
    
    return metrics

def main():
    exts = ['*.bmp', '*.jpg', '*.png']
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(IMG_FOLDER, e)))
    if not paths:
        print("未找到图片，请检查路径。")
        return

    experiments = [
        ("1_NoLight_NoCorr_SIFT", False, False, 'SIFT'),
        ("2_NoLight_NoCorr_ORB",  False, False, 'ORB'),
        # ("3_Light_NoCorr_SIFT",   True,  False, 'SIFT'), # 这里现在意味着使用 CLAHE 增强
        # ("4_Light_NoCorr_ORB",    True,  False, 'ORB'),  # 这里现在意味着使用 CLAHE 增强
        ("5_NoLight_Corr_ORB",    False, True,  'ORB'),
        ("6_NoLight_Corr_SIFT",   False, True,  'SIFT'),
        # ("0_Full_Baseline",       True,  True,  'SIFT'),
    ]

    results = []

    for params in experiments:
        name, light, corr, feat = params
        m = run_single_experiment(name, light, corr, feat, paths)
        results.append({
            "Name": name,
            **m
        })

    print("\n\n" + "="*85)
    header = f"{'Experiment':<25} | {'RMSE':<8} | {'SSIM':<8} | {'Var':<8} | {'ColorDiff':<10} | {'SeamDiff'}"
    print(header)
    print("-" * 85)
    
    for r in results:
        if 'RMSE' in r:
            print(f"{r['Name']:<25} | "
                  f"{r['RMSE']:<8.2f} | "
                  f"{r['SSIM']:<8.4f} | "
                  f"{r['Var']:<8.2f} | "
                  f"{r['ColorDiff']:<10.2f} | "
                  f"{r['SeamDiff']:.2f}")
        else:
            print(f"{r['Name']:<25} | FAILED")
    print("="*85)

if __name__ == "__main__":
    main()