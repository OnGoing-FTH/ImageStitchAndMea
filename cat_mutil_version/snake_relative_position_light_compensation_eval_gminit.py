"""
grid_stitch_snake_bmp.py

功能：
- 针对 S形(蛇形)扫描 的 BMP 图像进行拼接。
- 逻辑更新：
  1. 奇数行 (1, 3...)：从左到右 (1-1 -> 1-n)
  2. 偶数行 (2, 4...)：从右到左 (2-1 在 1-n 下方，2-n 在 1-1 下方)
  3. 自动将文件名映射到正确的逻辑网格坐标。
  4. 依然包含中位值修正功能，防止个别图片匹配错误带偏整体。这里使用中位数进行修正。
  5: 使用光照模型进行补偿

使用:
    python grid_stitch_snake_bmp.py
"""

import cv2
import numpy as np
import glob
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim

# ----------------- 配置 -----------------
IMG_FOLDER = "D:\paper_3\data1216\diandong6"       # 图片文件夹
OUTPUT_FILE = "diandong6_9.jpg"

USE_SIFT = True              # 推荐 True
RATIO_TEST = 0.75
MIN_MATCHES = 8              # 最小匹配点数
OUTLIER_THRESH = 0.05       # 异常值容忍度 (15%)
# -----------------------------------------

def parse_row_col(filename):
    basename = os.path.basename(filename)
    match = re.search(r'(\d+)[-_](\d+)', basename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None

# ---------- 1. 特征提取 ----------
# -------------------- 方案一：光照补偿法 --------------------

def extract_features(path):
    img = cv2.imread(path)
    if img is None: return None
    
    # 1. 转灰度
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # --- [核心修改] 光照估算与补偿 ---
    # 2. 估算光照模型：使用巨大的高斯核模糊原图
    # 核大小通常取图片短边的 1/20 到 1/10 左右，必须是奇数
    ksize = int(min(h, w) / 15) | 1 
    illumination_model = cv2.GaussianBlur(gray, (ksize, ksize), 0)

    # 3. 除去光照：原图 / 光照图
    # 加上 1e-5 防止除零，乘以 127 还原到平均灰度
    normalized = gray.astype(np.float32) / (illumination_model.astype(np.float32) + 1e-5) * 127.0
    normalized = np.clip(normalized, 0, 255).astype(np.uint8)
    
    # (可选) 稍微增加一点对比度，让特征更明显
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    final_img = clahe.apply(normalized)
    # ------------------------------------

    # 4. 特征提取 (保持不变)
    if USE_SIFT:
        detector = cv2.SIFT_create()
    else:
        # 如果去光后噪点较多，建议增加 ORB 的特征点数
        detector = cv2.ORB_create(10000) 
        
    kp, des = detector.detectAndCompute(final_img, None)
    
    return {
        'path': path,
        'shape': (w, h),
        'kp': kp,
        'des': des,
        'file_rc': parse_row_col(path)
    }
def batch_extract(image_paths):
    raw_db = []
    print(f"1. 正在提取 {len(image_paths)} 张图片的特征...")
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(extract_features, p): p for p in image_paths}
        for f in tqdm(as_completed(futures), total=len(futures)):
            res = f.result()
            if res and res['file_rc']:
                raw_db.append(res)
    return raw_db

# ---------- 2. 坐标映射 (S形逻辑核心) ----------

def map_to_logical_grid(raw_db):
    """
    将原始文件行列 (r, c) 映射为 逻辑网格坐标 (grid_x, grid_y)
    处理 S形 扫描：偶数行反转 X 轴
    """
    if not raw_db: return {}
    
    # 1. 找到最大的列号 N (决定画布宽度)
    all_cols = [item['file_rc'][1] for item in raw_db]
    max_col = max(all_cols)
    print(f"  检测到最大列号 N = {max_col} (用于偶数行反转)")

    logical_db = {} # Key: (grid_x, grid_y) -> Value: FeatureDict

    for item in raw_db:
        r, c = item['file_rc']
        
        # 逻辑 Y 就是行号
        grid_y = r 
        
        # 逻辑 X 根据奇偶行判断
        if r % 2 != 0:
            # 奇数行 (1, 3...): 左->右，x = c
            grid_x = c
        else:
            # 偶数行 (2, 4...): 右->左
            # 例如 N=5。2-1 (第1张) 应该在最右边 (5)。2-5 应该在最左边 (1)。
            # 公式: x = N - c + 1
            grid_x = max_col - c + 1
            
        # 保存新的逻辑坐标
        item['logical_pos'] = (grid_x, grid_y)
        logical_db[(grid_x, grid_y)] = item
        
    return logical_db

# ---------- 3. 计算位移与修正 ----------

# def compute_translation(item_src, item_dst):
#     # 代码同上，只计算纯平移
#     kp1, des1 = item_src['kp'], item_src['des']
#     kp2, des2 = item_dst['kp'], item_dst['des']
    
#     if des1 is None or des2 is None or len(des1) < MIN_MATCHES or len(des2) < MIN_MATCHES:
#         return None

#     if USE_SIFT:
#         matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
#         knn_matches = matcher.knnMatch(des1, des2, k=2)
#     else:
#         matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
#         knn_matches = matcher.knnMatch(des1, des2, k=2)

#     good = []
#     for m_n in knn_matches:
#         if len(m_n) != 2: continue
#         m, n = m_n
#         if m.distance < RATIO_TEST * n.distance:
#             good.append(m)

#     if len(good) < MIN_MATCHES:
#         return None

#     pts_src = np.float32([kp1[m.queryIdx].pt for m in good])
#     pts_dst = np.float32([kp2[m.trainIdx].pt for m in good])

#     diff = pts_dst - pts_src
#     dx = np.median(diff[:, 0])
#     dy = np.median(diff[:, 1])
#     return np.array([dx, dy])


def compute_translation(item_src, item_dst):
    """
    计算 src -> dst 的平移量，并打印调试信息
    """
    # 获取文件名方便打印
    name_src = os.path.basename(item_src['path'])
    name_dst = os.path.basename(item_dst['path'])

    kp1, des1 = item_src['kp'], item_src['des']
    kp2, des2 = item_dst['kp'], item_dst['des']
    
    # 1. 检查特征点是否存在
    if des1 is None or des2 is None or len(des1) < MIN_MATCHES or len(des2) < MIN_MATCHES:
        print(f"  [跳过] {name_src} <-> {name_dst}: 特征点不足")
        return None

    # 2. 特征匹配
    if USE_SIFT:
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

    # 3. 检查匹配数量
    if len(good) < MIN_MATCHES:
        print(f"  [失败] {name_src} <-> {name_dst}: 有效匹配太少 ({len(good)})")
        return None

    # 4. 计算位移中位数
    pts_src = np.float32([kp1[m.queryIdx].pt for m in good])
    pts_dst = np.float32([kp2[m.trainIdx].pt for m in good])

    diff = pts_dst - pts_src
    dx = np.median(diff[:, 0])
    dy = np.median(diff[:, 1])
    
    # --- [新增] 打印计算结果 ---
    # dx: 水平平移 (负数代表 dst 在 src 的右边，如果是左右拼接)
    # dy: 垂直平移 (负数代表 dst 在 src 的下边，如果是上下拼接)
    print(f"  [计算] {name_src} -> {name_dst}: dx={dx:7.2f}, dy={dy:7.2f} (匹配点:{len(good)})")
    
    return np.array([dx, dy])

def build_graph(logical_db):
    print("2. 基于逻辑坐标构建连接图 (S形已展开)...")
    
    raw_edges = []
    keys = sorted(logical_db.keys()) # 排序 (grid_x, grid_y)
    
    # 遍历逻辑网格
    for (lx, ly) in keys:
        # 1. 找右边的邻居 (Logic X + 1)
        if (lx + 1, ly) in logical_db:
            shift = compute_translation(logical_db[(lx,ly)], logical_db[(lx+1, ly)])
            raw_edges.append({
                'src': (lx,ly), 'dst': (lx+1, ly), 
                'shift': shift, 'type': 'H'
            })
            
        # 2. 找下面的邻居 (Logic Y + 1)
        if (lx, ly + 1) in logical_db:
            shift = compute_translation(logical_db[(lx,ly)], logical_db[(lx, ly+1)])
            raw_edges.append({
                'src': (lx,ly), 'dst': (lx, ly+1), 
                'shift': shift, 'type': 'V'
            })

    # --- 统计与修正逻辑 ---
    valid_h = [e['shift'] for e in raw_edges if e['type'] == 'H' and e['shift'] is not None]
    valid_v = [e['shift'] for e in raw_edges if e['type'] == 'V' and e['shift'] is not None]
    
    h_median = np.median(valid_h, axis=0) if valid_h else np.array([0., 0.])
    v_median = np.median(valid_v, axis=0) if valid_v else np.array([0., 0.])
    
    print(f"  [统计] 逻辑水平步长: {h_median}")
    print(f"  [统计] 逻辑垂直步长: {v_median}")
    
    adj_list = {}
    fixed_count = 0
    
    for edge in raw_edges:
        target = h_median if edge['type'] == 'H' else v_median
        computed = edge['shift']
        
        # 修正判定
        use_median = False
        if computed is None:
            use_median = True
        else:
            dist = np.linalg.norm(computed - target)
            ref = np.linalg.norm(target)
            if ref > 0 and (dist / ref > OUTLIER_THRESH) and dist > 10:
                use_median = True
        
        final_vec = target if use_median else computed
        if use_median: fixed_count += 1
        
        # 存入图
        src, dst = edge['src'], edge['dst']
        if src not in adj_list: adj_list[src] = []
        if dst not in adj_list: adj_list[dst] = []
        
        adj_list[src].append({'to': dst, 'vec': final_vec})
        adj_list[dst].append({'to': src, 'vec': -final_vec})
        
    print(f"  [修正] 已修正/补全 {fixed_count} 条连接。")
    return adj_list

# ---------- 4. 全局坐标与渲染 ----------

def calc_positions(logical_db, adj_list):
    print("3. 推导全局坐标 (BFS)...")
    if not logical_db: return {}
    
    # 找左上角作为起点 (通常是 1,1)
    # 如果 1,1 缺失 (背景)，找最小的存在的 (x,y)
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

def render(logical_db, positions):
    print("4. 渲染大图并准备评估数据...")
    if not positions: 
        print("无坐标数据。")
        return None, None, None

    # 计算画布范围
    all_x, all_y = [], []
    for k, pos in positions.items():
        w, h = logical_db[k]['shape']
        all_x.append(pos[0])
        all_x.append(pos[0] + w)
        all_y.append(pos[1])
        all_y.append(pos[1] + h)
        
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    
    cv_w = int(max_x - min_x)
    cv_h = int(max_y - min_y)
    print(f"  画布尺寸: {cv_w} x {cv_h}")

    off_x, off_y = -min_x, -min_y

    try:
        # canvas: 前三通道存BGR之和，第四通道存计数(weights)
        canvas = np.zeros((cv_h, cv_w, 4), dtype=np.float32)
        # canvas_sq: 存BGR像素值的平方和，用于计算方差
        canvas_sq = np.zeros((cv_h, cv_w, 3), dtype=np.float32)
    except MemoryError:
        print("错误：内存不足，无法创建画布。请尝试缩小图像或分块处理。")
        return None, None, None
    
    for k, pos in tqdm(positions.items(), desc="累加图像像素"):
        path = logical_db[k]['path']
        img = cv2.imread(path)
        if img is None: continue
        
        img_f = img.astype(np.float32) # 转为浮点数进行计算
        h, w = img_f.shape[:2]
        x_s = int(pos[0] + off_x)
        y_s = int(pos[1] + off_y)
        
        if x_s < 0 or y_s < 0 or x_s+w > cv_w or y_s+h > cv_h: continue
        
        # 累加原值和权重
        canvas[y_s:y_s+h, x_s:x_s+w, :3] += img_f
        canvas[y_s:y_s+h, x_s:x_s+w, 3] += 1.0
        # 累加平方值
        canvas_sq[y_s:y_s+h, x_s:x_s+w] += img_f ** 2
        
    # 计算均值融合结果
    weights = canvas[:,:,3]
    mask = weights > 0
    weights_3 = np.stack([weights[mask]] * 3, axis=1)
    
    # 融合图结果 (res)
    res_f = np.zeros((cv_h, cv_w, 3), dtype=np.float32)
    res_f[mask] = canvas[mask, :3] / weights_3
    
    # 计算重叠区域方差图 (var_map)
    # 公式: Var = E[X^2] - (E[X])^2
    var_map = np.zeros((cv_h, cv_w, 3), dtype=np.float32)
    overlap_mask = weights > 1 # 只在有重叠的像素点计算方差
    if np.any(overlap_mask):
        w_overlap_3 = np.stack([weights[overlap_mask]] * 3, axis=1)
        e_x2 = canvas_sq[overlap_mask] / w_overlap_3  # 像素平方的均值
        e_x_sq = (canvas[overlap_mask, :3] / w_overlap_3) ** 2 # 像素均值的平方
        var_map[overlap_mask] = np.maximum(0, e_x2 - e_x_sq) # 确保不出现微小的负数误差
    
    # 转回uint8并保存
    res = np.clip(res_f, 0, 255).astype(np.uint8)
    cv2.imwrite(OUTPUT_FILE, res)
    print(f"完成，已保存至 {OUTPUT_FILE}")
    
    # 返回评估所需的三个关键对象
    return res, var_map, weights

# eval images equallity


class StitchingEvaluator:
    def __init__(self, logical_db, positions, stitched_img, var_map, weights):
        self.logical_db = logical_db
        self.positions = positions
        self.img_out = stitched_img
        self.var_map = var_map
        self.weights = weights
        
        # 坐标偏移对齐
        all_x = [pos[0] for pos in positions.values()]
        all_y = [pos[1] for pos in positions.values()]
        self.off_x, self.off_y = -min(all_x), -min(all_y)

    def calculate_metrics(self):
        print("\n--- 拼接融合质量评估报告 ---")
        
        rmse_list = []
        ssim_list = []
        color_diff_list = []
        
        for k, pos in self.positions.items():
            # 1. 加载并预处理子图
            img_src = cv2.imread(self.logical_db[k]['path'])
            h, w = img_src.shape[:2]
            x_s, y_s = int(pos[0] + self.off_x), int(pos[1] + self.off_y)
            
            # 2. 提取大图中对应的ROI
            img_roi = self.img_out[y_s:y_s+h, x_s:x_s+w]
            if img_roi.shape != img_src.shape: continue

            # --- 指标 A: RMSE (对齐精度) ---
            mse = np.mean((img_src.astype(np.float32) - img_roi.astype(np.float32)) ** 2)
            rmse_list.append(np.sqrt(mse))

            # --- 指标 B: SSIM (结构相似性) ---
            g_src = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
            g_roi = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
            s = ssim(g_src, g_roi)
            ssim_list.append(s)

            # --- 指标 C: 色彩一致性 (Color Consistency) ---
            # 计算子图与融合后对应区域的亮度均值差异
            mean_src = np.mean(img_src, axis=(0,1))
            mean_roi = np.mean(img_roi, axis=(0,1))
            color_diff_list.append(np.linalg.norm(mean_src - mean_roi))

        # --- 指标 D: 重叠区域方差 (Overlap Variance) ---
        # 反映重叠位置多个像素点的一致性，越小说明对齐越准
        avg_overlap_var = np.mean(self.var_map[self.weights > 1])

        # --- 指标 E: 缝合线强度 (SD - Seam Difference) ---
        # 评估接缝处是否存在突变
        sd_value = self._calculate_seam_difference()

        print(f"1. 平均 RMSE (对齐误差): {np.mean(rmse_list):.4f} (越小对齐越准)")
        print(f"2. 平均 SSIM (结构保持): {np.mean(ssim_list):.4f} (越接近1越好)")
        print(f"3. 重叠区像素方差: {avg_overlap_var:.4f} (越小说明重叠区融合越无痕)")
        print(f"4. 色彩一致性差异: {np.mean(color_diff_list):.4f} (越小说明光照补偿越自然)")
        print(f"5. 缝合线跳变强度 (SD): {sd_value:.4f} (越小说明接缝越不明显)")

    def _calculate_seam_difference(self):
        """
        计算横向和纵向相邻图片接缝处的梯度差异
        """
        # 简化计算：通过计算融合图像在重叠边缘处的拉普拉斯梯度
        gray_out = cv2.cvtColor(self.img_out, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray_out, cv2.CV_32F)
        # 只看重叠区域的梯度，如果对齐不好，重叠边缘会有很强的伪梯度
        seam_grad = np.abs(laplacian[self.weights > 1])
        return np.mean(seam_grad)

# ---------- Main ----------

def main():
    exts = ['*.bmp', '*.jpg', '*.png']
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(IMG_FOLDER, e)))
        
    if not paths:
        print("未找到图片。")
        return

    # 1. 提取并携带原始行列信息
    raw_db = batch_extract(paths)
    
    # 2. 映射到 S形 逻辑坐标
    logical_db = map_to_logical_grid(raw_db)
    
    # 3. 基于逻辑坐标建立连接图
    adj = build_graph(logical_db)
    
    # 4. 计算
    pos = calc_positions(logical_db, adj)
    
    # 5. 渲染及计算误差
    final_img, var_map, weights = render(logical_db, pos)
    
    if final_img is not None:
        # 2. 调用评估类 (StitchingEvaluator 类定义参考上一个回复)
        evaluator = StitchingEvaluator(logical_db, pos, final_img, var_map, weights)
        evaluator.calculate_metrics()

if __name__ == "__main__":
    main()