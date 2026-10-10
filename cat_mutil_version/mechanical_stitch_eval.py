import cv2
import numpy as np
import pandas as pd
import glob
import os
from skimage.metrics import structural_similarity as ssim

# ----------------- 配置区域 -----------------
IMG_FOLDER = r"D:\paper_3\data1216\diandong6"    # 图片文件夹
EXCEL_PATH = r"D:\paper_3\word\experiment\exp1\点动6_displancement.xlsx"  # Excel 路径

# *** 关键参数: 分辨率 (像素/毫米) ***
# 含义：1mm 的物理移动对应多少个像素。
# 你可以用 (图片宽度像素 / 视野宽度毫米) 估算。
# 或者：查看 1-1 到 1-2 的 X 差值是 6mm，如果图片位移是 1980像素，则值为 330。
PIXELS_PER_MM = 117.34  

# 4. 坐标轴方向修正
#    如果拼出来左右反了 -> 改为 True
FLIP_X = False
#    如果拼出来上下反了 -> 改为 True (通常机械Y轴向上，图像坐标Y轴向下，所以常需翻转)
FLIP_Y = True
# ===========================================

class StitchingEvaluator:
    """
    评估器：计算 RMSE, SSIM, Var (重影方差), ColorDiff (色差), SeamDiff (拼缝差异)
    """
    def __init__(self, logical_db, positions, stitched_img, var_map, weights):
        self.logical_db = logical_db
        self.positions = positions
        self.img_out = stitched_img
        self.var_map = var_map
        self.weights = weights
        # 计算偏移量，用于将局部坐标转为大图坐标
        all_x = [pos[0] for pos in positions.values()]
        all_y = [pos[1] for pos in positions.values()]
        self.off_x, self.off_y = -min(all_x), -min(all_y)

    def calculate_metrics(self):
        rmse_list = []
        ssim_list = []
        color_diff_list = []
        
        # 遍历每张原图，与拼接结果中的对应区域进行比较
        for k, pos in self.positions.items():
            path = self.logical_db[k]['path']
            if not os.path.exists(path): continue
            
            img_src = cv2.imread(path)
            if img_src is None: continue
            
            h, w = img_src.shape[:2]
            # 计算在大图中的位置
            x_s, y_s = int(pos[0] + self.off_x), int(pos[1] + self.off_y)
            
            # 从拼接大图中切出 ROI
            img_roi = self.img_out[y_s:y_s+h, x_s:x_s+w]
            
            # 尺寸校验 (防止边缘溢出导致尺寸不一致)
            if img_roi.shape != img_src.shape: continue

            # --- 指标 1: RMSE (均方根误差) ---
            mse = np.mean((img_src.astype(np.float32) - img_roi.astype(np.float32)) ** 2)
            rmse_list.append(np.sqrt(mse))

            # --- 指标 2: SSIM (结构相似性) ---
            g_src = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
            g_roi = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
            s = ssim(g_src, g_roi)
            ssim_list.append(s)

            # --- 指标 4: ColorDiff (平均色差) ---
            # 计算两张图 R,G,B 通道的均值之差的欧氏距离
            mean_src = np.mean(img_src, axis=(0,1))
            mean_roi = np.mean(img_roi, axis=(0,1))
            color_diff_list.append(np.linalg.norm(mean_src - mean_roi))

        # --- 指标 3: Variance (重叠区域方差 - 衡量鬼影) ---
        # var_map 在渲染阶段已经计算好了，是像素值的方差
        if np.any(self.weights > 1):
            # 只统计重叠区域 (权重 > 1)
            avg_overlap_var = np.mean(self.var_map[self.weights > 1])
        else:
            avg_overlap_var = 0.0

        # --- 指标 5: SeamDiff (拼缝差异度) ---
        sd_value = self._calculate_seam_difference()

        return {
            "Name": "Mechanical_Excel", # 固定名称方便打印
            "RMSE": np.mean(rmse_list) if rmse_list else 0,
            "SSIM": np.mean(ssim_list) if ssim_list else 0,
            "Var": avg_overlap_var,
            "ColorDiff": np.mean(color_diff_list) if color_diff_list else 0,
            "SeamDiff": sd_value
        }

    def _calculate_seam_difference(self):
        """利用拉普拉斯算子检测重叠区域的高频边缘（拼缝）"""
        gray_out = cv2.cvtColor(self.img_out, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray_out, cv2.CV_32F)
        # 只关注重叠区域
        mask = self.weights > 1 
        if np.any(mask):
            # 取绝对值后的平均值，值越大说明拼缝处的跳变越剧烈
            seam_grad = np.abs(laplacian[mask])
            return np.mean(seam_grad)
        return 0.0

# ----------------- 数据处理模块 -----------------

def load_and_map_coordinates(img_folder, excel_path):
    print(f"正在读取 Excel: {excel_path} ...")
    
    # header=None: 假设没有表头，第一行就是数据
    df = pd.read_excel(excel_path, header=None)
    
    # 简单的检查：如果第一行包含字母，可能是有表头的，重新读一次
    first_val = str(df.iloc[0, 0])
    if "X" in first_val or "ID" in first_val or "name" in first_val.lower():
        print("检测到表头，重新读取...")
        df = pd.read_excel(excel_path, header=0)

    # 扫描图片文件
    exts = ['*.bmp', '*.jpg', '*.png', '*.tif']
    img_files_map = {} # {'1-1': 'fullpath'}
    
    raw_files = []
    for e in exts:
        raw_files.extend(glob.glob(os.path.join(img_folder, e)))
    
    for p in raw_files:
        fname = os.path.basename(p)
        name_no_ext = os.path.splitext(fname)[0]
        img_files_map[name_no_ext] = p

    logical_db = {}
    mech_positions = {}

    print(f"正在匹配坐标 (Excel行数: {len(df)})...")
    
    matched_count = 0
    # 遍历 Excel 行
    for index, row in df.iterrows():
        try:
            # 假设格式: 第0列=ID, 第1列=X, 第2列=Y
            img_id = str(row.iloc[0]).strip() # '1-1'
            x_mm = float(row.iloc[1])
            y_mm = float(row.iloc[2])
            # z_mm = float(row.iloc[3]) # Z轴暂时不用
        except Exception:
            continue

        # 尝试匹配文件名
        target_path = None
        if img_id in img_files_map:
            target_path = img_files_map[img_id]
        else:
            # 尝试替换 - 为 _ 的容错匹配
            alt_id = img_id.replace('-', '_')
            if alt_id in img_files_map:
                target_path = img_files_map[alt_id]

        if target_path:
            logical_db[img_id] = {'path': target_path, 'shape': None}
            mech_positions[img_id] = np.array([x_mm, y_mm])
            matched_count += 1

    print(f"匹配完成: {matched_count} 张图片。")
    return logical_db, mech_positions

def convert_mech_to_pixel(mech_positions, px_per_mm, flip_x=False, flip_y=True):
    if not mech_positions: return {}
    positions_px = {}
    
    coords = np.array(list(mech_positions.values()))
    # 找出物理坐标的边界
    min_x_mm = np.min(coords[:, 0])
    max_x_mm = np.max(coords[:, 0])
    min_y_mm = np.min(coords[:, 1])
    max_y_mm = np.max(coords[:, 1])
    
    for key, pos_mm in mech_positions.items():
        x_mm, y_mm = pos_mm
        
        # X 轴映射
        if flip_x:
            px_x = (max_x_mm - x_mm) * px_per_mm
        else:
            px_x = (x_mm - min_x_mm) * px_per_mm
            
        # Y 轴映射 (通常需要翻转，因为图像坐标系Y向下)
        if flip_y:
            px_y = (max_y_mm - y_mm) * px_per_mm
        else:
            px_y = (y_mm - min_y_mm) * px_per_mm
            
        positions_px[key] = np.array([px_x, px_y])
        
    return positions_px

def render_mechanical(logical_db, positions, output_name):
    if not logical_db: return None, None, None
    
    # 读取第一张图获取尺寸
    first_key = list(logical_db.keys())[0]
    sample_img = cv2.imread(logical_db[first_key]['path'])
    h_img, w_img = sample_img.shape[:2]
    
    for k in logical_db: logical_db[k]['shape'] = (w_img, h_img)
    
    # 计算画布大小
    all_x = [pos[0] for pos in positions.values()]
    all_y = [pos[1] for pos in positions.values()]
    min_x, max_x = min(all_x), max(all_x) + w_img
    min_y, max_y = min(all_y), max(all_y) + h_img
    
    cv_w = int(max_x - min_x)
    cv_h = int(max_y - min_y)
    off_x, off_y = -min_x, -min_y
    
    # 初始化画布 (带权重通道)
    # 通道0-2: BGR累加值, 通道3: 权重(重叠次数)
    canvas = np.zeros((cv_h, cv_w, 4), dtype=np.float32)
    # 辅助画布: 像素平方和 (用于计算方差)
    canvas_sq = np.zeros((cv_h, cv_w, 3), dtype=np.float32)
    
    for k, pos in positions.items():
        img = cv2.imread(logical_db[k]['path'])
        if img is None: continue
        img_f = img.astype(np.float32)
        
        x_s = int(pos[0] + off_x)
        y_s = int(pos[1] + off_y)
        
        # 边界保护
        if x_s < 0 or y_s < 0 or x_s+w_img > cv_w or y_s+h_img > cv_h: continue
        
        canvas[y_s:y_s+h_img, x_s:x_s+w_img, :3] += img_f
        canvas[y_s:y_s+h_img, x_s:x_s+w_img, 3] += 1.0
        canvas_sq[y_s:y_s+h_img, x_s:x_s+w_img] += img_f ** 2
        
    # 归一化 (求平均)
    weights = canvas[:,:,3]
    mask = weights > 0
    weights_3 = np.stack([weights[mask]] * 3, axis=1)
    
    res_f = np.zeros((cv_h, cv_w, 3), dtype=np.float32)
    res_f[mask] = canvas[mask, :3] / weights_3
    
    # 计算方差图 Var = E[X^2] - (E[X])^2
    var_map = np.zeros((cv_h, cv_w, 3), dtype=np.float32)
    overlap_mask = weights > 1
    if np.any(overlap_mask):
        w_overlap_3 = np.stack([weights[overlap_mask]] * 3, axis=1)
        e_x2 = canvas_sq[overlap_mask] / w_overlap_3  # E[X^2]
        e_x_sq = (canvas[overlap_mask, :3] / w_overlap_3) ** 2 # (E[X])^2
        var_map[overlap_mask] = np.maximum(0, e_x2 - e_x_sq)
    
    res = np.clip(res_f, 0, 255).astype(np.uint8)
    if output_name:
        cv2.imwrite(output_name, res)
        print(f"拼接图已保存: {output_name}")
        
    return res, var_map, weights

def main():
    if not os.path.exists(EXCEL_PATH):
        print("错误: 找不到 Excel 文件")
        return

    # 1. 加载数据
    logical_db, mech_positions = load_and_map_coordinates(IMG_FOLDER, EXCEL_PATH)
    if not logical_db:
        print("未匹配到任何图片。")
        return

    # 2. 坐标转换
    print(f"转换机械坐标 (Scale={PIXELS_PER_MM})...")
    pixel_positions = convert_mech_to_pixel(
        mech_positions, 
        px_per_mm=PIXELS_PER_MM, 
        flip_x=FLIP_X, 
        flip_y=FLIP_Y
    )

    # 3. 渲染
    output_filename = "result_mechanical_excel.jpg"
    print("开始渲染...")
    stitched_img, var_map, weights = render_mechanical(logical_db, pixel_positions, output_filename)

    # 4. 评估指标
    print("正在计算评估指标 (这可能需要几秒钟)...")
    if stitched_img is not None:
        evaluator = StitchingEvaluator(logical_db, pixel_positions, stitched_img, var_map, weights)
        r = evaluator.calculate_metrics() # 获取结果字典

        # 5. 打印结果 (完全符合你的格式要求)
        print("\n" + "="*85)
        header = f"{'Experiment':<25} | {'RMSE':<8} | {'SSIM':<8} | {'Var':<8} | {'ColorDiff':<10} | {'SeamDiff'}"
        print(header)
        print("-" * 85)
        
        print(f"{r['Name']:<25} | "
              f"{r['RMSE']:<8.2f} | "
              f"{r['SSIM']:<8.4f} | "
              f"{r['Var']:<8.2f} | "
              f"{r['ColorDiff']:<10.2f} | "
              f"{r['SeamDiff']:.2f}")
        print("="*85)

if __name__ == "__main__":
    main()