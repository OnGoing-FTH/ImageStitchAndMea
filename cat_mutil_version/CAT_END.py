import os
import re
import cv2
import numpy as np
import statistics

# --- 1. 文件与网格工具 ---
def parse_filename(filename):
    match = re.match(r'(\d+)-(\d+)\.bmp', filename)
    if match: return int(match.group(1)), int(match.group(2))
    return None, None

def create_image_grid_and_info(folder_path):
    image_grid = {}
    row_info = {}
    max_row, max_col_overall = 0, 0
    if not os.path.isdir(folder_path): return None, None, 0, 0

    temp_rows = {}
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith('.bmp'):
            row, col = parse_filename(filename)
            if row is not None:
                image_grid[(row, col)] = os.path.join(folder_path, filename)
                if row not in temp_rows: temp_rows[row] = []
                temp_rows[row].append(col)
                max_row = max(max_row, row)
                max_col_overall = max(max_col_overall, col)

    for r, cols in temp_rows.items():
        row_info[r] = (min(cols), max(cols))
        
    print(f"扫描: {len(image_grid)}张图片, {max_row}行, 最大列号 {max_col_overall}")
    return image_grid, row_info, max_row, max_col_overall

# --- 2. 特征匹配与重叠计算 ---
def find_translation(img1_gray, img2_gray):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1_gray, None)
    kp2, des2 = sift.detectAndCompute(img2_gray, None)

    if des1 is None or des2 is None: return None
    
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    matches = flann.knnMatch(des2, des1, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

    if len(good_matches) >= 10:
        src_pts = np.float32([kp2[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp1[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is None:
            return None # 如果找不到变换矩阵，安全返回
        return H[0, 2], H[1, 2] # dx, dy
    return None

def calculate_standard_overlaps(image_grid, row_info, max_row):
    h_overlaps, v_overlaps = [], []
    img_h, img_w, _ = cv2.imread(next(iter(image_grid.values()))).shape

    print("\n--- 阶段一: 分析重叠区域 ---")
    for r in range(1, max_row + 1):
        if r not in row_info: continue
        for c in range(row_info[r][0], row_info[r][1] + 1):
            pos1 = (r, c)
            if pos1 not in image_grid: continue
            
            # 分析严格水平邻居
            pos2_h = (r, c + 1)
            if pos2_h in image_grid:
                shift = find_translation(cv2.imread(image_grid[pos1], 0), cv2.imread(image_grid[pos2_h], 0))
                if shift:
                    overlap = img_w - shift[0]
                    h_overlaps.append(overlap)
                    print(f"分析水平对: {pos1} -> {pos2_h}, 重叠宽度: {overlap:.2f} 像素")

            # 分析严格垂直邻居
            pos2_v = (r + 1, c)
            if pos2_v in image_grid:
                shift = find_translation(cv2.imread(image_grid[pos1], 0), cv2.imread(image_grid[pos2_v], 0))
                if shift:
                    overlap = img_h - shift[1]
                    v_overlaps.append(overlap)
                    print(f"分析垂直对: {pos1} -> {pos2_v}, 重叠高度: {overlap:.2f} 像素")
    
    if not h_overlaps or not v_overlaps:
        raise ValueError("未能计算出有效的重叠区域。")

    std_overlap_x = statistics.median(h_overlaps)
    std_overlap_y = statistics.median(v_overlaps)
    print(f"\n标准水平重叠: {std_overlap_x:.2f} 像素")
    print(f"标准垂直重叠: {std_overlap_y:.2f} 像素")
    return std_overlap_x, std_overlap_y

# --- 3. 两步拼接流程 ---
def build_mosaic_by_rows(image_grid, max_row, max_col_overall, overlap_x, overlap_y):
    print("\n--- 阶段二: 行内拼接 ---")
    img_h, img_w, _ = cv2.imread(next(iter(image_grid.values()))).shape
    step_x = img_w - overlap_x
    step_y = img_h - overlap_y
    
    stitched_rows = []
    max_canvas_w = int((max_col_overall - 1) * step_x + img_w)

    for r in range(1, max_row + 1):
        row_canvas = np.zeros((img_h, max_canvas_w, 3), dtype=np.uint8)
        print(f"正在拼接第 {r} 行...")
        for c in range(1, max_col_overall + 1):
            pos = (r, c)
            if pos in image_grid:
                img = cv2.imread(image_grid[pos])
                x_offset = int((c - 1) * step_x)
                row_canvas[:, x_offset:x_offset + img_w] = img # 简单覆盖
        stitched_rows.append(row_canvas)
        # 可视化单行结果
        cv2.imshow("Row Stitching", row_canvas)
        cv2.waitKey(100)
    
    cv2.destroyWindow("Row Stitching")

    print("\n--- 阶段三: 行间拼接 ---")
    final_canvas_h = int((max_row - 1) * step_y + img_h)
    final_canvas = np.zeros((final_canvas_h, max_canvas_w, 3), dtype=np.uint8)

    for i, row_img in enumerate(stitched_rows):
        y_offset = int(i * step_y)
        # 使用 Alpha Blending 进行垂直融合
        if i > 0:
            overlap_area = final_canvas[y_offset : y_offset + img_h, :]
            final_canvas[y_offset : y_offset + img_h, :] = cv2.addWeighted(overlap_area, 0.5, row_img, 0.5, 0)
        else:
            final_canvas[y_offset : y_offset + img_h, :] = row_img

    cv2.imwrite("final_mosaic_by_rows.jpg", final_canvas)
    print("\n完成! 结果已保存为 'final_mosaic_by_rows.jpg'")
    cv2.imshow("Final Mosaic", cv2.resize(final_canvas, None, fx=0.2, fy=0.2))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    folder_path = 'data1'
    image_grid, row_info, max_row, max_col = create_image_grid_and_info(folder_path)
    if not image_grid: return
    
    try:
        overlap_x, overlap_y = calculate_standard_overlaps(image_grid, row_info, max_row)
        build_mosaic_by_rows(image_grid, max_row, max_col, overlap_x, overlap_y)
    except ValueError as e:
        print(f"错误: {e}")

if __name__ == '__main__':
    main()