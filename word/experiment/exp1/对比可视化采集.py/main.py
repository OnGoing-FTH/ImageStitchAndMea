import cv2
import os

# ================== 参数区 ==================
input_dir = r"D:\paper_3\word\experiment\exp1\result\diandong6"
output_dir = r"D:\paper_3\word\experiment\exp1\result\diandong"

# ROI 是【画图 800% 下的显示坐标】
roi_list_display = [
    (6085, 695, 6163, 773),
    (7671, 6546, 7749, 6624),
    (735, 2330, 813, 2408),
    (1651, 7437, 1729, 7515),
]

display_zoom = 8          # 800%
save_display_zoom = True  # 是否额外保存 800% 显示效果

# ===========================================
os.makedirs(output_dir, exist_ok=True)
img_exts = ('.jpg', '.png', '.bmp', '.tif')

def display_roi_to_image_roi(roi_list, zoom):
    """显示坐标 ROI → 原图像素 ROI"""
    img_rois = []
    for (x1, y1, x2, y2) in roi_list:
        ix1 = int(round(x1 / zoom))
        iy1 = int(round(y1 / zoom))
        ix2 = int(round(x2 / zoom))
        iy2 = int(round(y2 / zoom))
        img_rois.append((ix1, iy1, ix2, iy2))
    return img_rois

for fname in os.listdir(input_dir):
    if not fname.lower().endswith(img_exts):
        continue

    img_path = os.path.join(input_dir, fname)
    img = cv2.imread(img_path)
    if img is None:
        print(f"读取失败: {fname}")
        continue

    h_img, w_img = img.shape[:2]
    base_name = os.path.splitext(fname)[0]

    # ✅ 关键一步：显示坐标 → 原图坐标
    roi_list_image = display_roi_to_image_roi(
        roi_list_display, display_zoom
    )

    for idx, (x1, y1, x2, y2) in enumerate(roi_list_image, start=1):

        # ---------- 边界保护 ----------
        x1 = max(0, min(x1, w_img - 1))
        y1 = max(0, min(y1, h_img - 1))
        x2 = max(x1 + 1, min(x2, w_img))
        y2 = max(y1 + 1, min(y2, h_img))

        roi = img[y1:y2, x1:x2]

        # ===== 保存 1：真实像素 ROI（不模糊，最重要）=====
        out_real = os.path.join(
            output_dir, f"{base_name}_roi{idx}_real.png"
        )
        cv2.imwrite(out_real, roi)

        # ===== 保存 2：800% 显示效果（可选，仅用于看）=====
        if save_display_zoom:
            roi_disp = cv2.resize(
                roi,
                None,
                fx=display_zoom,
                fy=display_zoom,
                interpolation=cv2.INTER_NEAREST  # 画图 / VS Code 行为
            )
            out_disp = os.path.join(
                output_dir, f"{base_name}_roi{idx}_display800.png"
            )
            cv2.imwrite(out_disp, roi_disp)

    print(f"{fname} 处理完成")
