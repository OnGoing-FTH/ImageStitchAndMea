import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import rcParams, font_manager
import os

# ==================== 字体设置（英文 TNR + 中文宋体） ====================

# 英文字体：Times New Roman
rcParams['font.family'] = 'Times New Roman'
rcParams['axes.unicode_minus'] = False  # 负号正常显示

TIMES_PATH = r"C:\Windows\Fonts\times.ttf"   # Times New Roman Regular
font_times = font_manager.FontProperties(fname=TIMES_PATH)

def process_user_image(image_path):
    # ==================== 1. 加载图像 ====================
    if not os.path.exists(image_path):
        print(f"错误：找不到文件 {image_path}")
        return

    img = cv2.imread(image_path)
    if img is None:
        print("错误：图像加载失败，请检查格式。")
        return

    # ==================== 2. 背景参考色 ====================
    bg_roi = img[0:20, 0:20]
    bg_color = np.mean(bg_roi, axis=(0, 1))
    print(f"检测到背景参考色 (BGR): {bg_color}")

    # 扫描线端点
    p1 = np.array([487, 88])
    p2 = np.array([959, 793])

    # ==================== 3. 剖面采样 ====================
    length = np.linalg.norm(p2 - p1)
    num_samples = int(length)
    if num_samples < 2:
        return

    t = np.linspace(0, 1, num_samples)
    sample_coords = np.outer(1 - t, p1) + np.outer(t, p2)

    map_x = sample_coords[:, 0].astype(np.float32).reshape(1, -1)
    map_y = sample_coords[:, 1].astype(np.float32).reshape(1, -1)
    profile_colors = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)[0]

    # ==================== 4. 背景距离映射 ====================
    diff = profile_colors.astype(np.float32) - bg_color.astype(np.float32)
    dist_curve = np.linalg.norm(diff, axis=1)

    # 平滑 + 梯度
    dist_curve_smooth = cv2.medianBlur(
        dist_curve.reshape(-1, 1, 1).astype(np.float32), 3
    ).flatten()
    grads = np.abs(np.gradient(dist_curve_smooth))
    max_grad_idx = np.argmax(grads)
    edge_pt = sample_coords[max_grad_idx]

    # ==================== 5. 可视化 ====================
    fig = plt.figure(figsize=(14, 9))

    # ---------- 子图 1 ----------
    ax1 = fig.add_subplot(2, 2, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax1.imshow(img_rgb)
    ax1.plot([p1[0], p2[0]], [p1[1], p2[1]],
             color='yellow', linewidth=1.5, label='Scanning line')
    ax1.scatter(sample_coords[::5, 0], sample_coords[::5, 1],
                c='blue', s=5, label='Sample Points')
    ax1.scatter(edge_pt[0], edge_pt[1],
                color='red', marker='x', s=150, linewidth=2,
                label='Edge Point',zorder=10)
    ax1.set_title("1. Original image and sampling distribution", fontsize=14)
    ax1.legend(prop=font_times)

    # ---------- 子图 2 ----------
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(profile_colors[:, 2], color='red', label='Red Channel', alpha=0.8)
    ax2.plot(profile_colors[:, 1], color='green', label='Green Channel', alpha=0.8)
    ax2.plot(profile_colors[:, 0], color='blue', label='Blue Channel', alpha=0.8)
    ax2.set_title("2. Scanning profile raw color signals",
                  fontsize=14)
    ax2.set_xlabel("Sample point index")
    ax2.set_ylabel("Pixel value")
    ax2.legend()

    # ---------- 子图 3 ----------
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(dist_curve, color='purple', linewidth=1.5,
             label='Background deviation distance $D(i)$')
    ax3.fill_between(range(len(dist_curve)), dist_curve,
                     color='purple', alpha=0.1)
    ax3.set_title("3. Background color space distance mapping",
                  fontsize=14)
    ax3.set_xlabel("Sample point index")
    ax3.set_ylabel("Distance magnitude")
    ax3.legend(prop=font_times)

    # ---------- 子图 4 ----------
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(grads, color='darkorange', linewidth=2)
    ax4.axvline(x=max_grad_idx, color='red', linestyle='--',
                label='Edge position peak')
    ax4.set_title("4. Distance gradient distribution and sub-pixel positioning (gradient analysis)",
                  fontsize=14)
    ax4.set_xlabel("Sample point index")
    ax4.set_ylabel("Gradient intensity")
    ax4.legend(prop=font_times)
    plt.savefig('measurement_analysis.png', dpi=600)
    plt.tight_layout()
    plt.show()


# ==================== 执行入口 ====================
if __name__ == "__main__":
    YOUR_IMAGE_PATH = r"D:\paper_3\word\viual_code\diandong6_9.jpg"
    process_user_image(YOUR_IMAGE_PATH)
