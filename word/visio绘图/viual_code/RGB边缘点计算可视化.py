import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Arrow

def create_synthetic_image():
    """ 生成一个模拟工业图像：绿色背景 + 深色圆形零件 """
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    # 背景设为鲜艳绿色 (B, G, R)
    img[:] = [30, 220, 30] 
    # 绘制一个深灰色的圆形零件边缘
    cv2.circle(img, (150, 150), 80, (50, 50, 50), -1)
    # 添加一点高斯噪声模拟真实相机
    noise = np.random.normal(0, 3, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img

def visualize_edge_detection():
    # 1. 准备数据
    img = create_synthetic_image()
    bg_color = np.array([30, 220, 30]) # 预采样的背景色
    
    # 定义一条扫描线 (从背景进入零件)
    p1 = np.array([50, 150])  # 起点 (背景)
    p2 = np.array([120, 150]) # 终点 (零件内部)
    
    # 2. 执行采样 (基于线段长度均匀采样)
    length = np.linalg.norm(p2 - p1)
    num_samples = int(length)
    t = np.linspace(0, 1, num_samples)
    
    # 计算采样坐标点 (线性等距)
    sample_coords = np.outer(1-t, p1) + np.outer(t, p2)
    
    # 提取颜色剖面 (双线性插值简化版：直接提取像素)
    profile_colors = []
    for pt in sample_coords:
        x, y = int(pt[0]), int(pt[1])
        profile_colors.append(img[y, x])
    profile_colors = np.array(profile_colors)
    
    # 3. 计算颜色空间距离曲线 D(i)
    # D(i) = ||C_i - C_bg||
    diff = profile_colors.astype(np.float32) - bg_color.astype(np.float32)
    dist_curve = np.linalg.norm(diff, axis=1)
    
    # 4. 计算距离梯度曲线
    grads = np.abs(np.gradient(dist_curve))
    
    # 5. 寻找最大梯度位置 (亚像素简化：找峰值)
    max_grad_idx = np.argmax(grads)
    edge_pt = sample_coords[max_grad_idx]
    
    # ==================== 可视化绘制 ====================
    fig = plt.figure(figsize=(12, 8))
    plt.rcParams['font.sans-serif'] = ['The new Roman'] # 显示中文
    plt.rcParams['axes.unicode_minus'] = False

    # 图1：图像采样区域可视化
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 绘制扫描剖面线
    ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], color='yellow', linewidth=1, label='Scanning profile line')
    # 绘制采样点 (每隔5个点画一个，避免太密)
    ax1.scatter(sample_coords[::3, 0], sample_coords[::3, 1], c='cyan', s=10, label='Equidistant sampling points')
    # 标记最终检测到的边缘点
    ax1.scatter(edge_pt[0], edge_pt[1], color='red', marker='x', s=100, linewidth=2, label='Final edge point')
    ax1.set_title("1. Visualization of image section area and sampling points")
    ax1.legend()

    # 图2：剖面颜色特征 (RGB)
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(profile_colors[:, 2], color='red', label='R')
    ax2.plot(profile_colors[:, 1], color='green', label='G')
    ax2.plot(profile_colors[:, 0], color='blue', label='B')
    ax2.set_title("2. Original RGB signals at sampling points")
    ax2.set_xlabel("Sample point index")
    ax2.legend()

    # 图3：距离曲线 D(i)
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(dist_curve, color='purple', linewidth=2)
    ax3.fill_between(range(len(dist_curve)), dist_curve, color='purple', alpha=0.2)
    ax3.set_title("3. Background distance mapping D(i) (Background suppression)")
    ax3.set_ylabel("Distance amplitude")
    ax3.set_xlabel("Sample point index")

    # 图4：梯度曲线与边缘定位
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(grads, color='orange', linewidth=2)
    ax4.axvline(x=max_grad_idx, color='red', linestyle='--', label='Gradient peak (edge point)')
    ax4.set_title("4. Distance gradient curve |∇D(i)|")
    ax4.set_ylabel("Gradient intensity")
    ax4.set_xlabel("Sample point index")
    ax4.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_edge_detection()