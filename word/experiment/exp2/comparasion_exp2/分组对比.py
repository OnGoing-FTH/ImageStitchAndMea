import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm

# ==========================================
# 1. 数据加载与预处理
# ==========================================
file_path = r"D:\paper_3\word\experiment\exp2\comparasion_exp2\result1-2.xlsx"  
df_raw = pd.read_excel(file_path)

# 确保列名为 Category, Measurement, Debunchingfactor
df = df_raw.iloc[:, :3].copy()
df.columns = ['Category', 'Measurement', 'Debunchingfactor']

# 仅保留您需要的 5 种特定分类，并按顺序排列
target_categories = ['Color+Ls', 'Color+Nofit', 'Color+Robust', 'Grayproj+Ls', 'Grayproj+Nofit']
df = df[df['Category'].isin(target_categories)]

# 计算均值和标准误差 (SEM)
stats = df.groupby('Category').agg({
    'Measurement': ['mean', 'std', 'count'],
    'Debunchingfactor': ['mean', 'std', 'count']
})

# 平坦化索引并计算 SEM
stats.columns = ['_'.join(col) for col in stats.columns]
for metric in ['Measurement', 'Debunchingfactor']:
    stats[f'{metric}_sem'] = stats[f'{metric}_std'] / np.sqrt(stats[f'{metric}_count'])

# 按照指定的顺序重排 stats 索引
stats = stats.reindex(target_categories)

# ==========================================
# 2. 字体与样式设置
# ==========================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False 

font_zh = fm.FontProperties(family='SimSun', size=11)
font_zh_bold = fm.FontProperties(family='SimSun', size=12, weight='bold')

# ==========================================
# 3. 绘制双Y轴柱状图
# ==========================================
fig, ax1 = plt.subplots(figsize=(12, 7))
ax2 = ax1.twinx()  # 创建共享X轴的第二个Y轴

x = np.arange(len(target_categories))
width = 0.35  # 条形图宽度

# --- 绘制左轴：Measurement ---
bars1 = ax1.bar(x - width/2, stats['Measurement_mean'], width, 
                yerr=stats['Measurement_sem'], capsize=5,
                label='Measurement (左轴)', color='#7294D4', edgecolor='#333333', alpha=0.8)

# --- 绘制右轴：Debunchingfactor ---
bars2 = ax2.bar(x + width/2, stats['Debunchingfactor_mean'], width, 
                yerr=stats['Debunchingfactor_sem'], capsize=5,
                label='Outlier Rate (右轴)', color='#E88482', edgecolor='#333333', alpha=0.8)

# ==========================================
# 4. 数值标注
# ==========================================
def autolabel(bars, ax, decimal=3, offset=5):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.{decimal}f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, offset), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

autolabel(bars1, ax1, decimal=3)
autolabel(bars2, ax2, decimal=3)

# ==========================================
# 5. 坐标轴与细节优化
# ==========================================
# X轴设置
ax1.set_xticks(x)
ax1.set_xticklabels(target_categories, fontsize=10, fontweight='bold')
ax1.set_xlabel("Algorithm Categories", fontsize=12, labelpad=10)

# 左轴美化 (测量值)
ax1.set_ylabel("Measured Dimension (mm)", fontsize=12, color='#2E4A7D')
m_min, m_max = stats['Measurement_mean'].min(), stats['Measurement_mean'].max()
ax1.set_ylim(m_min - 0.05, m_max + 0.1)  # 局部放大测量值的差异
ax1.tick_params(axis='y', labelcolor='#2E4A7D')

# 右轴美化 (离群率)
ax2.set_ylabel("Outlier Rate (Debunching Factor)", fontsize=12, color='#A33A38')
ax2.set_ylim(0, stats['Debunchingfactor_mean'].max() * 1.5) # 留出标注空间
ax2.tick_params(axis='y', labelcolor='#A33A38')

# 标题
plt.title("不同算法分类的测量精度与离群率双轴对比", fontproperties=font_zh_bold, pad=20)

# 图例合并显示
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', prop=font_zh)

# 网格线 (仅保留左轴)
ax1.grid(axis='y', linestyle='--', alpha=0.5)
ax2.grid(False)

plt.tight_layout()
plt.savefig("dual_axis_comparison.png", dpi=300)
plt.show()