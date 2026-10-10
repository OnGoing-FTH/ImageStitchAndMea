import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm

# ==========================================
# 1. 数据加载与预处理
# ==========================================
file_path = r"D:\paper_3\word\experiment\exp2\comparasion_exp2\result.xlsx"  
df_raw = pd.read_excel(file_path)

# 强制取前 3 列并命名
df = df_raw.iloc[:, :3].copy() 
df.columns = ['Category', 'Value', 'Score']

def process_data(row):
    cat = str(row['Category'])
    base_methods = ['Color', 'Canny', 'Sobel', 'Grayproj', 'ZerNike']
    method = "Others"
    for m in base_methods:
        if m.lower() in cat.lower():
            method = m
            break
    strategy = 'Fit' if ('Ls' in cat or 'Robust' in cat) else 'Nofit'
    return pd.Series([method, strategy])

df[['Method', 'Strategy']] = df.apply(process_data, axis=1)

df = df[df['Method'] != 'Others'].reset_index(drop=True)
method_order = ['Canny', 'Sobel', 'Grayproj', 'ZerNike', 'Color']
strategy_order = ['Fit', 'Nofit']

# --- 【关键步骤】：预先计算每个组的 Mean 和 SEM ---
stats = df.groupby(['Method', 'Strategy'])['Value'].agg(['mean', 'std', 'count']).reset_index()
stats['sem'] = stats['std'] / np.sqrt(stats['count'])

# 设置索引方便后续快速查找-
stats.set_index(['Method', 'Strategy'], inplace=True)

# ==========================================
# 3. 绘图设置 (Times New Roman + 手动宋体)
# ==========================================
sns.set_theme(style="whitegrid") 

# 1. 设置全局字体为 Times New Roman (处理所有英文和数字)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False 

# 2. 定义宋体属性 (专门用于中文部分)
# 这里的 'SimSun' 必须是你系统中查到的那个名字
font_zh = fm.FontProperties(family='SimSun', size=12)
font_zh_bold = fm.FontProperties(family='SimSun', size=14, weight='bold')

# 专业配色方案
my_palette = {"Fit": "#A2C4E4", "Nofit": "#F2A2A2"} 

# 绘制柱状图
ax = sns.barplot(
    data=df,
    x="Method",
    y="Value",
    hue="Strategy",
    order=method_order,
    hue_order=strategy_order,
    palette=my_palette,
    capsize=0.15,
    errorbar=('se', 1),
    edgecolor="#555555",
    linewidth=1,
    err_kws={'linewidth': 2, 'color': '#333333'} 
)

# ==========================================
# 4. 动态标注：距离误差棒顶部一定距离
# ==========================================
patch_idx = 0
for s in strategy_order:
    for m in method_order:
        p = ax.patches[patch_idx]
        try:
            current_mean = stats.loc[(m, s), 'mean']
            current_sem = stats.loc[(m, s), 'sem']
            
            # 标注数字（数字用 Times New Roman，自动继承全局）
            ax.annotate(f'{current_mean:.3f}', 
                        xy=(p.get_x() + p.get_width() / 2., current_mean + current_sem),
                        ha='center', va='bottom', 
                        xytext=(0, 8), 
                        textcoords='offset points',
                        fontsize=10, 
                        fontweight='bold',
                        color='#222222')
        except KeyError:
            pass
        patch_idx += 1

# ==========================================
# 5. 解决“超出边框”并手动应用中文字体
# ==========================================
stats['top_edge'] = stats['mean'] + stats['sem']
y_max_with_error = stats['top_edge'].max()
y_min_val = stats['mean'].min()
y_range = y_max_with_error - y_min_val

plt.ylim(y_min_val - y_range*0.2, y_max_with_error + y_range*0.5) 

# --- 手动为包含中文的标签指定 fontproperties ---
plt.title("各算法测量精度对比 (误差棒: SEM)", fontproperties=font_zh_bold, pad=30)
plt.xlabel("Algorithm Method", fontsize=12, labelpad=10) # 纯英文自动用Times New Roman
plt.ylabel("Measured Dimension", fontsize=12, labelpad=10)

# 图例中文处理
# title_fontproperties 控制图例标题，prop 控制图例内容
plt.legend(title="处理策略", title_fontproperties=font_zh, loc='upper left')

plt.tight_layout()
plt.savefig("result_final_fixed.png", dpi=300)
plt.show()