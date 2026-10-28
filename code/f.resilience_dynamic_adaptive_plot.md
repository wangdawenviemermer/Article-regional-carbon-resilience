```
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# 1. 加载数据
df = pd.read_csv(r'E:\study\g.DATA\b.data_COVID_carbon\2.phase_method\c.Wave_Resilience_Metrics.csv')

# =========================================================
# 2. 基于“分位数截断”的非线性映射范围（方案三核心）
# =========================================================
vals_all = df['cumulative_loss_area'].to_numpy()

pos_vals = vals_all[vals_all >= 0]
neg_vals = vals_all[vals_all < 0]

# 默认兜底范围（你可按需要改）
p5_pos, p95_pos = 0.0, 31.5
p5_neg, p95_neg = -37.0, 0.0

if len(pos_vals) > 0:
    p5_pos = np.percentile(pos_vals, 5)
    p95_pos = np.percentile(pos_vals, 95)

if len(neg_vals) > 0:
    p5_neg = np.percentile(neg_vals, 5)   # 更负（接近最小值一侧）
    p95_neg = np.percentile(neg_vals, 95) # 更接近0一侧

# 正值：截断到 [0, p95_pos]
norm_pos = plt.Normalize(0, p95_pos)

# 负值：截断到 [p5_neg, 0]
norm_neg = plt.Normalize(p5_neg, 0)

# 颜色映射
cmap_pos = mcolors.LinearSegmentedColormap.from_list(
    "pos_grd", ["#FDF6F4", "#A66B63"]
)
cmap_neg = mcolors.LinearSegmentedColormap.from_list(
    "neg_grd", ["#597A68", "#F8FBF9"]
)

def get_smooth_color(val):
    """按分位数截断后再做线性映射"""
    if val >= 0:
        val_clipped = np.clip(val, 0, p95_pos)
        return cmap_pos(norm_pos(val_clipped))
    else:
        val_clipped = np.clip(val, p5_neg, 0)
        return cmap_neg(norm_neg(val_clipped))

# =========================
# 3. 绘图参数设置
# =========================
states = df['state'].unique()
phases = [1, 2, 3, 4]

fig, ax = plt.subplots(figsize=(15, len(states) * 0.7))
fig.subplots_adjust(bottom=0.12)  # 给底部色条留空间

cell_height = 0.5
cell_width = 1.5
phase_gap = 1.6

# 4. 开始绘制轨迹
for i, state in enumerate(reversed(states)):
    ax.text(-1.8, i, state, va='center', ha='right', fontsize=12, fontweight='bold')

    for p_idx, p in enumerate(phases):
        sub_df = df[(df['state'] == state) & (df['phase'] == p)]
        vals = sub_df['cumulative_loss_area'].tolist()  # ✅ 用子集，而不是整表

        x_start = p_idx * phase_gap

        # 背景底框
        bg_rect = plt.Rectangle(
            (x_start - cell_width / 2, i - cell_height / 2),
            cell_width, cell_height,
            fill=False, edgecolor='#eeeeee', lw=1, linestyle='--'
        )
        ax.add_patch(bg_rect)

        if len(vals) > 0:
            num_sub = len(vals)
            sub_w = cell_width / num_sub

            for j, v in enumerate(vals):
                color = get_smooth_color(v)
                rect = plt.Rectangle(
                    (x_start - cell_width / 2 + j * sub_w, i - cell_height / 2),
                    sub_w, cell_height,
                    facecolor=color, edgecolor='white', lw=0
                )
                ax.add_patch(rect)

            outer_rect = plt.Rectangle(
                (x_start - cell_width / 2, i - cell_height / 2),
                cell_width, cell_height,
                fill=False, edgecolor='#333333', lw=0.5
            )
            ax.add_patch(outer_rect)

# 5. 图表修饰
for p_idx, p_name in enumerate(['Phase I', 'Phase II', 'Phase III', 'Phase Ⅳ']):
    ax.text(p_idx * phase_gap, len(states), p_name,
            ha='center', va='bottom', fontsize=14, fontweight='bold')

ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)

ax.set_xlim(-2.5, (len(phases) - 1) * phase_gap + 1.5)
ax.set_ylim(-1, len(states) + 0.5)

# 6. 添加色彩渐变图例（与矩形块统一）
def add_colorbar():
    # 正值渐变
    ax_pos = fig.add_axes([0.28, 0.04, 0.18, 0.025])
    cb_pos = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm_pos, cmap=cmap_pos),
        cax=ax_pos,
        orientation='horizontal'
    )
    cb_pos.set_label(f'Positive (0 to p95={p95_pos:.2f})', fontsize=9)

    # 负值渐变
    ax_neg = fig.add_axes([0.54, 0.04, 0.18, 0.025])
    cb_neg = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm_neg, cmap=cmap_neg),
        cax=ax_neg,
        orientation='horizontal'
    )
    cb_neg.set_label(f'Negative (p5={p5_neg:.2f} to 0)', fontsize=9)

add_colorbar()

# 7. 保存并显示
plt.savefig(
    r'E:\study\g.DATA\b.data_COVID_carbon\2.phase_method\learning_ability_matrix_44.png',
    dpi=300,
    bbox_inches='tight'
)
print("图表已保存为: learning_ability_matrix_44.png")
plt.show()
```