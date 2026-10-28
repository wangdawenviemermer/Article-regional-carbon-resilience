```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import warnings

# 忽略特定警告
warnings.filterwarnings("ignore", category=DeprecationWarning)
# 显式接受 pandas 的未来行为，消除 Downcasting 警告
pd.set_option('future.no_silent_downcasting', True)

# ==========================================
# 1. 路径设置
# ==========================================
path_integrated_data = r"E:\0.data_COVID_carbon_only_total3(aviation).csv"
path_daily_covid_original = r"E:\0.Covid_Per_Capita_Results.csv"

output_wave_list = r"E:\aa.wave_list.csv"
output_wave_day = r"E:\aa.wave_day.csv"
output_viz_png = r"E:\aa.波段可视化图.png"

# ==========================================
# 2. 数据加载与预处理
# ==========================================
df_main = pd.read_csv(path_integrated_data)
df_main['date'] = pd.to_datetime(df_main['date'])
df_main['state'] = df_main['state'].str.strip()

df_cases = pd.read_csv(path_daily_covid_original)
df_cases['date'] = pd.to_datetime(df_cases['date'])
df_cases['state'] = df_cases['state'].str.strip()

# 整合感染强度数据 (假设强度列名为 existing_per_100k)
df = pd.merge(df_main, df_cases[['state', 'date', 'Existing Confirmed Cases']], on=['state', 'date'], how='left')
df['Existing Confirmed Cases'] = df['Existing Confirmed Cases'].fillna(0)

start_bound, end_bound = pd.Timestamp('2020-01-23'), pd.Timestamp('2022-12-24')
df = df[(df['date'] >= start_bound) & (df['date'] <= end_bound)].sort_values(['state', 'date'])


# ==========================================
# 3. 核心算法：波段识别、强制切断与逻辑优化
# ==========================================

def identify_pulses(group):
    """ 第一步：基础脉冲识别 """
    intensity = group['existing_per_100k'].fillna(0).values
    prev_7d = group['existing_per_100k'].shift(7).fillna(0).values
    weekly_growth = (intensity - prev_7d) / (prev_7d + 1e-9)
    wave_id = np.zeros(len(group), dtype=int)
    current_wave_count, in_wave, peak_val, low_count = 0, False, 0, 0

    for i in range(len(group)):
        if not in_wave:
            # 识别起始点
            if (i >= 2 and all(intensity[i - 2:i + 1] >= 0.5)) or (weekly_growth[i] >= 1.0 and intensity[i] >= 0.1):
                in_wave, current_wave_count, peak_val, low_count = True, current_wave_count + 1, intensity[i], 0
                wave_id[i] = current_wave_count
        else:
            # 追踪波段
            peak_val = max(peak_val, intensity[i])
            wave_id[i] = current_wave_count
            if intensity[i] < 0.1 or intensity[i] < 0.2 * peak_val:
                low_count += 1
            else:
                low_count = 0
            if low_count >= 14: in_wave = False
    group['wave_id'] = wave_id
    return group


# 初步波段识别
df = df.groupby('state', group_keys=False).apply(identify_pulses)


def refine_wave_logic(group):
    state_name = group.name
    OM_START = pd.Timestamp('2022-03-01')
    POLICY_DATE = pd.Timestamp('2022-11-11')
    MIN_PRE_DAYS = 14  # 缓冲区阈值：如果11-11之前的存续时间少于14天，则不切断，直接划归新阶段

    # A. 提取原始波段 (保持不变)
    raw_periods = []
    w1_end = max(pd.Timestamp('2020-04-08'),
                 group[group['wave_id'] == 1]['date'].max() if 1 in group['wave_id'].values else pd.Timestamp(
                     '2020-04-08'))
    raw_periods.append({'start': pd.Timestamp('2020-01-23'), 'end': w1_end})

    other_waves = group[(group['wave_id'] > 0) & (group['date'] > w1_end)]
    for wid in sorted(other_waves['wave_id'].unique()):
        w_data = other_waves[other_waves['wave_id'] == wid]
        raw_periods.append({'start': w_data['date'].min(), 'end': w_data['date'].max()})

    # B. 强制执行切断（加入缓冲区优化）
    cut_periods = []
    for p in raw_periods:
        # ---- 11-11 政策切断优化 ----
        if p['start'] < POLICY_DATE and p['end'] >= POLICY_DATE:
            # 计算该波段在 11-11 之前维持了多久
            pre_duration = (POLICY_DATE - p['start']).days

            if pre_duration < MIN_PRE_DAYS:
                # 【优化点】：如果太短，这一整段都算成 11-11 之后的
                # 不进行拆分，直接标记为 is_split=True (因为它是横跨的)，但起止点不切断
                cut_periods.append({'start': p['start'], 'end': p['end'], 'is_split': True})
            else:
                # 如果足够长（>14天），说明是一个独立波段，进行“手术刀”切断
                cut_periods.append({'start': p['start'], 'end': POLICY_DATE - pd.Timedelta(days=1), 'is_split': True})
                cut_periods.append({'start': POLICY_DATE, 'end': p['end'], 'is_split': True})

        # ---- 上海 3.1 切断逻辑 (保持严格) ----
        elif state_name == 'Shanghai' and p['start'] < OM_START and p['end'] >= OM_START:
            cut_periods.append({'start': p['start'], 'end': OM_START - pd.Timedelta(days=1), 'is_split': True})
            cut_periods.append({'start': OM_START, 'end': p['end'], 'is_split': True})

        else:
            cut_periods.append({**p, 'is_split': False})

    # C. 合并屏障逻辑 (14天)
    final_periods = []
    if cut_periods:
        cut_periods.sort(key=lambda x: x['start'])
        curr = cut_periods[0]
        for i in range(1, len(cut_periods)):
            nxt = cut_periods[i]
            # 严禁跨越政策红线合并
            barrier = (curr['end'] < OM_START <= nxt['start'] and state_name == 'Shanghai') or \
                      (curr['end'] < POLICY_DATE <= nxt['start'])

            if (nxt['start'] - curr['end']).days < 14 and not barrier:
                curr['end'] = max(curr['end'], nxt['end'])
                curr['is_split'] = curr['is_split'] or nxt['is_split']
            else:
                final_periods.append(curr)
                curr = nxt
        final_periods.append(curr)

    # D. 更新 ID 和标记列 (修复 FutureWarning)
    group['wave_id'] = 0
    split_info = {}
    for i, period in enumerate(final_periods, 1):
        mask = (group['date'] >= period['start']) & (group['date'] <= period['end'])
        group.loc[mask, 'wave_id'] = i
        split_info[i] = period['is_split']

    group['is_split_wave'] = group['wave_id'].map(lambda x: split_info.get(x, False))
    return group


df = df.groupby('state', group_keys=False).apply(refine_wave_logic)

# ==========================================
# 4. 统计与计算恢复率 (Recovery Rate)
# ==========================================
wave_summaries = []
for (state, wid), g in df.groupby(['state', 'wave_id']):
    if wid == 0: continue

    is_split = g['is_split_wave'].iloc[0]

    # 找到该波段的峰值强度和结束时强度
    peak_val = g['existing_per_100k'].max()
    end_val = g['existing_per_100k'].iloc[-1]

    # 恢复率计算逻辑
    if is_split:
        recovery_rate = np.nan
    else:
        # 如果峰值为0（极少数异常），恢复率设为1
        recovery_rate = (peak_val - end_val) / peak_val if peak_val > 0 else 1.0
        # 限制范围在 0-1 之间
        recovery_rate = max(0, min(1.0, recovery_rate))

    wave_summaries.append({
        'state': state,
        'wave_id': wid,
        'start_date': g['date'].min().date(),
        'end_date': g['date'].max().date(),
        'duration_days': len(g),
        'total_intensity': g['existing_per_100k'].sum(),
        'peak_intensity': peak_val,
        'is_split_by_policy': is_split,
        'recovery_rate': recovery_rate
    })

df_wave_list = pd.DataFrame(wave_summaries)
df_wave_list.to_csv(output_wave_list, index=False, encoding='utf-8-sig')
df[['state', 'date', 'wave_id', 'is_split_wave']].to_csv(output_wave_day, index=False, encoding='utf-8-sig')


# ==========================================
def plot_styled_waves(df_plot, summary_plot, save_path):
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['mathtext.fontset'] = 'stix'
    states = sorted(df_plot['state'].unique())
    n_states = len(states)
    fig, axes = plt.subplots(n_states, 1, figsize=(10, 0.5 * n_states), sharex=True)

    if n_states == 1:
        axes = [axes]

    norm = mcolors.LogNorm(vmin=0.4, vmax=3000)
    cmap = plt.get_cmap('Greens')

    event_dates = [
        pd.Timestamp('2020-04-08'),
        pd.Timestamp('2022-02-28'),
        pd.Timestamp('2022-11-11'),
    ]
    event_labels = ['2020/4/8', '2022/2/28', '2022/11/11']

    for i, state in enumerate(states):
        ax = axes[i]
        s_df = df_plot[df_plot['state'] == state]
        s_summ = summary_plot[summary_plot['state'] == state]

        ax.plot(s_df['date'], s_df['Existing Confirmed Cases'], color='black', linewidth=0.3)

        for _, row in s_summ.iterrows():
            w_start = pd.Timestamp(row['start_date'])
            w_end = pd.Timestamp(row['end_date'])
            w_data = s_df[(s_df['date'] >= w_start) & (s_df['date'] <= w_end)]

            fill_color = '#BBBBBB' if w_end >= event_dates[-1] else cmap(norm(row['total_intensity']))
            ax.fill_between(
                w_data['date'],
                0,
                w_data['Existing Confirmed Cases'],
                color=fill_color,
                alpha=0.8,
                edgecolor='none'
            )

        for j, dt in enumerate(event_dates):
            line_color = '#999999' if j < 2 else 'red'
            ax.axvline(dt, color=line_color, linestyle='--', linewidth=0.5)

        ax.set_yticks([])
        ax.set_ylabel(state, rotation=0, labelpad=10, fontsize=15, va='center', ha='right')

        # 边框更细
        for spine in ax.spines.values():
            spine.set_linewidth(0.4)
            spine.set_color('black')

        # x 轴刻度向内
        ax.tick_params(axis='x', direction='in', length=2, width=0.4)
        ax.tick_params(axis='y', length=0)

        # 只有最后一个子图显示日期标签
        if i == n_states - 1:
            ax.set_xticks(event_dates)
            ax.set_xticklabels(event_labels, fontsize=15)
        else:
            ax.tick_params(axis='x', labelbottom=False)

    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()
plot_styled_waves(df, df_wave_list, output_viz_png)
print(f"✅ 处理完成！CSV已更新，恢复率(Recovery Rate)已计算。\n结果见: {output_wave_list}")
```