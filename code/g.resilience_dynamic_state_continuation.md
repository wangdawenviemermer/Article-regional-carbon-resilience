```
import pandas as pd
import numpy as np
from datetime import timedelta
from scipy import stats
import pymannkendall as mk  # 如果未安装，请执行 pip install pymannkendall

# ==========================================
# 1. 数据加载与格式化
# ==========================================
# 请确保文件名与路径正确
try:
    df_wave = pd.read_csv(r'E:\study\g.DATA\b.data_COVID_carbon\phase4_method\aa.wave_list.csv')
    df_daily = pd.read_csv(r'E:\study\g.DATA\b.data_COVID_carbon\phase4_method\bb.wave_gap_rate_and_fCO2_M.csv')
except FileNotFoundError:
    print("错误：未找到 CSV 文件，请检查路径。")
    exit()

# 统一转换日期格式
df_wave['start_date'] = pd.to_datetime(df_wave['start_date'])
df_wave['end_date'] = pd.to_datetime(df_wave['end_date'])
df_daily['date'] = pd.to_datetime(df_daily['date'])

# 定义常量
SPLIT_DT = pd.to_datetime('2022-11-11')
FORCED_START_DT = pd.to_datetime('2022-11-12')
END_LIMIT = pd.to_datetime('2022-12-31')

results = []

# ==========================================
# 2. 逐省份（State）逻辑处理
# ==========================================
for state_name, group in df_wave.groupby('state'):
    # A. 确定该 state 的逻辑起点
    # 取 wave_id 最大的波段作为最后一个波段
    last_wave = group.sort_values('wave_id').iloc[-1]
    l_start = last_wave['start_date']
    l_end = last_wave['end_date']
    l_id = last_wave['wave_id']

    selected_start_date = None
    logic_desc = ""

    if l_start <= SPLIT_DT and l_end >= SPLIT_DT:
        # 情况 1: 横跨 11/11
        selected_start_date = l_start
        logic_desc = "横跨11/11:取波段起点"
    elif l_start > SPLIT_DT:
        # 情况 2: 最后一个波段在 11/11 之后开始
        selected_start_date = FORCED_START_DT
        logic_desc = "波段在11/11后:强制11/12起"
    else:
        # 情况 3: 最后一个波段在 11/11 前已结束
        selected_start_date = l_end + timedelta(days=1)
        logic_desc = "波段早于11/11结束:取波段后一天"

    # B. 从图2筛选数据 (排除 11/11)
    mask = (df_daily['state'] == state_name) & \
           (df_daily['date'] >= selected_start_date) & \
           (df_daily['date'] <= END_LIMIT) & \
           (df_daily['date'] != SPLIT_DT)

    period_df = df_daily[mask].copy().sort_values('date')

    # C. 统计与趋势计算
    # 至少需要一定天数才能做趋势分析 (建议至少 10 天)
    if not period_df.empty and len(period_df) >= 10:
        # 计算差值序列
        period_df['diff'] = period_df['Total_fCO2_M'] - period_df['Total_aCO2']

        # --- 3日滑动平均平滑 ---
        # 针对已校准数据，采用小窗口平滑，滤除高频随机误差，保留真实趋势
        period_df['diff_smooth'] = period_df['diff'].rolling(window=3, center=True, min_periods=1).mean()

        y = period_df['diff_smooth'].values
        x = np.arange(len(y))

        # --- 泰尔森斜率计算 (Theil-Sen Slope) ---
        slope, intercept, _, _ = stats.theilslopes(y, x)

        # --- 预白化曼-肯德尔检验 (TFPW-MK) ---
        # 预白化处理特别适用于校准后的逐日监测数据，可剔除序列自相关干扰
        try:
            mk_res = mk.trend_free_pre_whitening_modification_test(y)
        except Exception:
            # 如果样本极度特殊导致预白化失败，回退到原始MK检验
            mk_res = mk.original_test(y)

        # D. 聚合统计指标
        results.append({
            'state': state_name,
            '片段起点': selected_start_date.strftime('%Y-%m-%d'),
            '片段终点': '2022-12-31',
            '有效天数': len(y),
            '总差值和': round(period_df['diff'].sum(), 4),
            '日均差值': round(period_df['diff'].mean(), 4),
            '泰尔森斜率': round(slope, 6),
            'MK趋势判定': mk_res.trend,  # increasing / decreasing / no trend
            'P值(显著性)': round(mk_res.p, 4),
            '显著性结论': '显著' if mk_res.p < 0.05 else '不显著',
            'S统计量': mk_res.s,  # S>0 代表上升倾向, S<0 代表下降倾向
            '逻辑来源': logic_desc,
            '参考末波段ID': l_id
        })
    else:
        # 数据量不足处理
        msg = "数据少于10天" if not period_df.empty else "无匹配数据"
        results.append({
            'state': state_name,
            '片段起点': selected_start_date.strftime('%Y-%m-%d'),
            '逻辑来源': logic_desc,
            '备注': msg
        })

# ==========================================
# 3. 结果保存
# ==========================================
final_df = pd.DataFrame(results)
final_df.to_csv(r'E:\study\g.DATA\b.data_COVID_carbon\phase4_method\refined_trend_analysis.csv', index=False, encoding='utf-8-sig')

print("--------------------------------------------------")
print("分析完成！")
print("1. 已排除 2022/11/11 数据。")
print("2. 采用了 3 日滑动平均对校准残余噪声进行平滑。")
print("3. 采用了 Trend-Free Pre-Whitening (TFPW) 修正的 MK 检验。")
print("结果已保存至: refined_trend_analysis.csv")
print("--------------------------------------------------")
```