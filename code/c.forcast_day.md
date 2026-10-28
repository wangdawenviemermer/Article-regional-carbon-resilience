强度因子计算+逐日预测

```
import pandas as pd
import numpy as np
import os

# ==========================================
# 1. 路径与参数设置
# ==========================================
# 输入：2019, 2023, 2024 的逐日真实数据
path_historical = r"E:\0.CM_day_province2019,2023,2024.csv"
# 输入：2020-2022 的月度预测总量表 (需包含字段: state, sector, year, month, monthly_val)
path_monthly_forecast = r"E:\3.CEADs_CM_monthly_forecast(20-22).csv"
# 输出结果路径
path_final_output = r"E:\Final_Daily_Forecast_2020_2022.csv"
# 【输出2】中间强度因子表（展现每天的相对强弱波动）
path_factor_output = r"E:\2.Daily_Intensity_Factors_Table.csv"

cny_dates = {
    2019: "2019-02-05", 2023: "2023-01-22", 2024: "2024-02-10",
    2020: "2020-01-25", 2021: "2021-02-12", 2022: "2022-02-01"
}

# ==========================================
# 2. 训练阶段：提取“排放指纹”（计算因子）
# ==========================================
print("正在训练：提取历史排放日度波动指纹...")
df_hist = pd.read_csv(path_historical)
df_hist.columns = df_hist.columns.str.strip()
df_hist['date'] = pd.to_datetime(df_hist['date'])
df_hist['state'] = df_hist['state'].str.strip()
df_hist['sector'] = df_hist['sector'].str.strip()

val_col = "MtCO2 per day"
df_hist['year'] = df_hist['date'].dt.year
df_hist['month'] = df_hist['date'].dt.month
df_hist['day'] = df_hist['date'].dt.day
df_hist['weekday'] = df_hist['date'].dt.day_name()

# 计算因子：当日值 / 当月均值
df_hist['month_avg'] = df_hist.groupby(['state', 'sector', 'year', 'month'])[val_col].transform('mean')
df_hist['raw_factor'] = df_hist[val_col] / df_hist['month_avg']

# A. 春节移动因子
lunar_list = []
for yr, dt_str in cny_dates.items():
    if yr in [2019, 2023, 2024]:
        cny_dt = pd.to_datetime(dt_str)
        temp = df_hist[df_hist['year'] == yr].copy()
        temp['dist_to_cny'] = (temp['date'] - cny_dt).dt.days
        lunar_list.append(temp[(temp['dist_to_cny'] >= -7) & (temp['dist_to_cny'] <= 7)])
df_lunar = pd.concat(lunar_list)
lunar_profile = df_lunar.groupby(['state', 'sector', 'dist_to_cny'])['raw_factor'].mean().reset_index()
lunar_profile.rename(columns={'raw_factor': 'lunar_factor'}, inplace=True)

# B. 公历日期指纹
df_no_cny = df_hist.copy()
for yr, dt_str in cny_dates.items():
    if yr in [2019, 2023, 2024]:
        cny_dt = pd.to_datetime(dt_str)
        cny_range = pd.date_range(cny_dt - pd.offsets.Day(7), cny_dt + pd.offsets.Day(7))
        df_no_cny = df_no_cny[~df_no_cny['date'].isin(cny_range)]
daily_profile = df_no_cny.groupby(['state', 'sector', 'month', 'day'])['raw_factor'].mean().reset_index()
daily_profile.rename(columns={'raw_factor': 'base_daily_factor'}, inplace=True)

# C. 星期修正因子
weekday_profile = df_no_cny.groupby(['state', 'sector', 'weekday'])['raw_factor'].mean().reset_index()
weekday_profile['weekday_factor'] = weekday_profile['raw_factor'] / weekday_profile.groupby(['state', 'sector'])['raw_factor'].transform('mean')

# ==========================================
# 3. 预测与分配阶段：合成逐日预测
# ==========================================
print("正在预测：合成 2020-2022 强度因子并分摊月度总量...")

# 1. 准备月度长表
df_mon_raw = pd.read_csv(path_monthly_forecast)
df_mon_long = df_mon_raw.melt(
    id_vars=['state', 'sector', 'month'],
    value_vars=[c for c in df_mon_raw.columns if 'forecast' in c],
    var_name='year_str', value_name='monthly_val'
).copy()
df_mon_long['year'] = df_mon_long['year_str'].str.extract('(\d+)').astype(int)
df_mon_long['state'] = df_mon_long['state'].str.strip()
df_mon_long['sector'] = df_mon_long['sector'].str.strip()

# 2. 构建 2020-2022 全维度骨架
states, sectors = df_mon_long['state'].unique(), df_mon_long['sector'].unique()
dates_pred = pd.date_range("2020-01-01", "2022-12-31", freq='D')
index = pd.MultiIndex.from_product([dates_pred, states, sectors], names=['date', 'state', 'sector'])
grid = pd.DataFrame(index=index).reset_index()

grid['year'], grid['month'], grid['day'] = grid['date'].dt.year, grid['date'].dt.month, grid['date'].dt.day
grid['weekday'] = grid['date'].dt.day_name()

# 3. 因子挂载
grid = pd.merge(grid, daily_profile, on=['state', 'sector', 'month', 'day'], how='left')
grid = pd.merge(grid, weekday_profile[['state', 'sector', 'weekday', 'weekday_factor']],
                on=['state', 'sector', 'weekday'], how='left')

# 春节对齐
grid['dist_to_cny'] = 999
for yr, dt_str in cny_dates.items():
    if yr in [2020, 2021, 2022]:
        mask = grid['year'] == yr
        grid.loc[mask, 'dist_to_cny'] = (grid.loc[mask, 'date'] - pd.to_datetime(dt_str)).dt.days
grid = pd.merge(grid, lunar_profile, on=['state', 'sector', 'dist_to_cny'], how='left')

# --- 关键：生成【复合日度强度因子】 ---
grid['Daily_Intensity_Weight'] = np.where(
    grid['lunar_factor'].notna(),
    grid['lunar_factor'],
    grid['base_daily_factor'] * grid['weekday_factor']
)
grid['Daily_Intensity_Weight'] = grid['Daily_Intensity_Weight'].fillna(1.0)

# ==========================================
# 4. 中间表导出：强度因子表 (Intensity Factors)
# ==========================================
# 这里的因子反映的是该日在当月均值水平上的相对强弱（比如 1.1 代表超出均值 10%）
factor_table = grid[['state', 'sector', 'date', 'weekday', 'dist_to_cny', 'Daily_Intensity_Weight']].copy()
factor_table.to_csv(path_factor_output, index=False, encoding='utf-8-sig')
print(f"✅ 中间强度因子表已导出: {path_factor_output}")

# ==========================================
# 5. 归一化分配并输出最终结果
# ==========================================
# 每个月内的因数求和并归一化，确保分配后的总和等于月预测值
group_keys = ['state', 'sector', 'year', 'month']
grid['weight_norm'] = grid['Daily_Intensity_Weight'] / grid.groupby(group_keys)['Daily_Intensity_Weight'].transform('sum')

final = pd.merge(grid, df_mon_long[['state', 'sector', 'year', 'month', 'monthly_val']],
                 on=['state', 'sector', 'year', 'month'], how='inner')

final['forecast_MtCO2_per_day'] = final['monthly_val'] * final['weight_norm']

# 保存最终预测结果
result = final[['state', 'sector', 'date', 'forecast_MtCO2_per_day']].copy()
result['date'] = result['date'].dt.strftime('%Y/%m/%d')
result.sort_values(by=['state', 'sector', 'date']).to_csv(path_final_output, index=False, encoding='utf-8-sig')

print(f"✅ 最终逐日预测结果已导出: {path_final_output}")
```