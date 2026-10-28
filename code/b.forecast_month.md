运行代码前，需要有两个数据：1.校准过的年度预测总量表（CEADs预测值×k）。2.逐日CM原始数据（无疫情时的，用来校准每个月度的正常占比）

```
import pandas as pd
import os

# =========================================================
# 1. 路径设置
# =========================================================
# 原始逐日数据路径 (包含 state, date, sector, value)
path_daily_raw = r"E:\c.year_month_day(20-22)\0.CM_day_province2019,2023,2024.csv"
# 年度预测总量表
path_annual_wide = r"E:\c.year_month_day(20-22)\2.CEADs_CM_forecast_yearxk(20-22).csv"
# 最终结果保存路径
output_path = r"E:\3.CEADs_CM_monthly_forecast(20-22).csv"

# =========================================================
# 2. 计算月度占比因子 (基于逐日数据)
# =========================================================
print("正在从逐日数据计算月度占比因子...")
df_daily = pd.read_csv(path_daily_raw)
df_daily.columns = df_daily.columns.str.strip() # 清理表头空格

# A. 提取时间维度
df_daily['date'] = pd.to_datetime(df_daily['date'])
df_daily['year_tmp'] = df_daily['date'].dt.year
df_daily['month'] = df_daily['date'].dt.month

val_col = 'MtCO2 per day' # 确保这是你逐日表里的数值列名

# B. 聚合计算月占比
df_monthly = df_daily.groupby(['state', 'sector', 'year_tmp', 'month'])[val_col].sum().reset_index()
# 计算该月在当年该行业中的占比
df_monthly['share'] = df_monthly[val_col] / df_monthly.groupby(['state', 'sector', 'year_tmp'])[val_col].transform('sum')

# C. 计算跨年平均占比 (季节性波动基准)
df_share = df_monthly.groupby(['state', 'sector', 'month'])['share'].mean().reset_index()
df_share.rename(columns={'share': 'average_share'}, inplace=True)

# D. 归一化：确保12个月之和严格为1
df_share['average_share'] = df_share['average_share'] / df_share.groupby(['state', 'sector'])['average_share'].transform('sum')

print("✅ 月度占比因子计算完成。")

# =========================================================
# 3. 处理年度预测表 (宽转长)
# =========================================================
print("正在读取年度预测表并进行宽转长操作...")
# 这里我们统一变量名为 df_annual
df_annual = pd.read_csv(path_annual_wide)
df_annual.columns = df_annual.columns.str.strip() # 清理表头空格

# 【宽转长】
# 注意：value_vars 里的名称必须和你的 CSV 表头一模一样（区分大小写和空格）
df_annual_long = df_annual.melt(
    id_vars=['state', 'year'],
    value_vars=['Industry', 'Power', 'Residential', 'Transport'],
    var_name='sector',
    value_name='annual_total'
)

# 清理名称空格，防止关联失败
df_share['state'] = df_share['state'].str.strip()
df_share['sector'] = df_share['sector'].str.strip()
df_annual_long['state'] = df_annual_long['state'].str.strip()
df_annual_long['sector'] = df_annual_long['sector'].str.strip()

# =========================================================
# 4. 提取 2020 -2022 的预测值
# =========================================================
# 现在的结构已经是长表，直接过滤即可，无需 Melt
df_2020 = df_annual_long[df_annual_long['year'] == 2020][['state', 'sector', 'annual_total']].copy()
df_2021 = df_annual_long[df_annual_long['year'] == 2021][['state', 'sector', 'annual_total']].copy()
df_2022 = df_annual_long[df_annual_long['year'] == 2022][['state', 'sector', 'annual_total']].copy()

df_2020 = df_2020.rename(columns={'annual_total': 'total_2020'})
df_2021 = df_2021.rename(columns={'annual_total': 'total_2021'})
df_2022 = df_2022.rename(columns={'annual_total': 'total_2022'})

# =========================================================
# 5. 合并并计算
# =========================================================
# 以占比表为基准
result = pd.merge(df_share, df_2020, on=['state', 'sector'], how='left')
result = pd.merge(result, df_2021, on=['state', 'sector'], how='left')
result = pd.merge(result, df_2022, on=['state', 'sector'], how='left')

#核心计算
result['2020forecast'] = result['average_share'] * result['total_2020']
result['2021forecast'] = result['average_share'] * result['total_2021']
result['2022forecast'] = result['average_share'] * result['total_2022']

# =========================================================
# 6. 输出结果
# =========================================================
final_table = result[['state', 'sector', 'month', 'average_share', '2020forecast', '2021forecast', '2022forecast']]

# 排序
final_table = final_table.sort_values(by=['state', 'sector', 'month']).reset_index(drop=True)

# 保存
final_table.to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"✅ 处理完成！")
print(f"结果预览：\n{final_table.head(12)}")
```