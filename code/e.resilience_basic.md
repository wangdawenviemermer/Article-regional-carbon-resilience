```
import pandas as pd
import numpy as np

# ==========================================
# 1. 路径设置
# ==========================================
input_path = r"E:\study\g.DATA\b.data_COVID_carbon\phase_method\b.wave_gap_rate_and_fCO2_M.csv"
output_resilience_path = r"E:\study\g.DATA\b.data_COVID_carbon\phase_method\c.Wave_Resilience_Metrics.csv"

# ==========================================
# 2. 加载数据
# ==========================================
df = pd.read_csv(input_path)
df['date'] = pd.to_datetime(df['date'])

# 只分析波段内的数据 (wave_id > 0)
df_waves = df[df['wave_id'] > 0].copy()

# 计算每日偏差 (Deviation)
# Positive value means actual < predicted (emission loss)
df_waves['daily_deviation'] = df_waves['Total_fCO2_M'] - df_waves['Total_aCO2']


# ==========================================
# 3. 计算韧性指标
# ==========================================
def calculate_metrics(group):
    # a. 累计损失面积 (等同于这段时间内的总减排量)
    cumulative_loss = group['daily_deviation'].sum()

    # b. 最大深度 (偏差最大的那一时刻)
    max_depth = group['daily_deviation'].max()

    # c. 寻找最大深度的日期
    max_depth_idx = group['daily_deviation'].idxmax()
    max_depth_date = group.loc[max_depth_idx, 'date']

    # d. 恢复速率计算
    # 逻辑：从达到最深处那天，到波段结束那天，偏差缩小的速率
    end_date = group['date'].max()
    end_deviation = group.iloc[-1]['daily_deviation']

    recovery_days = (end_date - max_depth_date).days

    if recovery_days > 0:
        # 速率 = (起点深度 - 终点深度) / 天数
        recovery_rate = (max_depth - end_deviation) / recovery_days
    else:
        # 如果最深点就是最后一天，则速率为0
        recovery_rate = 0.0

    return pd.Series({
        'start_date': group['date'].min().date(),
        'end_date': end_date.date(),
        'duration': len(group),
        'cumulative_loss_area': cumulative_loss,
        'max_depth': max_depth,
        'peak_impact_date': max_depth_date.date(),
        'recovery_rate': recovery_rate,
        'end_deviation': end_deviation
    })


# 按省份和波段分组计算
resilience_results = df_waves.groupby(['state', 'wave_id']).apply(calculate_metrics, include_groups=False).reset_index()

# ==========================================
# 4. 保存结果
# ==========================================
resilience_results.to_csv(output_resilience_path, index=False, encoding='utf-8-sig')

print(f"✅ 指标计算完成！结果已存至: {output_resilience_path}")

# 展示前5行结果
print(resilience_results.head())
```