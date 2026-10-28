```
import os
import warnings
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf

warnings.filterwarnings("ignore")

# =========================================================
# 1. 路径与参数设置
# =========================================================
input_file = r"E:\0.CEADs_province_3sector(2003-2019).csv"
output_dir = r"E:\1.forecast_year_CEADs_province"
os.makedirs(output_dir, exist_ok=True)

target_cols = ["Total", "Industry", "Power", "Transport"]
train_start_year = 2003
train_end_year = 2019

# --- 关键修改：预测步长改为 3 ---
forecast_horizon = 3
min_train_size = 5
selection_metric = "MAE"

method_list = ["Naive_LastYear", "ETS", "ARIMA"]
method_priority = {"Naive_LastYear": 0, "ETS": 1, "ARIMA": 2}


# =========================================================
# 2. 基础建模函数 (保持不变)
# =========================================================
def safe_mape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else np.nan


def calc_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    err = y_true - y_pred
    return {"MAE": np.mean(np.abs(err)), "RMSE": np.sqrt(np.mean(err ** 2)), "MAPE": safe_mape(y_true, y_pred)}


def prep_series(s):
    s = pd.to_numeric(s, errors="coerce").astype(float)
    return s.interpolate(method="linear").ffill().bfill() if s.isna().any() else s


def fit_best_ets(y):
    y = pd.Series(y).astype(float).reset_index(drop=True)
    candidates = [
        {"error": "add", "trend": None, "damped_trend": False, "name": "ETS(A,N,N)"},
        {"error": "add", "trend": "add", "damped_trend": False, "name": "ETS(A,A,N)"},
        {"error": "add", "trend": "add", "damped_trend": True, "name": "ETS(A,Ad,N)"},
    ]
    best_aic, best_res, best_name = np.inf, None, None
    for spec in candidates:
        try:
            res = ETSModel(y, error=spec["error"], trend=spec["trend"], damped_trend=spec["damped_trend"],
                           seasonal=None, initialization_method="estimated").fit(disp=False)
            if np.isfinite(res.aic) and res.aic < best_aic:
                best_aic, best_res, best_name = res.aic, res, spec["name"]
        except:
            continue
    if best_res is None: raise RuntimeError("ETS失败")
    return best_res, best_name


def fit_best_arima(y, p_max=2, d_list=(0, 1), q_max=2):
    y = pd.Series(y).astype(float).reset_index(drop=True)
    best_aic, best_res, best_order = np.inf, None, None
    for p in range(p_max + 1):
        for d in d_list:
            for q in range(q_max + 1):
                if p + d + q >= len(y): continue
                try:
                    res = ARIMA(y, order=(p, d, q), trend="c" if d == 0 else "n").fit()
                    if np.isfinite(res.aic) and res.aic < best_aic:
                        best_aic, best_res, best_order = res.aic, res, (p, d, q)
                except:
                    continue
    if best_res is None: raise RuntimeError("ARIMA失败")
    return best_res, best_order


def fit_forecast_by_method(method, y_train, years_train, years_future):
    if method == "Naive_LastYear":
        return np.r_[np.nan, y_train[:-1]], np.repeat(y_train[-1], len(years_future)), "Naive_LastYear"
    elif method == "ETS":
        res, name = fit_best_ets(y_train)
        return np.asarray(res.fittedvalues), np.asarray(res.forecast(len(years_future))), name
    elif method == "ARIMA":
        res, order = fit_best_arima(y_train)
        return np.asarray(res.fittedvalues), np.asarray(res.forecast(len(years_future))), f"ARIMA{order}"
    raise ValueError(f"未知方法: {method}")


# =========================================================
# 3. 绘图相关函数 (优化：视觉衔接)
# =========================================================
def plot_residuals_and_acf(years, residuals, title_prefix, save_path):
    mask = ~np.isnan(residuals)
    res, yrs = residuals[mask], years[mask]
    if len(res) < 2: return
    fig, axes = plt.subplots(2, 1, figsize=(8, 6))
    axes[0].plot(yrs, res, marker="o", linewidth=1.5)
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set_title(f"{title_prefix} - Residuals")
    plot_acf(res, ax=axes[1], lags=min(8, len(res) - 1), zero=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_forecast_visual(years_train, y_train, years_future, y_fc, lower, upper, title, save_path):
    plt.figure(figsize=(10, 5))
    # 历史数据
    plt.plot(years_train, y_train, 'o-', label=f"Historical ({train_start_year}-{train_end_year})", color="black",
             zorder=3)

    # --- 衔接：构造从2019年开始到2022年的点 ---
    connect_years = np.r_[years_train[-1:], years_future]
    connect_vals = np.r_[y_train[-1:], y_fc]

    # 预测数据
    plt.plot(connect_years, connect_vals, 's--', label="Forecast (2020-2022)", color="blue", zorder=2)

    # 置信区间
    plt.fill_between(years_future, lower, upper, color='blue', alpha=0.15, label="95% Confidence Interval")

    plt.title(title)
    plt.xlabel("Year")
    plt.ylabel("Emissions ($MtCO_2$)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(save_path, dpi=200)
    plt.close()


# =========================================================
# 4. 主程序
# =========================================================
def main():
    print(f"载入数据: {input_file}")
    df = pd.read_csv(input_file)
    df.columns = df.columns.str.strip()

    global_history_forecast = []
    global_selection, global_metrics = [], []

    states = df["state"].unique()
    for state_name in states:
        print(f"▶ 处理中: {state_name}")
        state_df = df[df["state"] == state_name].sort_values("year").reset_index(drop=True)
        s_out_dir = os.path.join(output_dir, state_name)
        p_dir = os.path.join(s_out_dir, "plots")
        os.makedirs(p_dir, exist_ok=True)

        train_data = state_df[(state_df["year"] >= train_start_year) & (state_df["year"] <= train_end_year)]
        if len(train_data) < min_train_size: continue

        for col in target_cols:
            if col not in state_df.columns: continue
            try:
                y_train = prep_series(train_data[col]).values
                yrs_train = train_data["year"].values

                # --- 滚动回测 ---
                records = []
                for i in range(min_train_size, len(y_train)):
                    row = {"year": yrs_train[i], "actual": y_train[i]}
                    for m in method_list:
                        try:
                            _, fc, _ = fit_forecast_by_method(m, y_train[:i], yrs_train[:i], [yrs_train[i]])
                            row[m] = fc[0]
                        except:
                            row[m] = np.nan
                    records.append(row)
                bt_df = pd.DataFrame(records)

                # --- 选模型 ---
                met_list = []
                for m in method_list:
                    v = bt_df[["actual", m]].dropna()
                    if not v.empty:
                        m_res = calc_metrics(v["actual"], v[m])
                        m_res.update({"method": m, "state": state_name, "variable": col})
                        met_list.append(m_res)
                met_df = pd.DataFrame(met_list)

                best_m = "Naive_LastYear"
                if not met_df.empty:
                    tmp = met_df.copy()
                    tmp["priority"] = tmp["method"].map(method_priority)
                    best_m = tmp.sort_values([selection_metric, "priority"]).iloc[0]["method"]

                # --- 关键修改：未来的年份数组设为 2020, 2021, 2022 ---
                fut_yrs = np.array([2020, 2021, 2022])
                fitted, fc, detail = fit_forecast_by_method(best_m, y_train, yrs_train, fut_yrs)

                # 计算置信区间 (随步长增加而变宽)
                resid = (y_train - fitted)[~np.isnan(y_train - fitted)]
                sigma = np.std(resid, ddof=1) if len(resid) > 1 else 0
                se = sigma * np.sqrt(np.arange(1, len(fc) + 1))
                low, upp = np.maximum(fc - 1.96 * se, 0), fc + 1.96 * se

                # --- 绘图 (衔接模式) ---
                plot_residuals_and_acf(yrs_train, y_train - fitted, f"{state_name}-{col}",
                                       os.path.join(p_dir, f"{col}_residual_acf.png"))
                plot_forecast_visual(yrs_train, y_train, fut_yrs, fc, low, upp, f"{state_name} - {col} Forecast",
                                     os.path.join(p_dir, f"{col}_forecast_comparison.png"))

                # --- 数据衔接存入记录 ---
                for y, val in zip(yrs_train, y_train):
                    global_history_forecast.append({
                        "state": state_name, "year": y, "variable": col,
                        "value": val, "type": "History", "lower95": np.nan, "upper95": np.nan
                    })
                for y, fv, l, u in zip(fut_yrs, fc, low, upp):
                    global_history_forecast.append({
                        "state": state_name, "year": y, "variable": col,
                        "value": fv, "type": "Forecast", "lower95": l, "upper95": u
                    })

                global_selection.append({"state": state_name, "variable": col, "best_method": best_m, "detail": detail})

            except Exception as e:
                print(f"错误 {state_name}-{col}: {e}")

    # 合并输出
    if global_history_forecast:
        full_df = pd.DataFrame(global_history_forecast)
        full_df.to_csv(os.path.join(output_dir, "Full_History_and_Forecast_2003_2022.csv"), index=False,
                       encoding="utf-8-sig")

        # 单独留一份纯预测表
        full_df[full_df["type"] == "Forecast"].to_csv(os.path.join(output_dir, "Only_Forecast_2020_2022.csv"),
                                                      index=False, encoding="utf-8-sig")

        pd.DataFrame(global_selection).to_csv(os.path.join(output_dir, "ALL_states_best_models.csv"), index=False,
                                              encoding="utf-8-sig")
        print(f"\n🚀 完成！已生成 2020-2022 三期预测数据及衔接图表。")


if __name__ == "__main__":
    main()
```