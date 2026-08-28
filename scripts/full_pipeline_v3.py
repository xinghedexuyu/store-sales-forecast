# -*- coding: utf-8 -*-
"""
Store Sales 全流程 v3

升级点：
1. 特征工程更完整：更多滞后、滚动、节假日距离、促销、油价、交易量特征
2. 多模型对比：上一周基线、Ridge、LightGBM、XGBoost、LightGBM+XGBoost 融合
3. 特征分析：相关性、特征重要性
4. 高级交互式可视化：Plotly 生成 HTML
5. 输出 Kaggle 标准提交文件

运行：python scripts/full_pipeline_v3.py
"""

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import plotly.express as px
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = Path(r"C:\Users\xingh\Desktop\sales_analysis.csv")
TEST_CSV = ROOT / "data" / "test.csv"
HOLIDAY_CSV = ROOT / "data" / "holidays_events.csv"
OIL_CSV = ROOT / "data" / "oil.csv"
OUT_DIR = ROOT / "outputs_v3"
PLOT_DIR = ROOT / "plots_v3"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_END = pd.Timestamp("2017-08-15")
VAL_START = pd.Timestamp("2017-08-01")
VAL_END = pd.Timestamp("2017-08-15")
TEST_START = pd.Timestamp("2017-08-16")
TEST_END = pd.Timestamp("2017-08-31")

FEATURES = [
    "store_code",
    "family_code",
    "store_type_code",
    "state_code",
    "cluster",
    "year",
    "month",
    "day",
    "dayofweek",
    "weekofyear",
    "is_weekend",
    "is_month_end",
    "is_quarter_end",
    "is_payday",
    "is_holiday",
    "days_to_holiday",
    "days_since_holiday",
    "onpromotion",
    "onpromotion_lag7",
    "onpromotion_roll7",
    "oil_price",
    "oil_lag7",
    "oil_roll7",
    "daily_transactions",
    "transactions_lag7",
    "transactions_roll7",
    "store_avg_sales",
    "family_avg_sales",
    "store_family_avg",
    "sales_dow_mean",
    "sales_lag_1",
    "sales_lag_7",
    "sales_lag_14",
    "sales_lag_28",
    "sales_roll7",
    "sales_roll14",
    "sales_roll28",
    "sales_std7",
    "sales_max7",
    "sales_min7",
]


def rmsle_log(y_true, log_pred):
    """在 log1p 空间计算 RMSLE，值越小越好。"""
    return float(np.sqrt(np.mean((np.log1p(y_true) - log_pred) ** 2)))


def rmsle(y_true, y_pred):
    """普通 RMSLE。"""
    y_pred = np.clip(y_pred, 0, None)
    return float(np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2)))


def load_data():
    """读取宽表并重建节假日标记。"""
    print("1/10 读取数据 ...")
    df = pd.read_csv(CSV_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)

    holidays = pd.read_csv(HOLIDAY_CSV)
    holidays["date"] = pd.to_datetime(holidays["date"])
    holidays["transferred"] = holidays["transferred"].map(
        lambda v: 1 if str(v).lower() in ("1", "true", "1.0") else 0
    )
    holiday_dates = set(holidays.loc[holidays["transferred"] == 0, "date"])
    df["is_holiday"] = df["date"].isin(holiday_dates).astype(int)
    return df, holiday_dates


def add_features(df, holiday_dates):
    """多维特征工程：时间、门店、品类、促销、油价、交易量、历史销量。"""
    print("2/10 特征工程 ...")
    df = df.sort_values("date").reset_index(drop=True)
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["dayofweek"] = df["date"].dt.dayofweek
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["is_month_end"] = (df["day"] >= 28).astype(int)
    df["is_quarter_end"] = df["date"].dt.is_quarter_end.astype(int)
    df["is_payday"] = df["day"].isin([15, 30]).astype(int)

    df["store_code"] = df["store_nbr"].astype("category").cat.codes
    df["family_code"] = df["family"].astype("category").cat.codes
    df["store_type_code"] = df["store_type"].astype("category").cat.codes
    df["state_code"] = df["state"].astype("category").cat.codes

    hol = pd.DataFrame({"date": sorted(holiday_dates)}).drop_duplicates().sort_values("date")
    left = df[["date"]].sort_values("date")
    next_hol = pd.merge_asof(left, hol, on="date", direction="forward")
    prev_hol = pd.merge_asof(left, hol, on="date", direction="backward")
    df["days_to_holiday"] = (next_hol["date"].values - df["date"].values) / pd.Timedelta(days=1)
    df["days_since_holiday"] = (df["date"].values - prev_hol["date"].values) / pd.Timedelta(days=1)

    agg_base = df[df["date"] <= pd.Timestamp("2017-07-31")]
    store_avg = agg_base.groupby("store_nbr")["sales"].mean().rename("store_avg_sales")
    family_avg = agg_base.groupby("family")["sales"].mean().rename("family_avg_sales")
    store_family_avg = agg_base.groupby(["store_nbr", "family"])["sales"].mean().rename("store_family_avg")
    dow_mean = agg_base.groupby(["store_nbr", "family", "dayofweek"])["sales"].mean().rename("sales_dow_mean")
    df = (
        df.merge(store_avg, on="store_nbr", how="left")
        .merge(family_avg, on="family", how="left")
        .merge(store_family_avg, on=["store_nbr", "family"], how="left")
        .merge(dow_mean, on=["store_nbr", "family", "dayofweek"], how="left")
    )

    grouped = df.groupby(["store_nbr", "family"], sort=False)["sales"]
    df["sales_lag_1"] = grouped.shift(1)
    df["sales_lag_7"] = grouped.shift(7)
    df["sales_lag_14"] = grouped.shift(14)
    df["sales_lag_28"] = grouped.shift(28)
    df["sales_roll7"] = grouped.transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
    df["sales_roll14"] = grouped.transform(lambda s: s.shift(1).rolling(14, min_periods=1).mean())
    df["sales_roll28"] = grouped.transform(lambda s: s.shift(1).rolling(28, min_periods=1).mean())
    df["sales_std7"] = grouped.transform(lambda s: s.shift(1).rolling(7, min_periods=1).std())
    df["sales_max7"] = grouped.transform(lambda s: s.shift(1).rolling(7, min_periods=1).max())
    df["sales_min7"] = grouped.transform(lambda s: s.shift(1).rolling(7, min_periods=1).min())

    promo_grouped = df.groupby(["store_nbr", "family"], sort=False)["onpromotion"]
    df["onpromotion_lag7"] = promo_grouped.shift(7)
    df["onpromotion_roll7"] = promo_grouped.transform(
        lambda s: s.shift(1).rolling(7, min_periods=1).mean()
    )

    oil_grouped = df.groupby("store_nbr", sort=False)["oil_price"]
    df["oil_lag7"] = oil_grouped.shift(7)
    df["oil_roll7"] = oil_grouped.transform(
        lambda s: s.shift(1).rolling(7, min_periods=1).mean()
    )

    trans_grouped = df.groupby("store_nbr", sort=False)["daily_transactions"]
    df["transactions_lag7"] = trans_grouped.shift(7)
    df["transactions_roll7"] = trans_grouped.transform(
        lambda s: s.shift(1).rolling(7, min_periods=1).mean()
    )
    return df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)


def feature_analysis(df):
    """特征分析：与销量的相关性 + 特征重要性图。"""
    print("3/10 特征分析 ...")
    numeric_cols = [c for c in FEATURES if c in df.columns]
    corr = df[numeric_cols + ["sales"]].corr(numeric_only=True)["sales"].drop("sales").sort_values()
    corr_df = corr.reset_index()
    corr_df.columns = ["feature", "corr"]
    corr_df.to_csv(OUT_DIR / "feature_sales_correlation.csv", index=False)

    fig = px.bar(
        corr_df,
        x="corr",
        y="feature",
        orientation="h",
        title="特征与销量的相关性",
        color="corr",
        color_continuous_scale="RdBu_r",
    )
    fig.update_layout(height=700, yaxis=dict(autorange="reversed"))
    fig.write_html(PLOT_DIR / "feature_correlation.html")
    print("  相关性表：", OUT_DIR / "feature_sales_correlation.csv")


def compare_models(df):
    """多模型对比：基线、线性、LightGBM、XGBoost、融合。"""
    print("4/10 多模型对比 ...")
    tr = df[df["date"] < VAL_START].dropna(subset=FEATURES)
    va = df[(df["date"] >= VAL_START) & (df["date"] <= VAL_END)].dropna(subset=FEATURES)
    y_tr_log = np.log1p(tr["sales"])
    y_va = va["sales"]
    results = {}

    results["baseline_last_week"] = rmsle(y_va, va["sales_lag_7"].fillna(0))
    print("  基线 RMSLE：", round(results["baseline_last_week"], 4))

    scaler = StandardScaler().fit(tr[FEATURES])
    ridge = Ridge(alpha=1.0)
    ridge.fit(scaler.transform(tr[FEATURES]), y_tr_log)
    ridge_log = ridge.predict(scaler.transform(va[FEATURES]))
    results["ridge"] = rmsle_log(y_va, ridge_log)
    print("  Ridge RMSLE：", round(results["ridge"], 4))

    lgb_model = lgb.LGBMRegressor(
        n_estimators=150, learning_rate=0.05, num_leaves=31,
        random_state=42, verbose=-1, n_jobs=-1,
    )
    lgb_model.fit(tr[FEATURES], y_tr_log)
    lgb_log = lgb_model.predict(va[FEATURES])
    results["lightgbm"] = rmsle_log(y_va, lgb_log)
    print("  LightGBM RMSLE：", round(results["lightgbm"], 4))

    xgb_model = xgb.XGBRegressor(
        n_estimators=150, learning_rate=0.05, max_depth=6,
        tree_method="hist", random_state=42, n_jobs=-1, verbosity=0,
    )
    xgb_model.fit(tr[FEATURES], y_tr_log)
    xgb_log = xgb_model.predict(va[FEATURES])
    results["xgboost"] = rmsle_log(y_va, xgb_log)
    print("  XGBoost RMSLE：", round(results["xgboost"], 4))

    ens_log = (lgb_log + xgb_log) / 2
    results["ensemble_lgb_xgb"] = rmsle_log(y_va, ens_log)
    print("  融合 RMSLE：", round(results["ensemble_lgb_xgb"], 4))

    pd.Series(results).to_csv(OUT_DIR / "model_comparison.csv", header=["rmsle"])
    fig = px.bar(
        pd.Series(results).reset_index().rename(columns={"index": "model", 0: "rmsle"}),
        x="model",
        y="rmsle",
        title="模型对比（RMSLE 越小越好）",
        color="rmsle",
        color_continuous_scale="YlOrRd_r",
    )
    fig.write_html(PLOT_DIR / "model_comparison.html")

    val_df = pd.DataFrame({"actual": y_va, "pred": np.expm1(ens_log), "family": va["family"]})
    return results, val_df


def train_final_models(df):
    """训练最终 LightGBM 和 XGBoost，并输出特征重要性。"""
    print("5/10 训练最终模型 ...")
    tr = df[df["date"] <= TRAIN_END].dropna(subset=FEATURES)
    y_tr_log = np.log1p(tr["sales"])

    lgb_model = lgb.LGBMRegressor(
        n_estimators=200, learning_rate=0.05, num_leaves=31,
        random_state=42, verbose=-1, n_jobs=-1,
    )
    lgb_model.fit(tr[FEATURES], y_tr_log)

    xgb_model = xgb.XGBRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=6,
        tree_method="hist", random_state=42, n_jobs=-1, verbosity=0,
    )
    xgb_model.fit(tr[FEATURES], y_tr_log)

    imp = pd.DataFrame({"feature": FEATURES, "importance": lgb_model.feature_importances_})
    imp = imp.sort_values("importance", ascending=False).reset_index(drop=True)
    imp.to_csv(OUT_DIR / "feature_importance.csv", index=False)
    fig = px.bar(
        imp.head(20).iloc[::-1],
        x="importance",
        y="feature",
        orientation="h",
        title="LightGBM 特征重要性 Top 20",
        color="importance",
        color_continuous_scale="Blues",
    )
    fig.update_layout(height=700)
    fig.write_html(PLOT_DIR / "feature_importance.html")
    return lgb_model, xgb_model


def predict_test(df, lgb_model, xgb_model, holiday_dates):
    """逐日递归预测未来 16 天，融合 LightGBM 和 XGBoost。"""
    print("6/10 预测未来 16 天 ...")
    store_codes = df.groupby("store_nbr")["store_code"].first().to_dict()
    family_codes = df.groupby("family")["family_code"].first().to_dict()
    type_codes = df.groupby("store_nbr")["store_type_code"].first().to_dict()
    state_codes = df.groupby("store_nbr")["state_code"].first().to_dict()
    cluster_map = df.groupby("store_nbr")["cluster"].first().to_dict()
    store_avg = df.groupby("store_nbr")["store_avg_sales"].first().to_dict()
    family_avg = df.groupby("family")["family_avg_sales"].first().to_dict()
    store_family_avg = df.groupby(["store_nbr", "family"])["store_family_avg"].first().to_dict()
    dow_mean_map = df.groupby(["store_nbr", "family", "dayofweek"])["sales_dow_mean"].first().to_dict()
    trans_avg = df.groupby("store_nbr")["daily_transactions"].mean().to_dict()

    history = df[df["date"] <= TRAIN_END].sort_values(["store_nbr", "family", "date"])
    recent = {}
    recent_promo = {}
    recent_trans = {}
    for key, grp in history.groupby(["store_nbr", "family"], sort=False):
        recent[key] = list(grp["sales"].tail(28))
        recent_promo[key] = list(grp["onpromotion"].tail(28))
        recent_trans[key] = list(grp["daily_transactions"].tail(28))

    test_info = pd.read_csv(TEST_CSV)
    test_info["date"] = pd.to_datetime(test_info["date"])
    promo_map = {
        (r.date, r.store_nbr, r.family): r.onpromotion
        for r in test_info.itertuples(index=False)
    }

    oil = pd.read_csv(OIL_CSV)
    oil["date"] = pd.to_datetime(oil["date"])
    oil = oil.sort_values("date").ffill()
    oil_map = dict(zip(oil["date"], oil["dcoilwtico"]))
    hol_dates = np.array(sorted(holiday_dates), dtype="datetime64[D]")

    def days_to_holiday(date):
        idx = np.searchsorted(hol_dates, np.datetime64(date, "D"))
        return int((hol_dates[idx] - np.datetime64(date, "D")).astype(int)) if idx < len(hol_dates) else np.nan

    def days_since_holiday(date):
        idx = np.searchsorted(hol_dates, np.datetime64(date, "D"))
        return int((np.datetime64(date, "D") - hol_dates[idx - 1]).astype(int)) if idx > 0 else np.nan

    def mean_last(arr, n):
        return float(np.mean(arr[-n:])) if arr else np.nan

    def std_last(arr, n):
        return float(np.std(arr[-n:])) if len(arr) >= n else np.nan

    def max_last(arr, n):
        return float(np.max(arr[-n:])) if arr else np.nan

    def min_last(arr, n):
        return float(np.min(arr[-n:])) if arr else np.nan

    rows = []
    for date in pd.date_range(TEST_START, TEST_END, freq="D"):
        date_rows = []
        for (store, family), seq in recent.items():
            date_rows.append(
                {
                    "date": date,
                    "store_nbr": store,
                    "family": family,
                    "store_code": store_codes[store],
                    "family_code": family_codes[family],
                    "store_type_code": type_codes[store],
                    "state_code": state_codes[store],
                    "cluster": cluster_map[store],
                    "year": date.year,
                    "month": date.month,
                    "day": date.day,
                    "dayofweek": date.dayofweek,
                    "weekofyear": date.isocalendar().week,
                    "is_weekend": int(date.dayofweek >= 5),
                    "is_month_end": int(date.day >= 28),
                    "is_quarter_end": int(date.month in (3, 6, 9, 12) and date.day >= 28),
                    "is_payday": int(date.day in (15, 30)),
                    "is_holiday": int(date in holiday_dates),
                    "days_to_holiday": days_to_holiday(date),
                    "days_since_holiday": days_since_holiday(date),
                    "onpromotion": promo_map.get((date, store, family), 0),
                    "onpromotion_lag7": recent_promo[(store, family)][-7] if len(recent_promo[(store, family)]) >= 7 else np.nan,
                    "onpromotion_roll7": mean_last(recent_promo[(store, family)], 7),
                    "oil_price": oil_map.get(date, np.nan),
                    "oil_lag7": oil_map.get(date - pd.Timedelta(days=7), np.nan),
                    "oil_roll7": float(np.nanmean([oil_map.get(date - pd.Timedelta(days=i), np.nan) for i in range(7)])),
                    "daily_transactions": trans_avg.get(store, 0),
                    "transactions_lag7": recent_trans[(store, family)][-7] if len(recent_trans[(store, family)]) >= 7 else np.nan,
                    "transactions_roll7": mean_last(recent_trans[(store, family)], 7),
                    "store_avg_sales": store_avg.get(store, 0),
                    "family_avg_sales": family_avg.get(family, 0),
                    "store_family_avg": store_family_avg.get((store, family), 0),
                    "sales_dow_mean": dow_mean_map.get((store, family, date.dayofweek), np.nan),
                    "sales_lag_1": seq[-1] if seq else np.nan,
                    "sales_lag_7": seq[-7] if len(seq) >= 7 else np.nan,
                    "sales_lag_14": seq[-14] if len(seq) >= 14 else np.nan,
                    "sales_lag_28": seq[-28] if len(seq) >= 28 else np.nan,
                    "sales_roll7": mean_last(seq, 7),
                    "sales_roll14": mean_last(seq, 14),
                    "sales_roll28": mean_last(seq, 28),
                    "sales_std7": std_last(seq, 7),
                    "sales_max7": max_last(seq, 7),
                    "sales_min7": min_last(seq, 7),
                }
            )
        x_test = pd.DataFrame(date_rows)[FEATURES].fillna(0)
        lgb_log = lgb_model.predict(x_test)
        xgb_log = xgb_model.predict(x_test)
        ens_log = (lgb_log + xgb_log) / 2
        preds = np.expm1(ens_log)
        for row, pred in zip(date_rows, preds):
            pred = max(0.0, float(pred))
            recent[(row["store_nbr"], row["family"])].append(pred)
            recent_promo[(row["store_nbr"], row["family"])].append(row["onpromotion"])
            recent_trans[(row["store_nbr"], row["family"])].append(row["daily_transactions"])
            rows.append({**row, "sales": pred})

    return pd.DataFrame(rows)[["date", "store_nbr", "family", "sales"]]


def save_submission(pred_df):
    """按 sample_submission 的 id 顺序导出提交文件。"""
    print("7/10 生成提交文件 ...")
    test_info = pd.read_csv(TEST_CSV)
    test_info["date"] = pd.to_datetime(test_info["date"])
    merged = test_info.merge(pred_df, on=["date", "store_nbr", "family"], how="left")
    merged["sales"] = merged["sales"].fillna(0)
    submission = merged[["id", "sales"]].sort_values("id")
    submission.to_csv(OUT_DIR / "submission.csv", index=False)
    return submission


def make_business_plots(df, val_df):
    """生成高级交互式业务图表。"""
    print("8/10 生成业务图表 ...")

    daily = df.groupby("date")["sales"].sum().reset_index()
    daily["rolling7"] = daily["sales"].rolling(7).mean()
    fig = px.line(
        daily,
        x="date",
        y=["sales", "rolling7"],
        title="每日总销量与 7 日移动平均",
        labels={"value": "销量", "date": "日期", "variable": "类型"},
    )
    fig.write_html(PLOT_DIR / "01_daily_sales_trend.html")

    seasonal = df.pivot_table(index="dayofweek", columns="month", values="sales", aggfunc="mean")
    seasonal.index = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    fig = px.imshow(
        seasonal,
        text_auto=".0f",
        title="周几 x 月份 平均销量热力图",
        color_continuous_scale="YlOrRd",
    )
    fig.update_layout(height=500)
    fig.write_html(PLOT_DIR / "02_seasonal_heatmap.html")

    top_family = df.groupby("family")["sales"].mean().sort_values(ascending=False).head(10)
    fig = px.bar(
        top_family.reset_index(),
        x="sales",
        y="family",
        orientation="h",
        title="平均销量最高的 10 个品类",
        color="sales",
        color_continuous_scale="Blues",
    )
    fig.update_layout(yaxis=dict(autorange="reversed"))
    fig.write_html(PLOT_DIR / "03_top_families.html")

    top_store = df.groupby("store_nbr")["sales"].sum().sort_values(ascending=False).head(10)
    fig = px.bar(
        top_store.reset_index(),
        x="sales",
        y="store_nbr",
        orientation="h",
        title="总销量最高的 10 家门店",
        color="sales",
        color_continuous_scale="Oranges",
    )
    fig.update_layout(yaxis=dict(autorange="reversed"))
    fig.write_html(PLOT_DIR / "04_top_stores.html")

    holiday_avg = df.groupby("is_holiday")["sales"].mean().reset_index()
    holiday_avg["is_holiday"] = holiday_avg["is_holiday"].map({0: "普通日", 1: "节假日"})
    fig = px.bar(
        holiday_avg,
        x="is_holiday",
        y="sales",
        title="节假日 vs 普通日平均销量",
        color="is_holiday",
        color_discrete_map={"普通日": "#94A3B8", "节假日": "#EF4444"},
    )
    fig.write_html(PLOT_DIR / "05_holiday_effect.html")

    fig = px.scatter(
        val_df,
        x="actual",
        y="pred",
        title="验证集：实际销量 vs 融合模型预测",
        labels={"actual": "实际销量", "pred": "预测销量"},
        opacity=0.2,
        color="family",
    )
    fig.add_shape(type="line", x0=0, y0=0, x1=val_df["actual"].max(), y1=val_df["actual"].max(), line=dict(color="black", dash="dash"))
    fig.write_html(PLOT_DIR / "06_validation_scatter.html")

    corr = pd.read_csv(OUT_DIR / "feature_sales_correlation.csv")
    corr.columns = ["feature", "corr"]
    fig = px.bar(
        corr.head(20).iloc[::-1],
        x="corr",
        y="feature",
        orientation="h",
        title="特征与销量相关性 Top 20",
        color="corr",
        color_continuous_scale="RdBu_r",
    )
    fig.update_layout(height=700)
    fig.write_html(PLOT_DIR / "07_feature_correlation.html")

    print("  图表目录：", PLOT_DIR)


def main():
    df, holiday_dates = load_data()
    df = add_features(df, holiday_dates)
    feature_analysis(df)
    results, val_df = compare_models(df)
    lgb_model, xgb_model = train_final_models(df)
    pred_df = predict_test(df, lgb_model, xgb_model, holiday_dates)
    submission = save_submission(pred_df)
    make_business_plots(df, val_df)

    metrics = {
        "model_comparison": results,
        "best_model": min(results, key=results.get),
        "final_test_rows": len(pred_df),
    }
    with open(OUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("9/10 保存指标 ...")
    print("10/10 完成！")
    print("提交文件：", OUT_DIR / "submission.csv")
    print("指标文件：", OUT_DIR / "metrics.json")
    print(submission.head(10))


if __name__ == "__main__":
    main()
