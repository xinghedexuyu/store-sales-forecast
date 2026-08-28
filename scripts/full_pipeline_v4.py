# -*- coding: utf-8 -*-
"""
Store Sales 全流程 v4

升级点：
1. 约 50 维特征：时间、节假日距离、促销、油价、交易量、门店/品类历史统计
2. 3 段滚动交叉验证：基线、Ridge、LightGBM、XGBoost、加权融合
3. 加权融合：在验证集上搜索 LightGBM 与 XGBoost 的最优权重
4. 缺失特征回填，不再简单填 0
5. 自动生成业务结论报告
6. 统一风格的交互式图表 + 综合 Dashboard

运行：python scripts/full_pipeline_v4.py
"""

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import xgboost as xgb
from plotly.subplots import make_subplots
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
OUT_DIR = ROOT / "outputs_v4"
PLOT_DIR = ROOT / "plots_v4"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_END = pd.Timestamp("2017-08-15")
TEST_START = pd.Timestamp("2017-08-16")
TEST_END = pd.Timestamp("2017-08-31")

FOLDS = [
    ("2017-06-30", "2017-07-01", "2017-07-15"),
    ("2017-07-15", "2017-07-16", "2017-07-31"),
    ("2017-07-31", "2017-08-01", "2017-08-15"),
]

FEATURES = [
    "store_code",
    "family_code",
    "store_type_code",
    "state_code",
    "cluster",
    "year",
    "month",
    "day",
    "dayofyear",
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
    "promo_freq28",
    "oil_price",
    "oil_lag7",
    "oil_roll7",
    "oil_pct_change_7",
    "daily_transactions",
    "transactions_lag7",
    "transactions_roll7",
    "transactions_per_sales",
    "store_avg_sales",
    "family_avg_sales",
    "store_family_avg",
    "sales_dow_mean",
    "family_share",
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
    "sales_pct_change_7",
    "sales_trend",
    "holiday_promo_interact",
    "weekend_promo_interact",
]


def rmsle_log(y_true, log_pred):
    return float(np.sqrt(np.mean((np.log1p(y_true) - log_pred) ** 2)))


def rmsle(y_true, y_pred):
    y_pred = np.clip(y_pred, 0, None)
    return float(np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2)))


def load_data():
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
    print("2/10 多维特征工程 ...")
    df = df.sort_values("date").reset_index(drop=True)
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["dayofyear"] = df["date"].dt.dayofyear
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
    family_total = agg_base.groupby("family")["sales"].sum()
    family_share = (family_total / agg_base["sales"].sum()).rename("family_share")
    df = (
        df.merge(store_avg, on="store_nbr", how="left")
        .merge(family_avg, on="family", how="left")
        .merge(store_family_avg, on=["store_nbr", "family"], how="left")
        .merge(dow_mean, on=["store_nbr", "family", "dayofweek"], how="left")
        .merge(family_share, on="family", how="left")
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
    df["onpromotion_roll7"] = promo_grouped.transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
    df["promo_flag"] = (df["onpromotion"] > 0).astype(int)
    df["promo_freq28"] = df.groupby(["store_nbr", "family"], sort=False)["promo_flag"].transform(
        lambda s: s.shift(1).rolling(28, min_periods=1).sum()
    )

    oil_grouped = df.groupby("store_nbr", sort=False)["oil_price"]
    df["oil_lag7"] = oil_grouped.shift(7)
    df["oil_roll7"] = oil_grouped.transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
    df["oil_pct_change_7"] = (df["oil_price"] - df["oil_lag7"]) / df["oil_lag7"]

    trans_grouped = df.groupby("store_nbr", sort=False)["daily_transactions"]
    df["transactions_lag7"] = trans_grouped.shift(7)
    df["transactions_roll7"] = trans_grouped.transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
    df["transactions_per_sales"] = df["daily_transactions"] / (df["sales"] + 1)
    df["sales_pct_change_7"] = (df["sales_lag_7"] - df["sales_lag_14"]) / df["sales_lag_14"]
    df["sales_trend"] = df["sales_roll7"] / df["store_family_avg"]
    df["holiday_promo_interact"] = df["is_holiday"] * (df["onpromotion"] > 0).astype(int)
    df["weekend_promo_interact"] = df["is_weekend"] * (df["onpromotion"] > 0).astype(int)

    df = df.replace([np.inf, -np.inf], np.nan)
    return df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)


def run_cv(df):
    print("3/10 3 段滚动交叉验证 ...")
    lgb_logs, xgb_logs, ridge_logs, base_preds, y_list = [], [], [], [], []
    fold_rows = []
    val_df = None

    for train_end, val_start, val_end in FOLDS:
        tr = df[df["date"] <= pd.Timestamp(train_end)].dropna(subset=FEATURES)
        va = df[(df["date"] >= pd.Timestamp(val_start)) & (df["date"] <= pd.Timestamp(val_end))].dropna(subset=FEATURES)
        y_tr_log = np.log1p(tr["sales"])
        y_va = va["sales"].values

        base_pred = va["sales_lag_7"].fillna(0).values
        base_preds.append(base_pred)

        scaler = StandardScaler().fit(tr[FEATURES])
        ridge = Ridge(alpha=1.0)
        ridge.fit(scaler.transform(tr[FEATURES]), y_tr_log)
        ridge_log = ridge.predict(scaler.transform(va[FEATURES]))

        lgb_model = lgb.LGBMRegressor(
            n_estimators=120, learning_rate=0.05, num_leaves=31,
            random_state=42, verbose=-1, n_jobs=-1,
        )
        lgb_model.fit(tr[FEATURES], y_tr_log)
        lgb_log = lgb_model.predict(va[FEATURES])

        xgb_model = xgb.XGBRegressor(
            n_estimators=120, learning_rate=0.05, max_depth=6,
            tree_method="hist", random_state=42, n_jobs=-1, verbosity=0,
        )
        xgb_model.fit(tr[FEATURES], y_tr_log)
        xgb_log = xgb_model.predict(va[FEATURES])

        lgb_logs.append(lgb_log)
        xgb_logs.append(xgb_log)
        ridge_logs.append(ridge_log)
        y_list.append(y_va)

        fold_rows.append(
            {
                "fold": f"{train_end} -> {val_end}",
                "baseline": rmsle(y_va, base_pred),
                "ridge": rmsle_log(y_va, ridge_log),
                "lightgbm": rmsle_log(y_va, lgb_log),
                "xgboost": rmsle_log(y_va, xgb_log),
            }
        )
        if train_end == FOLDS[-1][0]:
            val_df = pd.DataFrame(
                {
                    "actual": y_va,
                    "log_lgb": lgb_log,
                    "log_xgb": xgb_log,
                    "family": va["family"].values,
                }
            )

    all_y = np.concatenate(y_list)
    all_lgb = np.concatenate(lgb_logs)
    all_xgb = np.concatenate(xgb_logs)
    all_ridge = np.concatenate(ridge_logs)
    all_base = np.concatenate(base_preds)

    results = {
        "baseline_last_week": rmsle(all_y, all_base),
        "ridge": rmsle_log(all_y, all_ridge),
        "lightgbm": rmsle_log(all_y, all_lgb),
        "xgboost": rmsle_log(all_y, all_xgb),
    }

    best_w, best_score = 0.5, np.inf
    for w in np.arange(0, 1.001, 0.05):
        ens = w * all_lgb + (1 - w) * all_xgb
        score = rmsle_log(all_y, ens)
        if score < best_score:
            best_w, best_score = float(w), score
    results["ensemble_weighted"] = best_score

    val_df["pred"] = np.expm1(best_w * val_df["log_lgb"] + (1 - best_w) * val_df["log_xgb"])
    print("  折级结果：")
    for row in fold_rows:
        print("  ", row)
    print("  最优融合权重 w(LightGBM) =", round(best_w, 2))
    return results, best_w, fold_rows, val_df


def train_final_models(df):
    print("4/10 训练最终模型 ...")
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
    return lgb_model, xgb_model, imp


def backfill_test_features(x_test):
    for col in [
        "sales_lag_1", "sales_lag_7", "sales_lag_14", "sales_lag_28",
        "sales_roll7", "sales_roll14", "sales_roll28", "sales_std7",
        "sales_max7", "sales_min7", "sales_dow_mean", "sales_trend",
    ]:
        x_test[col] = x_test[col].fillna(x_test["store_family_avg"])
    x_test["sales_pct_change_7"] = x_test["sales_pct_change_7"].fillna(0)
    x_test["oil_lag7"] = x_test["oil_lag7"].fillna(x_test["oil_price"])
    x_test["oil_roll7"] = x_test["oil_roll7"].fillna(x_test["oil_price"])
    x_test["oil_pct_change_7"] = x_test["oil_pct_change_7"].fillna(0)
    x_test["transactions_lag7"] = x_test["transactions_lag7"].fillna(x_test["transactions_roll7"])
    x_test["transactions_roll7"] = x_test["transactions_roll7"].fillna(x_test["daily_transactions"])
    x_test["days_to_holiday"] = x_test["days_to_holiday"].fillna(30)
    x_test["days_since_holiday"] = x_test["days_since_holiday"].fillna(0)
    x_test["transactions_per_sales"] = x_test["transactions_per_sales"].fillna(0)
    return x_test.fillna(0)


def predict_test(df, lgb_model, xgb_model, holiday_dates, best_w):
    print("5/10 预测未来 16 天 ...")
    store_codes = df.groupby("store_nbr")["store_code"].first().to_dict()
    family_codes = df.groupby("family")["family_code"].first().to_dict()
    type_codes = df.groupby("store_nbr")["store_type_code"].first().to_dict()
    state_codes = df.groupby("store_nbr")["state_code"].first().to_dict()
    cluster_map = df.groupby("store_nbr")["cluster"].first().to_dict()
    store_avg = df.groupby("store_nbr")["store_avg_sales"].first().to_dict()
    family_avg = df.groupby("family")["family_avg_sales"].first().to_dict()
    store_family_avg = df.groupby(["store_nbr", "family"])["store_family_avg"].first().to_dict()
    dow_mean_map = df.groupby(["store_nbr", "family", "dayofweek"])["sales_dow_mean"].first().to_dict()
    family_share_map = df.groupby("family")["family_share"].first().to_dict()
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
            lag7 = seq[-7] if len(seq) >= 7 else np.nan
            lag14 = seq[-14] if len(seq) >= 14 else np.nan
            oil_price = oil_map.get(date, np.nan)
            oil_lag7 = oil_map.get(date - pd.Timedelta(days=7), np.nan)
            promo_history = recent_promo[(store, family)]
            trans_history = recent_trans[(store, family)]
            sf_avg = store_family_avg.get((store, family), 0) or 0
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
                    "dayofyear": date.dayofyear,
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
                    "onpromotion_lag7": promo_history[-7] if len(promo_history) >= 7 else np.nan,
                    "onpromotion_roll7": mean_last(promo_history, 7),
                    "promo_freq28": float(np.sum([1 for v in promo_history[-28:] if v > 0])),
                    "oil_price": oil_price,
                    "oil_lag7": oil_lag7,
                    "oil_roll7": float(np.nanmean([oil_map.get(date - pd.Timedelta(days=i), np.nan) for i in range(7)])),
                    "oil_pct_change_7": ((oil_price - oil_lag7) / oil_lag7) if oil_lag7 not in (None, np.nan) and oil_lag7 else 0,
                    "daily_transactions": trans_avg.get(store, 0),
                    "transactions_lag7": trans_history[-7] if len(trans_history) >= 7 else np.nan,
                    "transactions_roll7": mean_last(trans_history, 7),
                    "transactions_per_sales": trans_avg.get(store, 0) / (sf_avg + 1),
                    "store_avg_sales": store_avg.get(store, 0),
                    "family_avg_sales": family_avg.get(family, 0),
                    "store_family_avg": sf_avg,
                    "sales_dow_mean": dow_mean_map.get((store, family, date.dayofweek), np.nan),
                    "family_share": family_share_map.get(family, 0),
                    "sales_lag_1": seq[-1] if seq else np.nan,
                    "sales_lag_7": lag7,
                    "sales_lag_14": lag14,
                    "sales_lag_28": seq[-28] if len(seq) >= 28 else np.nan,
                    "sales_roll7": mean_last(seq, 7),
                    "sales_roll14": mean_last(seq, 14),
                    "sales_roll28": mean_last(seq, 28),
                    "sales_std7": std_last(seq, 7),
                    "sales_max7": max_last(seq, 7),
                    "sales_min7": min_last(seq, 7),
                    "sales_pct_change_7": ((lag7 - lag14) / lag14) if lag14 not in (None, np.nan) and lag14 else 0,
                    "sales_trend": (mean_last(seq, 7) / sf_avg) if sf_avg else 0,
                    "holiday_promo_interact": int(date in holiday_dates) * int(promo_map.get((date, store, family), 0) > 0),
                    "weekend_promo_interact": int(date.dayofweek >= 5) * int(promo_map.get((date, store, family), 0) > 0),
                }
            )
        x_test = backfill_test_features(pd.DataFrame(date_rows)[FEATURES])
        lgb_log = lgb_model.predict(x_test)
        xgb_log = xgb_model.predict(x_test)
        ens_log = best_w * lgb_log + (1 - best_w) * xgb_log
        preds = np.expm1(ens_log)
        for row, pred in zip(date_rows, preds):
            pred = max(0.0, float(pred))
            recent[(row["store_nbr"], row["family"])].append(pred)
            recent_promo[(row["store_nbr"], row["family"])].append(row["onpromotion"])
            recent_trans[(row["store_nbr"], row["family"])].append(row["daily_transactions"])
            rows.append({**row, "sales": pred})

    return pd.DataFrame(rows)[["date", "store_nbr", "family", "sales"]]


def save_submission(pred_df):
    print("6/10 生成提交文件 ...")
    test_info = pd.read_csv(TEST_CSV)
    test_info["date"] = pd.to_datetime(test_info["date"])
    merged = test_info.merge(pred_df, on=["date", "store_nbr", "family"], how="left")
    merged["sales"] = merged["sales"].fillna(0)
    submission = merged[["id", "sales"]].sort_values("id")
    submission.to_csv(OUT_DIR / "submission.csv", index=False)
    return submission


def style_fig(fig, title):
    fig.update_layout(
        template="plotly_white",
        title=dict(text=title, x=0.5, font=dict(size=18)),
        font=dict(family="Microsoft YaHei, sans-serif", size=13),
        margin=dict(l=70, r=50, t=70, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def make_plots(df, results, best_w, val_df, imp):
    print("7/10 生成可视化图表 ...")

    daily = df.groupby("date")["sales"].sum().reset_index()
    daily["rolling7"] = daily["sales"].rolling(7).mean()
    fig = px.line(
        daily, x="date", y=["sales", "rolling7"],
        labels={"value": "销量", "date": "日期", "variable": "类型"},
        color_discrete_map={"sales": "#2563EB", "rolling7": "#F97316"},
    )
    style_fig(fig, "每日总销量与 7 日移动平均")
    fig.update_layout(hovermode="x unified")
    fig.write_html(PLOT_DIR / "01_daily_sales_trend.html")

    seasonal = df.pivot_table(index="dayofweek", columns="month", values="sales", aggfunc="mean")
    seasonal.index = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    fig = px.imshow(seasonal, text_auto=".0f", color_continuous_scale="YlOrRd")
    style_fig(fig, "周几 x 月份 平均销量热力图")
    fig.update_layout(height=500)
    fig.write_html(PLOT_DIR / "02_seasonal_heatmap.html")

    df["ym"] = df["date"].dt.to_period("M").astype(str)
    monthly_year = df.groupby(["year", "ym"])["sales"].sum().reset_index()
    fig = px.line(
        monthly_year, x="ym", y="sales", color="year",
        labels={"ym": "月份", "sales": "总销量", "year": "年份"},
        color_discrete_sequence=px.colors.qualitative.Set1,
    )
    style_fig(fig, "各年份月度销量趋势")
    fig.write_html(PLOT_DIR / "03_monthly_yearly_trend.html")

    top_family = df.groupby("family")["sales"].mean().sort_values(ascending=False).head(10)
    fig = px.bar(
        top_family.reset_index(), x="sales", y="family", orientation="h",
        color="sales", color_continuous_scale="Blues",
    )
    style_fig(fig, "平均销量最高的 10 个品类")
    fig.update_layout(yaxis=dict(autorange="reversed"))
    fig.write_html(PLOT_DIR / "04_top_families.html")

    top_store = df.groupby("store_nbr")["sales"].sum().sort_values(ascending=False).head(10)
    fig = px.bar(
        top_store.reset_index(), x="sales", y="store_nbr", orientation="h",
        color="sales", color_continuous_scale="Oranges",
    )
    style_fig(fig, "总销量最高的 10 家门店")
    fig.update_layout(yaxis=dict(autorange="reversed"))
    fig.write_html(PLOT_DIR / "05_top_stores.html")

    holiday_avg = df.groupby("is_holiday")["sales"].mean().reset_index()
    holiday_avg["is_holiday"] = holiday_avg["is_holiday"].map({0: "普通日", 1: "节假日"})
    promo_avg = df.groupby(df["onpromotion"] > 0)["sales"].mean().reset_index()
    promo_avg.columns = ["promo", "sales"]
    promo_avg["promo"] = promo_avg["promo"].map({False: "无促销", True: "有促销"})
    fig = make_subplots(rows=1, cols=2, subplot_titles=("节假日 vs 普通日", "促销 vs 无促销"))
    fig.add_trace(go.Bar(x=holiday_avg["is_holiday"], y=holiday_avg["sales"], marker_color=["#94A3B8", "#EF4444"]), 1, 1)
    fig.add_trace(go.Bar(x=promo_avg["promo"], y=promo_avg["sales"], marker_color=["#94A3B8", "#F97316"]), 1, 2)
    style_fig(fig, "节假日与促销对销量的影响")
    fig.write_html(PLOT_DIR / "06_holiday_promo_effect.html")

    df["month_key"] = df["date"].dt.to_period("M").astype(str)
    monthly_biz = df.groupby("month_key").agg(total_sales=("sales", "sum"), avg_oil=("oil_price", "mean")).reset_index()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=monthly_biz["month_key"], y=monthly_biz["total_sales"], name="总销量", line=dict(color="#2563EB")), secondary_y=False)
    fig.add_trace(go.Scatter(x=monthly_biz["month_key"], y=monthly_biz["avg_oil"], name="平均油价", line=dict(color="#F97316", dash="dash")), secondary_y=True)
    fig.update_yaxes(title_text="总销量", secondary_y=False)
    fig.update_yaxes(title_text="平均油价", secondary_y=True)
    style_fig(fig, "月度销量与油价趋势")
    fig.write_html(PLOT_DIR / "07_oil_sales_trend.html")

    fig = px.bar(
        imp.head(20).iloc[::-1], x="importance", y="feature", orientation="h",
        color="importance", color_continuous_scale="Blues",
    )
    style_fig(fig, "LightGBM 特征重要性 Top 20")
    fig.update_layout(height=700)
    fig.write_html(PLOT_DIR / "08_feature_importance.html")

    results_df = pd.Series(results).reset_index().rename(columns={"index": "model", 0: "rmsle"})
    fig = px.bar(
        results_df, x="rmsle", y="model", orientation="h",
        color="rmsle", color_continuous_scale="YlOrRd_r",
    )
    style_fig(fig, "模型对比：RMSLE 越小越好")
    fig.update_layout(yaxis=dict(autorange="reversed"))
    fig.write_html(PLOT_DIR / "09_model_comparison.html")

    fig = px.scatter(
        val_df, x="actual", y="pred", opacity=0.2, color="family",
        labels={"actual": "实际销量", "pred": "预测销量"},
    )
    fig.add_shape(type="line", x0=0, y0=0, x1=val_df["actual"].max(), y1=val_df["actual"].max(), line=dict(color="black", dash="dash"))
    style_fig(fig, "验证集：实际销量 vs 加权融合预测")
    fig.write_html(PLOT_DIR / "10_validation_scatter.html")

    corr = pd.read_csv(OUT_DIR / "feature_sales_correlation.csv")
    corr = corr.reindex(corr["corr"].abs().sort_values(ascending=False).index).head(20)
    fig = px.bar(
        corr.iloc[::-1], x="corr", y="feature", orientation="h",
        color="corr", color_continuous_scale="RdBu_r",
    )
    style_fig(fig, "特征与销量相关性 Top 20")
    fig.update_layout(height=700)
    fig.write_html(PLOT_DIR / "11_feature_correlation.html")

    dashboard = make_subplots(
        rows=2, cols=2,
        subplot_titles=("每日总销量趋势", "平均销量 Top 品类", "节假日 vs 普通日", "模型对比 RMSLE"),
    )
    dashboard.add_trace(go.Scatter(x=daily["date"], y=daily["sales"], mode="lines", name="日销量", line=dict(color="#2563EB")), 1, 1)
    dashboard.add_trace(
        go.Bar(
            x=top_family.reset_index()["sales"],
            y=top_family.reset_index()["family"],
            orientation="h",
            name="Top 品类",
            marker_color="#3B82F6",
        ),
        1, 2,
    )
    dashboard.add_trace(go.Bar(x=holiday_avg["is_holiday"], y=holiday_avg["sales"], name="节假日影响", marker_color=["#94A3B8", "#EF4444"]), 2, 1)
    dashboard.add_trace(go.Bar(x=results_df["model"], y=results_df["rmsle"], name="模型 RMSLE", marker_color="#F97316"), 2, 2)
    style_fig(dashboard, "Store Sales 综合 Dashboard")
    dashboard.update_layout(height=800)
    dashboard.write_html(PLOT_DIR / "00_dashboard.html")

    print("  图表目录：", PLOT_DIR)


def write_business_report(df, results, best_w, val_df, imp):
    print("8/10 生成业务结论报告 ...")
    top_family = df.groupby("family")["sales"].mean().sort_values(ascending=False).head(5)
    top_store = df.groupby("store_nbr")["sales"].sum().sort_values(ascending=False).head(5)
    holiday = df.groupby("is_holiday")["sales"].mean()
    promo = df.groupby(df["onpromotion"] > 0)["sales"].mean()
    monthly = df.groupby(df["date"].dt.to_period("M"))["sales"].sum()
    peak_month = monthly.idxmax()

    val_df["log_err"] = (np.log1p(val_df["actual"]) - np.log1p(val_df["pred"])) ** 2
    per_family = val_df.groupby("family")["log_err"].mean().apply(np.sqrt).sort_values(ascending=False).head(5)
    per_family.to_csv(OUT_DIR / "per_family_errors.csv", header=["rmsle"])

    lines = [
        "# Store Sales 销量预测项目分析报告\n",
        "## 一、模型结论\n",
        f"- 最优融合权重：LightGBM {best_w:.2f} / XGBoost {1 - best_w:.2f}",
        f"- 基线（上一周销量）RMSLE：{results['baseline_last_week']:.4f}",
        f"- LightGBM RMSLE：{results['lightgbm']:.4f}",
        f"- XGBoost RMSLE：{results['xgboost']:.4f}",
        f"- 加权融合 RMSLE：{results['ensemble_weighted']:.4f}\n",
        "## 二、业务洞察\n",
        f"- 平均销量最高的品类：{', '.join(top_family.index[:5].astype(str))}",
        f"- 总销量最高的门店：{', '.join(top_store.index[:5].astype(str))}",
        f"- 节假日平均销量 {holiday.get(1, 0):.1f}，普通日平均销量 {holiday.get(0, 0):.1f}",
        f"- 促销日平均销量 {promo.get(True, 0):.1f}，非促销日平均销量 {promo.get(False, 0):.1f}",
        f"- 全年销量峰值月份：{peak_month}\n",
        "## 三、预测难点\n",
        f"- 误差最大的品类：{', '.join(per_family.index[:5].astype(str))}",
        "建议对高误差品类单独建模或增加外部数据。\n",
        "## 四、下一步建议\n",
        "1. 用 Optuna 自动调参",
        "2. 尝试 Tweedie 目标函数处理零销量",
        "3. 对高销量门店/品类分层建模",
        "4. 把 Streamlit 看板与报告一起上线，形成完整作品集",
    ]
    report = "\n".join(lines)
    report_path = OUT_DIR / "business_report.md"
    report_path.write_text(report, encoding="utf-8")
    return report_path, per_family


def main():
    df, holiday_dates = load_data()
    df = add_features(df, holiday_dates)

    numeric_cols = [c for c in FEATURES if c in df.columns]
    corr = df[numeric_cols + ["sales"]].corr(numeric_only=True)["sales"].drop("sales").sort_values()
    corr_df = corr.reset_index()
    corr_df.columns = ["feature", "corr"]
    corr_df.to_csv(OUT_DIR / "feature_sales_correlation.csv", index=False)

    results, best_w, fold_rows, val_df = run_cv(df)
    lgb_model, xgb_model, imp = train_final_models(df)
    pred_df = predict_test(df, lgb_model, xgb_model, holiday_dates, best_w)
    submission = save_submission(pred_df)
    make_plots(df, results, best_w, val_df, imp)
    report_path, per_family = write_business_report(df, results, best_w, val_df, imp)

    metrics = {
        "model_comparison": results,
        "best_weight_lgb": best_w,
        "best_weight_xgb": 1 - best_w,
        "folds": fold_rows,
        "final_test_rows": len(pred_df),
    }
    with open(OUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("9/10 保存指标与报告 ...")
    print("10/10 完成！")
    print("报告：", report_path)
    print("提交文件：", OUT_DIR / "submission.csv")
    print(submission.head(10))


if __name__ == "__main__":
    main()
