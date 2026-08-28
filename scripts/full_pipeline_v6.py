# -*- coding: utf-8 -*-
"""
Store Sales 全流程 v6：直接多步预测

核心改进：
1. 不再递归回填预测值，直接为第 1~16 天分别训练
2. 特征全部来自预测起点已知信息 + 目标日期的日历/节假日/促销/油价
3. 与“上周同星期销量”基线做融合
4. 保留 Tweedie 目标函数

运行：python scripts/full_pipeline_v6.py
"""

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

import full_pipeline_v4 as v4

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs_v6"
PLOT_DIR = ROOT / "plots_v6"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)
v4.OUT_DIR = OUT_DIR
v4.PLOT_DIR = PLOT_DIR

FEATURES = v4.FEATURES
FOLDS = v4.FOLDS

DIRECT_FEATURES = [
    "store_code",
    "family_code",
    "store_type_code",
    "state_code",
    "cluster",
    "horizon",
    "target_year",
    "target_month",
    "target_day",
    "target_dayofweek",
    "target_is_weekend",
    "target_is_holiday",
    "target_payday",
    "target_days_to_holiday",
    "target_days_since_holiday",
    "onpromotion",
    "oil_price",
    "sales_dow_mean",
    "last_sales",
    "sales_lag_7",
    "sales_lag_14",
    "sales_lag_28",
    "sales_roll7",
    "sales_roll14",
    "sales_roll28",
    "sales_std7",
    "sales_max7",
    "sales_min7",
    "store_avg_sales",
    "family_avg_sales",
    "store_family_avg",
    "family_share",
]


def build_calendar(df, holiday_dates):
    dates = pd.date_range(
        df["date"].min(), df["date"].max() + pd.Timedelta(days=20), freq="D"
    )
    cal = pd.DataFrame({"date": dates})
    cal["c_year"] = cal["date"].dt.year
    cal["c_month"] = cal["date"].dt.month
    cal["c_day"] = cal["date"].dt.day
    cal["c_dayofweek"] = cal["date"].dt.dayofweek
    cal["c_is_weekend"] = (cal["c_dayofweek"] >= 5).astype(int)
    cal["c_is_holiday"] = cal["date"].isin(holiday_dates).astype(int)
    cal["c_payday"] = cal["date"].dt.day.isin([15, 30]).astype(int)

    hol = pd.DataFrame({"date": sorted(holiday_dates)}).sort_values("date")
    left = cal[["date"]].sort_values("date")
    nxt = pd.merge_asof(left, hol, on="date", direction="forward")
    prv = pd.merge_asof(left, hol, on="date", direction="backward")
    cal = cal.sort_values("date").reset_index(drop=True)
    cal["c_days_to_holiday"] = (nxt["date"].values - cal["date"].values) / pd.Timedelta(days=1)
    cal["c_days_since_holiday"] = (cal["date"].values - prv["date"].values) / pd.Timedelta(days=1)
    return cal


def build_lookup_tables(df):
    promo = df[["date", "store_nbr", "family", "onpromotion"]].drop_duplicates()
    test_info = pd.read_csv(v4.TEST_CSV)
    test_info["date"] = pd.to_datetime(test_info["date"])
    promo = pd.concat(
        [promo, test_info[["date", "store_nbr", "family", "onpromotion"]]]
    ).drop_duplicates(["date", "store_nbr", "family"], keep="last")
    promo = promo.rename(columns={"date": "target_date"})

    oil = pd.read_csv(v4.OIL_CSV)
    oil["date"] = pd.to_datetime(oil["date"])
    oil = oil.sort_values("date").ffill().rename(columns={"dcoilwtico": "oil_price"})
    oil = oil[["date", "oil_price"]]
    oil = oil.rename(columns={"date": "target_date"})

    dow = df[["store_nbr", "family", "dayofweek", "sales_dow_mean"]].drop_duplicates()
    actual = df[["date", "store_nbr", "family", "sales"]].rename(
        columns={"date": "target_date", "sales": "target_sales"}
    )
    return promo, oil, dow, actual


def build_direct_rows(df, cal, promo, oil, dow, actual, cutoff, sample_n, max_h=16):
    hist = df[
        (df["date"] <= cutoff) & (df["date"] >= cutoff - pd.Timedelta(days=730))
    ]
    origins = hist.sample(n=min(sample_n, len(hist)), random_state=42)
    target_cal = cal.rename(
        columns={
            "date": "target_date",
            "c_year": "target_year",
            "c_month": "target_month",
            "c_day": "target_day",
            "c_dayofweek": "target_dayofweek",
            "c_is_weekend": "target_is_weekend",
            "c_is_holiday": "target_is_holiday",
            "c_payday": "target_payday",
            "c_days_to_holiday": "target_days_to_holiday",
            "c_days_since_holiday": "target_days_since_holiday",
        }
    )
    frames = []
    for h in range(1, max_h + 1):
        t = origins.copy()
        t["target_date"] = t["date"] + pd.Timedelta(days=h)
        t["horizon"] = h
        t = t.merge(target_cal, on="target_date", how="left")
        t = t.drop(columns=["sales_dow_mean", "onpromotion", "oil_price"])
        t = t.merge(promo, on=["target_date", "store_nbr", "family"], how="left")
        t = t.merge(oil, on="target_date", how="left")
        t = t.merge(
            dow,
            left_on=["store_nbr", "family", "target_dayofweek"],
            right_on=["store_nbr", "family", "dayofweek"],
            how="left",
            suffixes=("", "_dow"),
        )
        t = t.merge(actual, on=["target_date", "store_nbr", "family"], how="left")
        t = t.dropna(subset=["target_sales"])
        t = t.rename(columns={"sales": "last_sales"})
        frames.append(t)
    return pd.concat(frames, ignore_index=True)


def build_pred_rows(df, cal, promo, oil, dow, cutoff, max_h=16):
    hist = df[df["date"] <= cutoff].sort_values(["store_nbr", "family", "date"])
    origins = hist.groupby(["store_nbr", "family"], sort=False).tail(1).copy()
    target_cal = cal.rename(
        columns={
            "date": "target_date",
            "c_year": "target_year",
            "c_month": "target_month",
            "c_day": "target_day",
            "c_dayofweek": "target_dayofweek",
            "c_is_weekend": "target_is_weekend",
            "c_is_holiday": "target_is_holiday",
            "c_payday": "target_payday",
            "c_days_to_holiday": "target_days_to_holiday",
            "c_days_since_holiday": "target_days_since_holiday",
        }
    )
    frames = []
    for h in range(1, max_h + 1):
        t = origins.copy()
        t["target_date"] = pd.Timestamp(cutoff) + pd.Timedelta(days=h)
        t["horizon"] = h
        t = t.merge(target_cal, on="target_date", how="left")
        t = t.drop(columns=["sales_dow_mean", "onpromotion", "oil_price"])
        t = t.merge(promo, on=["target_date", "store_nbr", "family"], how="left")
        t = t.merge(oil, on="target_date", how="left")
        t = t.merge(
            dow,
            left_on=["store_nbr", "family", "target_dayofweek"],
            right_on=["store_nbr", "family", "dayofweek"],
            how="left",
            suffixes=("", "_dow"),
        )
        t = t.rename(columns={"sales": "last_sales"})
        frames.append(t)
    return pd.concat(frames, ignore_index=True)


def train_direct(tr, params_lgb, params_xgb):
    lgb_model = lgb.LGBMRegressor(
        **params_lgb,
        objective="tweedie",
        tweedie_variance_power=1.1,
        random_state=42,
        verbose=-1,
        n_jobs=-1,
    )
    lgb_model.fit(tr[DIRECT_FEATURES], tr["target_sales"])

    xgb_model = xgb.XGBRegressor(
        **params_xgb,
        objective="reg:tweedie",
        tweedie_variance_power=1.1,
        random_state=42,
        verbosity=0,
        n_jobs=-1,
    )
    xgb_model.fit(tr[DIRECT_FEATURES], tr["target_sales"])
    return lgb_model, xgb_model


def run_cv(df, cal, promo, oil, dow, actual, params_lgb, params_xgb):
    print("4/10 3 段直接多步验证 ...")
    lgb_list, xgb_list, base_list, act_list = [], [], [], []
    fold_rows = []
    val_df = None

    for train_end, val_start, val_end in FOLDS:
        cutoff = pd.Timestamp(train_end)
        tr = build_direct_rows(
            df, cal, promo, oil, dow, actual, cutoff, sample_n=60000, max_h=16
        )
        lgb_model, xgb_model = train_direct(tr, params_lgb, params_xgb)
        win = build_pred_rows(df, cal, promo, oil, dow, cutoff, max_h=15)
        act_map = df[
            (df["date"] >= pd.Timestamp(val_start)) & (df["date"] <= pd.Timestamp(val_end))
        ].set_index(["date", "store_nbr", "family"])["sales"]
        win["actual"] = win.apply(
            lambda r: act_map.get((r["target_date"], r["store_nbr"], r["family"]), np.nan),
            axis=1,
        )
        win = win.dropna(subset=["actual"]).reset_index(drop=True)

        lgb_pred = np.clip(lgb_model.predict(win[DIRECT_FEATURES]), 0, None)
        xgb_pred = np.clip(xgb_model.predict(win[DIRECT_FEATURES]), 0, None)
        lgb_list.append(lgb_pred)
        xgb_list.append(xgb_pred)
        base_list.append(win["sales_lag_7"].values)
        act_list.append(win["actual"].values)

        fold_rows.append(
            {
                "fold": f"{train_end} -> {val_end}",
                "baseline": v4.rmsle(win["actual"], win["sales_lag_7"]),
                "lightgbm": v4.rmsle(win["actual"], lgb_pred),
                "xgboost": v4.rmsle(win["actual"], xgb_pred),
            }
        )
        if train_end == FOLDS[-1][0]:
            val_df = win.copy()
            val_df["pred_lgb"] = lgb_pred
            val_df["pred_xgb"] = xgb_pred

    all_y = np.concatenate(act_list)
    all_lgb = np.concatenate(lgb_list)
    all_xgb = np.concatenate(xgb_list)
    all_base = np.concatenate(base_list)

    best_w, best_score = 0.5, np.inf
    for w in np.arange(0, 1.001, 0.05):
        score = v4.rmsle(all_y, w * all_lgb + (1 - w) * all_xgb)
        if score < best_score:
            best_w, best_score = float(w), score
    model_pred = best_w * all_lgb + (1 - best_w) * all_xgb

    best_b, best_blend = 0.5, np.inf
    for b in np.arange(0, 1.001, 0.05):
        score = v4.rmsle(all_y, b * model_pred + (1 - b) * all_base)
        if score < best_blend:
            best_b, best_blend = float(b), score

    results = {
        "baseline_last_week": v4.rmsle(all_y, all_base),
        "lightgbm": v4.rmsle(all_y, all_lgb),
        "xgboost": v4.rmsle(all_y, all_xgb),
        "ensemble_weighted": best_score,
        "blend_with_baseline": best_blend,
    }
    val_df["pred"] = best_b * (
        best_w * val_df["pred_lgb"] + (1 - best_w) * val_df["pred_xgb"]
    ) + (1 - best_b) * val_df["sales_lag_7"]
    print("  折级结果：")
    for row in fold_rows:
        print("  ", row)
    print("  最优模型权重 w(LightGBM) =", round(best_w, 2))
    print("  最优基线融合比例 =", round(best_b, 2))
    return results, best_w, best_b, fold_rows, val_df


def train_final(df, cal, promo, oil, dow, actual, params_lgb, params_xgb):
    print("5/10 训练最终直接多步模型 ...")
    tr = build_direct_rows(
        df, cal, promo, oil, dow, actual, v4.TRAIN_END, sample_n=100000, max_h=16
    )
    lgb_model, xgb_model = train_direct(tr, params_lgb, params_xgb)
    imp = pd.DataFrame({"feature": DIRECT_FEATURES, "importance": lgb_model.feature_importances_})
    imp = imp.sort_values("importance", ascending=False).reset_index(drop=True)
    imp.to_csv(OUT_DIR / "feature_importance.csv", index=False)
    return lgb_model, xgb_model, imp


def save_submission(pred_df):
    print("6/10 生成提交文件 ...")
    test_info = pd.read_csv(v4.TEST_CSV)
    test_info["date"] = pd.to_datetime(test_info["date"])
    pred_df["date"] = pred_df["target_date"]
    merged = test_info.merge(
        pred_df[["date", "store_nbr", "family", "sales"]],
        on=["date", "store_nbr", "family"],
        how="left",
    )
    merged["sales"] = merged["sales"].fillna(0)
    submission = merged[["id", "sales"]].sort_values("id")
    submission.to_csv(OUT_DIR / "submission.csv", index=False)
    return submission


def main():
    df, holiday_dates = v4.load_data()
    df = v4.add_features(df, holiday_dates)
    cal = build_calendar(df, holiday_dates)
    promo, oil, dow, actual = build_lookup_tables(df)

    with open(ROOT / "outputs_v5" / "metrics.json", encoding="utf-8") as f:
        tuned = json.load(f)
    params_lgb = tuned["params_lgb"]
    params_xgb = tuned["params_xgb"]

    results, best_w, best_b, fold_rows, val_df = run_cv(
        df, cal, promo, oil, dow, actual, params_lgb, params_xgb
    )
    lgb_model, xgb_model, imp = train_final(
        df, cal, promo, oil, dow, actual, params_lgb, params_xgb
    )

    numeric_cols = [c for c in DIRECT_FEATURES if c in df.columns]
    corr = df[numeric_cols + ["sales"]].corr(numeric_only=True)["sales"].drop("sales").sort_values()
    corr_df = corr.reset_index()
    corr_df.columns = ["feature", "corr"]
    corr_df.to_csv(OUT_DIR / "feature_sales_correlation.csv", index=False)

    win = build_pred_rows(df, cal, promo, oil, dow, v4.TRAIN_END, max_h=16)
    lgb_pred = np.clip(lgb_model.predict(win[DIRECT_FEATURES]), 0, None)
    xgb_pred = np.clip(xgb_model.predict(win[DIRECT_FEATURES]), 0, None)
    model_pred = best_w * lgb_pred + (1 - best_w) * xgb_pred
    win["sales"] = best_b * model_pred + (1 - best_b) * win["sales_lag_7"]
    submission = save_submission(win)

    v4.make_plots(df, results, best_w, val_df, imp)
    report_path, _ = v4.write_business_report(df, results, best_w, val_df, imp)

    metrics = {
        "params_lgb": params_lgb,
        "params_xgb": params_xgb,
        "model_comparison": results,
        "best_weight_lgb": best_w,
        "best_weight_xgb": 1 - best_w,
        "best_baseline_blend": best_b,
        "folds": fold_rows,
        "final_test_rows": len(win),
    }
    with open(OUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("7/10 保存指标与报告 ...")
    print("8/10 完成！")
    print("报告：", report_path)
    print("提交文件：", OUT_DIR / "submission.csv")
    print(submission.head(10))


if __name__ == "__main__":
    main()
