# -*- coding: utf-8 -*-
"""
Store Sales 全流程 v2

相比 v1 的升级：
1. 更多特征：节假日距离、促销、油价、交易量、门店/品类历史均值
2. 多段时间序列交叉验证（3 段 walk-forward）
3. 输出特征重要性
4. 输出分品类误差分析
5. 输出 Kaggle 标准提交文件 submission.csv

运行：python scripts/full_pipeline_v2.py
"""

import json
import sys
from pathlib import Path

import lightgbm as lgb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = Path(r"C:\Users\xingh\Desktop\sales_analysis.csv")
TEST_CSV = ROOT / "data" / "test.csv"
SAMPLE_SUB_CSV = ROOT / "data" / "sample_submission.csv"
HOLIDAY_CSV = ROOT / "data" / "holidays_events.csv"
OIL_CSV = ROOT / "data" / "oil.csv"
OUT_DIR = ROOT / "outputs"
PLOT_DIR = ROOT / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_END = pd.Timestamp("2017-08-15")
TEST_START = pd.Timestamp("2017-08-16")
TEST_END = pd.Timestamp("2017-08-31")

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

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
    "is_payday",
    "is_holiday",
    "days_to_holiday",
    "days_since_holiday",
    "onpromotion",
    "onpromotion_roll7",
    "oil_price",
    "oil_roll7",
    "daily_transactions",
    "transactions_roll7",
    "store_avg_sales",
    "family_avg_sales",
    "sales_lag_1",
    "sales_lag_7",
    "sales_lag_28",
    "sales_roll7",
    "sales_roll28",
    "sales_std7",
]


def rmsle(y_true, y_pred):
    """RMSLE：值越小越好，是 Kaggle 官方评估指标。"""
    y_pred = np.clip(y_pred, 0, None)
    return float(np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2)))


def load_data():
    """读取宽表并重建节假日标记。"""
    print("1/8 读取数据 ...")
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
    """从多个维度构造特征：时间、门店、品类、促销、油价、交易量、历史销量。"""
    print("2/8 特征工程 ...")
    df = df.sort_values("date").reset_index(drop=True)
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["dayofweek"] = df["date"].dt.dayofweek
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["is_month_end"] = (df["day"] >= 28).astype(int)
    df["is_payday"] = df["day"].isin([15, 30]).astype(int)

    df["store_code"] = df["store_nbr"].astype("category").cat.codes
    df["family_code"] = df["family"].astype("category").cat.codes
    df["store_type_code"] = df["store_type"].astype("category").cat.codes
    df["state_code"] = df["state"].astype("category").cat.codes

    # 节假日距离特征：距离最近节假日还有几天/已经过去几天
    hol = pd.DataFrame({"date": sorted(holiday_dates)}).drop_duplicates().sort_values("date")
    left = df[["date"]].sort_values("date")
    next_hol = pd.merge_asof(left, hol, on="date", direction="forward")
    prev_hol = pd.merge_asof(left, hol, on="date", direction="backward")
    df["days_to_holiday"] = (
        (next_hol["date"].values - df["date"].values) / pd.Timedelta(days=1)
    )
    df["days_since_holiday"] = (
        (df["date"].values - prev_hol["date"].values) / pd.Timedelta(days=1)
    )

    # 历史平均销量（只用训练期计算，避免数据泄露）
    agg_base = df[df["date"] <= pd.Timestamp("2017-07-31")]
    store_avg = agg_base.groupby("store_nbr")["sales"].mean().rename("store_avg_sales")
    family_avg = agg_base.groupby("family")["sales"].mean().rename("family_avg_sales")
    df = df.merge(store_avg, on="store_nbr", how="left").merge(family_avg, on="family", how="left")

    # 时间序列特征：滞后、滚动均值、滚动标准差
    grouped = df.groupby(["store_nbr", "family"], sort=False)["sales"]
    df["sales_lag_1"] = grouped.shift(1)
    df["sales_lag_7"] = grouped.shift(7)
    df["sales_lag_28"] = grouped.shift(28)
    df["sales_roll7"] = grouped.transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
    df["sales_roll28"] = grouped.transform(lambda s: s.shift(1).rolling(28, min_periods=1).mean())
    df["sales_std7"] = grouped.transform(lambda s: s.shift(1).rolling(7, min_periods=1).std())

    # 促销、油价、交易量的滚动特征
    promo_grouped = df.groupby(["store_nbr", "family"], sort=False)["onpromotion"]
    df["onpromotion_roll7"] = promo_grouped.transform(
        lambda s: s.shift(1).rolling(7, min_periods=1).mean()
    )
    df["oil_roll7"] = df.groupby("store_nbr", sort=False)["oil_price"].transform(
        lambda s: s.shift(1).rolling(7, min_periods=1).mean()
    )
    df["transactions_roll7"] = df.groupby("store_nbr", sort=False)["daily_transactions"].transform(
        lambda s: s.shift(1).rolling(7, min_periods=1).mean()
    )
    return df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)


def walk_forward_cv(df):
    """3 段滚动验证：6月底/7月中/7月底分别训练，验证未来15天。"""
    print("3/8 多段交叉验证 ...")
    folds = [
        ("2017-06-30", "2017-07-01", "2017-07-15"),
        ("2017-07-15", "2017-07-16", "2017-07-31"),
        ("2017-07-31", "2017-08-01", "2017-08-15"),
    ]
    cv_rmsles = []
    for train_end, val_start, val_end in folds:
        tr = df[df["date"] <= pd.Timestamp(train_end)].dropna(subset=FEATURES)
        va = df[(df["date"] >= pd.Timestamp(val_start)) & (df["date"] <= pd.Timestamp(val_end))].dropna(subset=FEATURES)
        model = lgb.LGBMRegressor(
            n_estimators=150, learning_rate=0.05, num_leaves=31,
            random_state=42, verbose=-1,
        )
        model.fit(tr[FEATURES], np.log1p(tr["sales"]))
        pred = np.expm1(model.predict(va[FEATURES]))
        score = rmsle(va["sales"], pred)
        cv_rmsles.append(score)
        print(f"  验证段 {train_end} -> {val_end}：RMSLE {score:.4f}")
    return cv_rmsles


def train_final_model(df):
    """用全部训练期数据训练最终模型。"""
    print("4/8 训练最终模型 ...")
    tr = df[df["date"] <= TRAIN_END].dropna(subset=FEATURES)
    model = lgb.LGBMRegressor(
        n_estimators=200, learning_rate=0.05, num_leaves=31,
        random_state=42, verbose=-1,
    )
    model.fit(tr[FEATURES], np.log1p(tr["sales"]))

    # 用最后一段验证集评估最终模型
    va = df[(df["date"] >= pd.Timestamp("2017-08-01")) & (df["date"] <= pd.Timestamp("2017-08-15"))].dropna(subset=FEATURES)
    pred = np.expm1(model.predict(va[FEATURES]))
    final_rmsle = rmsle(va["sales"], pred)
    baseline_rmsle = rmsle(va["sales"], va["sales_lag_7"].fillna(0))
    print(f"  最终模型验证集 RMSLE：{final_rmsle:.4f}")
    print(f"  上一周基线 RMSLE：{baseline_rmsle:.4f}")

    # 分品类误差：看哪些品类最难预测
    err = pd.DataFrame({"family": va["family"], "actual": va["sales"], "pred": pred})
    per_family = []
    for family, g in err.groupby("family"):
        per_family.append({"family": family, "rmsle": rmsle(g["actual"], g["pred"]), "rows": len(g)})
    per_family_df = pd.DataFrame(per_family).sort_values("rmsle", ascending=False)
    per_family_df.to_csv(OUT_DIR / "per_family_errors.csv", index=False)
    print("  分品类误差已保存：", OUT_DIR / "per_family_errors.csv")
    return model, final_rmsle, baseline_rmsle


def save_feature_importance(model):
    """输出特征重要性并画图。"""
    print("5/8 保存特征重要性 ...")
    imp = pd.DataFrame({"feature": FEATURES, "importance": model.feature_importances_})
    imp = imp.sort_values("importance", ascending=False).reset_index(drop=True)
    imp.to_csv(OUT_DIR / "feature_importance.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 10))
    imp.head(20).iloc[::-1].plot.barh(x="feature", y="importance", ax=ax, color="#2563EB")
    ax.set_title("LightGBM 特征重要性 Top 20")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "05_feature_importance.png", dpi=120)
    plt.close(fig)


def predict_test(df, model, holiday_dates):
    """逐日递归预测未来 16 天。"""
    print("6/8 预测未来 16 天 ...")
    store_codes = df.groupby("store_nbr")["store_code"].first().to_dict()
    family_codes = df.groupby("family")["family_code"].first().to_dict()
    type_codes = df.groupby("store_nbr")["store_type_code"].first().to_dict()
    state_codes = df.groupby("store_nbr")["state_code"].first().to_dict()
    cluster_map = df.groupby("store_nbr")["cluster"].first().to_dict()
    store_avg = df.groupby("store_nbr")["store_avg_sales"].first().to_dict()
    family_avg = df.groupby("family")["family_avg_sales"].first().to_dict()
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
        date_int = np.datetime64(date, "D")
        idx = np.searchsorted(hol_dates, date_int)
        if idx < len(hol_dates):
            return int((hol_dates[idx] - date_int).astype(int))
        return np.nan

    def days_since_holiday(date):
        date_int = np.datetime64(date, "D")
        idx = np.searchsorted(hol_dates, date_int)
        if idx > 0:
            return int((date_int - hol_dates[idx - 1]).astype(int))
        return np.nan

    def oil_roll7(date):
        vals = [
            oil_map.get(date - pd.Timedelta(days=i), np.nan)
            for i in range(7)
        ]
        return float(np.nanmean(vals)) if not np.isnan(np.nanmean(vals)) else np.nan

    rows = []
    for date in pd.date_range(TEST_START, TEST_END, freq="D"):
        date_rows = []
        for (store, family), seq in recent.items():
            def roll(arr, n):
                return float(np.mean(arr[-n:])) if arr else np.nan

            def std(arr, n):
                return float(np.std(arr[-n:])) if len(arr) >= n else np.nan

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
                    "is_payday": int(date.day in (15, 30)),
                    "is_holiday": int(date in holiday_dates),
                    "days_to_holiday": days_to_holiday(date),
                    "days_since_holiday": days_since_holiday(date),
                    "onpromotion": promo_map.get((date, store, family), 0),
                    "onpromotion_roll7": roll(recent_promo[(store, family)], 7),
                    "oil_price": oil_map.get(date, np.nan),
                    "oil_roll7": oil_roll7(date),
                    "daily_transactions": trans_avg.get(store, 0),
                    "transactions_roll7": roll(recent_trans[(store, family)], 7),
                    "store_avg_sales": store_avg.get(store, 0),
                    "family_avg_sales": family_avg.get(family, 0),
                    "sales_lag_1": seq[-1] if seq else np.nan,
                    "sales_lag_7": seq[-7] if len(seq) >= 7 else np.nan,
                    "sales_lag_28": seq[-28] if len(seq) >= 28 else np.nan,
                    "sales_roll7": roll(seq, 7),
                    "sales_roll28": roll(seq, 28),
                    "sales_std7": std(seq, 7),
                }
            )
        x_test = pd.DataFrame(date_rows)[FEATURES].fillna(0)
        preds = np.expm1(model.predict(x_test))
        for row, pred in zip(date_rows, preds):
            pred = max(0.0, float(pred))
            recent[(row["store_nbr"], row["family"])].append(pred)
            recent_promo[(row["store_nbr"], row["family"])].append(row["onpromotion"])
            recent_trans[(row["store_nbr"], row["family"])].append(row["daily_transactions"])
            rows.append({**row, "sales": pred})

    return pd.DataFrame(rows)[["date", "store_nbr", "family", "sales"]]


def save_submission(pred_df):
    """按 sample_submission 的 id 顺序导出 Kaggle 提交文件。"""
    print("7/8 生成提交文件 ...")
    test_info = pd.read_csv(TEST_CSV)
    test_info["date"] = pd.to_datetime(test_info["date"])
    merged = test_info.merge(
        pred_df, on=["date", "store_nbr", "family"], how="left"
    )
    merged["sales"] = merged["sales"].fillna(0)
    submission = merged[["id", "sales"]].sort_values("id")
    submission.to_csv(OUT_DIR / "submission.csv", index=False)
    return submission


def main():
    df, holiday_dates = load_data()
    df = add_features(df, holiday_dates)
    cv_rmsles = walk_forward_cv(df)
    model, final_rmsle, baseline_rmsle = train_final_model(df)
    save_feature_importance(model)
    pred_df = predict_test(df, model, holiday_dates)
    submission = save_submission(pred_df)

    metrics = {
        "cv_rmsles": cv_rmsles,
        "cv_mean_rmsle": float(np.mean(cv_rmsles)),
        "final_model_rmsle": final_rmsle,
        "baseline_rmsle": baseline_rmsle,
    }
    with open(OUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("8/8 完成！")
    print("提交文件：", OUT_DIR / "submission.csv")
    print("指标文件：", OUT_DIR / "metrics.json")
    print(submission.head(10))


if __name__ == "__main__":
    main()
