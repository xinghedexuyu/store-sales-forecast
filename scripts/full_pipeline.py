# -*- coding: utf-8 -*-
"""
Store Sales 全流程脚本

功能：数据读取 -> 特征工程 -> 建模 -> 验证 -> 未来16天预测 -> 评估 -> 导出
运行：python scripts/full_pipeline.py
"""

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

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
OUT_CSV = ROOT / "predictions.csv"

TRAIN_END = pd.Timestamp("2017-08-15")
VAL_START = pd.Timestamp("2017-08-01")
VAL_END = pd.Timestamp("2017-08-15")
TEST_START = pd.Timestamp("2017-08-16")
TEST_END = pd.Timestamp("2017-08-31")

FEATURES = [
    "store_code",
    "family_code",
    "year",
    "month",
    "day",
    "dayofweek",
    "weekofyear",
    "is_weekend",
    "is_holiday",
    "onpromotion",
    "oil_price",
    "daily_transactions",
    "sales_lag_7",
    "sales_lag_28",
    "sales_roll7",
    "sales_roll28",
]


def rmsle(y_true, y_pred):
    """计算 RMSLE，Kaggle 官方评估指标，值越小越好。"""
    y_pred = np.clip(y_pred, 0, None)
    return float(np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2)))


def load_data():
    """读取宽表，并重建节假日标记。"""
    print("1/6 读取数据 ...")
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


def add_features(df):
    """日期特征 + 滞后特征 + 滚动平均特征。"""
    print("2/6 特征工程 ...")
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["dayofweek"] = df["date"].dt.dayofweek
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["store_code"] = df["store_nbr"].astype("category").cat.codes
    df["family_code"] = df["family"].astype("category").cat.codes

    grouped = df.groupby(["store_nbr", "family"], sort=False)["sales"]
    df["sales_lag_7"] = grouped.shift(7)
    df["sales_lag_28"] = grouped.shift(28)
    df["sales_roll7"] = grouped.transform(
        lambda s: s.shift(1).rolling(7, min_periods=1).mean()
    )
    df["sales_roll28"] = grouped.transform(
        lambda s: s.shift(1).rolling(28, min_periods=1).mean()
    )
    return df


def train_and_validate(df):
    """用 8 月 1 日到 8 月 15 日做验证集，训练 LightGBM。"""
    print("3/6 训练模型并验证 ...")
    train_df = df[df["date"] < VAL_START].dropna(subset=FEATURES)
    val_df = df[(df["date"] >= VAL_START) & (df["date"] <= VAL_END)].dropna(subset=FEATURES)

    model = lgb.LGBMRegressor(
        n_estimators=150,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        verbose=-1,
    )
    model.fit(train_df[FEATURES], np.log1p(train_df["sales"]))

    val_pred = np.expm1(model.predict(val_df[FEATURES]))
    model_rmsle = rmsle(val_df["sales"], val_pred)
    baseline_rmsle = rmsle(val_df["sales"], val_df["sales_lag_7"].fillna(0))
    print(f"LightGBM 验证集 RMSLE：{model_rmsle:.4f}")
    print(f"上一周基线 RMSLE：{baseline_rmsle:.4f}")
    return model


def predict_test(df, model, holiday_dates):
    """逐日递归预测 2017-08-16 到 2017-08-31 的销量。"""
    print("4/6 预测未来 16 天 ...")
    store_codes = df.groupby("store_nbr")["store_code"].first().to_dict()
    family_codes = df.groupby("family")["family_code"].first().to_dict()
    trans_avg = df.groupby("store_nbr")["daily_transactions"].mean().to_dict()

    history = df[df["date"] <= TRAIN_END].sort_values(["store_nbr", "family", "date"])
    recent = {}
    for key, grp in history.groupby(["store_nbr", "family"], sort=False):
        recent[key] = list(grp["sales"].tail(28))

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
                    "year": date.year,
                    "month": date.month,
                    "day": date.day,
                    "dayofweek": date.dayofweek,
                    "weekofyear": date.isocalendar().week,
                    "is_weekend": int(date.dayofweek >= 5),
                    "is_holiday": int(date in holiday_dates),
                    "onpromotion": promo_map.get((date, store, family), 0),
                    "oil_price": oil_map.get(date, np.nan),
                    "daily_transactions": trans_avg.get(store, 0),
                    "sales_lag_7": seq[-7] if len(seq) >= 7 else np.nan,
                    "sales_lag_28": seq[-28] if len(seq) >= 28 else np.nan,
                    "sales_roll7": float(np.mean(seq[-7:])) if seq else np.nan,
                    "sales_roll28": float(np.mean(seq[-28:])) if seq else np.nan,
                }
            )
        x_test = pd.DataFrame(date_rows)[FEATURES].fillna(0)
        preds = np.expm1(model.predict(x_test))
        for row, pred in zip(date_rows, preds):
            pred = max(0.0, float(pred))
            recent[(row["store_nbr"], row["family"])].append(pred)
            rows.append({**row, "sales": pred})

    pred_df = pd.DataFrame(rows)
    return pred_df[["date", "store_nbr", "family", "sales"]]


def main():
    df, holiday_dates = load_data()
    df = add_features(df)
    model = train_and_validate(df)
    pred_df = predict_test(df, model, holiday_dates)

    print("5/6 导出预测结果 ...")
    pred_df.to_csv(OUT_CSV, index=False)
    print("6/6 完成！预测文件：", OUT_CSV)
    print(pred_df.head(10))


if __name__ == "__main__":
    main()
