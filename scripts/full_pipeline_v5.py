# -*- coding: utf-8 -*-
"""
Store Sales 全流程 v5

相比 v4 的升级：
1. 多步递归验证：验证集也按“从某天开始连续预测 15 天”评估，和真实提交一致
2. Optuna 自动调参：如果安装了 optuna 就使用，否则自动退回内置随机搜索
3. Tweedie 目标函数：处理零销量和离散销售数据
4. 高销量门店/品类分层建模：Top 10 组合单独训练模型
5. Streamlit 看板 + 业务报告：形成可上线作品集

运行：python scripts/full_pipeline_v5.py
"""

import json
import random
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

import full_pipeline_v4 as v4

try:
    import optuna

    OPTUNA_AVAILABLE = True
except Exception:
    optuna = None
    OPTUNA_AVAILABLE = False

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs_v5"
PLOT_DIR = ROOT / "plots_v5"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)
v4.OUT_DIR = OUT_DIR
v4.PLOT_DIR = PLOT_DIR

FEATURES = v4.FEATURES
FOLDS = v4.FOLDS


def tune_params(df):
    """自动调参：有 Optuna 用 Optuna，否则用内置随机搜索。"""
    print("3/11 自动调参 ...")
    sample = (
        df[df["date"] < pd.Timestamp("2017-08-01")]
        .dropna(subset=FEATURES)
        .sample(n=200000, random_state=42)
    )
    tr = sample.sample(n=160000, random_state=42)
    va = sample.loc[~sample.index.isin(tr.index)]
    X_tr, y_tr = tr[FEATURES], tr["sales"].values
    X_va, y_va = va[FEATURES], va["sales"].values

    def eval_lgb(params):
        model = lgb.LGBMRegressor(
            **params,
            objective="tweedie",
            tweedie_variance_power=1.1,
            random_state=42,
            verbose=-1,
            n_jobs=-1,
        )
        model.fit(X_tr, y_tr)
        return v4.rmsle(y_va, model.predict(X_va))

    def eval_xgb(params):
        model = xgb.XGBRegressor(
            **params,
            objective="reg:tweedie",
            tweedie_variance_power=1.1,
            random_state=42,
            verbosity=0,
            n_jobs=-1,
        )
        model.fit(X_tr, y_tr)
        return v4.rmsle(y_va, model.predict(X_va))

    if OPTUNA_AVAILABLE:
        print("  使用 Optuna ...")

        def lgb_objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 250),
                "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.15, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 16, 96),
                "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
                "subsample": trial.suggest_float("subsample", 0.7, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            }
            return eval_lgb(params)

        def xgb_objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 250),
                "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.15, log=True),
                "max_depth": trial.suggest_int("max_depth", 4, 9),
                "min_child_weight": trial.suggest_int("min_child_weight", 5, 40),
                "subsample": trial.suggest_float("subsample", 0.7, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            }
            return eval_xgb(params)

        study_lgb = optuna.create_study(direction="minimize")
        study_lgb.optimize(lgb_objective, n_trials=6)
        study_xgb = optuna.create_study(direction="minimize")
        study_xgb.optimize(xgb_objective, n_trials=6)
        params_lgb = study_lgb.best_params
        params_xgb = study_xgb.best_params
    else:
        print("  Optuna 未安装，使用内置随机搜索（8 轮） ...")
        rng = random.Random(42)
        best_lgb, best_xgb = None, None
        best_lgb_score, best_xgb_score = np.inf, np.inf
        for _ in range(8):
            lgb_params = {
                "n_estimators": rng.randint(100, 250),
                "learning_rate": round(10 ** rng.uniform(-1.5, -0.8), 4),
                "num_leaves": rng.randint(16, 96),
                "min_child_samples": rng.randint(20, 100),
                "subsample": round(rng.uniform(0.7, 1.0), 2),
                "colsample_bytree": round(rng.uniform(0.6, 1.0), 2),
                "reg_alpha": round(10 ** rng.uniform(-3, 1), 4),
                "reg_lambda": round(10 ** rng.uniform(-3, 1), 4),
            }
            xgb_params = {
                "n_estimators": rng.randint(100, 250),
                "learning_rate": round(10 ** rng.uniform(-1.5, -0.8), 4),
                "max_depth": rng.randint(4, 9),
                "min_child_weight": rng.randint(5, 40),
                "subsample": round(rng.uniform(0.7, 1.0), 2),
                "colsample_bytree": round(rng.uniform(0.6, 1.0), 2),
                "reg_alpha": round(10 ** rng.uniform(-3, 1), 4),
                "reg_lambda": round(10 ** rng.uniform(-3, 1), 4),
            }
            s_lgb = eval_lgb(lgb_params)
            s_xgb = eval_xgb(xgb_params)
            if s_lgb < best_lgb_score:
                best_lgb_score, best_lgb = s_lgb, lgb_params
            if s_xgb < best_xgb_score:
                best_xgb_score, best_xgb = s_xgb, xgb_params
        params_lgb, params_xgb = best_lgb, best_xgb

    print("  LightGBM 最优参数：", params_lgb)
    print("  XGBoost 最优参数：", params_xgb)
    return params_lgb, params_xgb


def train_pair(tr, params_lgb, params_xgb):
    """训练 LightGBM 和 XGBoost，均使用 Tweedie 目标函数。"""
    lgb_model = lgb.LGBMRegressor(
        **params_lgb,
        objective="tweedie",
        tweedie_variance_power=1.1,
        random_state=42,
        verbose=-1,
        n_jobs=-1,
    )
    lgb_model.fit(tr[FEATURES], tr["sales"])

    xgb_model = xgb.XGBRegressor(
        **params_xgb,
        objective="reg:tweedie",
        tweedie_variance_power=1.1,
        random_state=42,
        verbosity=0,
        n_jobs=-1,
    )
    xgb_model.fit(tr[FEATURES], tr["sales"])
    return lgb_model, xgb_model


def train_segment_models(df, history_cutoff, top_n=10):
    """对 Top 10 高销量门店+品类组合单独建模。"""
    history = df[df["date"] <= pd.Timestamp(history_cutoff)]
    totals = history.groupby(["store_nbr", "family"])["sales"].sum().nlargest(top_n)
    models = {}
    for combo in totals.index:
        sub = history[
            (history["store_nbr"] == combo[0]) & (history["family"] == combo[1])
        ].dropna(subset=FEATURES)
        if len(sub) < 100:
            continue
        model = xgb.XGBRegressor(
            n_estimators=80,
            learning_rate=0.05,
            max_depth=5,
            objective="reg:tweedie",
            tweedie_variance_power=1.1,
            random_state=42,
            verbosity=0,
            n_jobs=-1,
        )
        model.fit(sub[FEATURES], sub["sales"])
        models[combo] = model
    return models


def predict_window(
    df,
    lgb_model,
    xgb_model,
    holiday_dates,
    history_cutoff,
    start,
    end,
    segment_models=None,
    recur_w=0.5,
):
    """从 history_cutoff 开始，逐日递归预测 start 到 end，不用未来真实值。"""
    segment_models = segment_models or {}
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

    history = df[df["date"] <= pd.Timestamp(history_cutoff)].sort_values(["store_nbr", "family", "date"])
    recent = {}
    recent_promo = {}
    recent_trans = {}
    baseline_map = {}
    for key, grp in history.groupby(["store_nbr", "family"], sort=False):
        recent[key] = list(grp["sales"].tail(28))
        recent_promo[key] = list(grp["onpromotion"].tail(28))
        recent_trans[key] = list(grp["daily_transactions"].tail(28))
        baseline_map[key] = float(grp["sales"].iloc[-1]) if len(grp) else 0.0

    promo_map = {
        (r.date, r.store_nbr, r.family): r.onpromotion
        for r in df[["date", "store_nbr", "family", "onpromotion"]].itertuples(index=False)
    }
    test_info = pd.read_csv(v4.TEST_CSV)
    test_info["date"] = pd.to_datetime(test_info["date"])
    promo_map.update(
        {
            (r.date, r.store_nbr, r.family): r.onpromotion
            for r in test_info.itertuples(index=False)
        }
    )

    oil = pd.read_csv(v4.OIL_CSV)
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
    for date in pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="D"):
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
                    "promo_freq28": float(np.sum([1 for x in promo_history[-28:] if x > 0])),
                    "oil_price": oil_price,
                    "oil_lag7": oil_lag7,
                    "oil_roll7": float(np.nanmean([oil_map.get(date - pd.Timedelta(days=i), np.nan) for i in range(7)])),
                    "oil_pct_change_7": ((oil_price - oil_lag7) / oil_lag7) if pd.notna(oil_lag7) and oil_lag7 else 0,
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
                    "sales_pct_change_7": ((lag7 - lag14) / lag14) if pd.notna(lag14) and lag14 else 0,
                    "sales_trend": (mean_last(seq, 7) / sf_avg) if sf_avg else 0,
                    "holiday_promo_interact": int(date in holiday_dates) * int(promo_map.get((date, store, family), 0) > 0),
                    "weekend_promo_interact": int(date.dayofweek >= 5) * int(promo_map.get((date, store, family), 0) > 0),
                }
            )
        x_test = v4.backfill_test_features(pd.DataFrame(date_rows)[FEATURES])
        lgb_pred = np.clip(lgb_model.predict(x_test), 0, None)
        xgb_pred = np.clip(xgb_model.predict(x_test), 0, None)

        for i, (row, lp, xp) in enumerate(zip(date_rows, lgb_pred, xgb_pred)):
            key = (row["store_nbr"], row["family"])
            if key in segment_models:
                seg_pred = max(0.0, float(segment_models[key].predict(x_test.iloc[[i]])[0]))
                lp = seg_pred
                xp = seg_pred
            else:
                lp = float(lp)
                xp = float(xp)
            ens = recur_w * lp + (1 - recur_w) * xp
            recent[key].append(max(0.0, ens))
            recent_promo[key].append(row["onpromotion"])
            recent_trans[key].append(row["daily_transactions"])
            rows.append(
                {
                    "date": row["date"],
                    "store_nbr": row["store_nbr"],
                    "family": row["family"],
                    "pred_lgb": lp,
                    "pred_xgb": xp,
                    "baseline": baseline_map.get(key, 0.0),
                }
            )

    return pd.DataFrame(rows)


def run_recursive_cv(df, params_lgb, params_xgb, holiday_dates):
    """3 段多步递归验证，和真实 16 天预测逻辑一致。"""
    print("4/11 多步递归验证 ...")
    lgb_list, xgb_list, act_list, base_list = [], [], [], []
    fold_rows = []
    val_df = None

    for train_end, val_start, val_end in FOLDS:
        tr = df[df["date"] <= pd.Timestamp(train_end)].dropna(subset=FEATURES)
        lgb_model, xgb_model = train_pair(tr, params_lgb, params_xgb)
        segment_models = train_segment_models(df, train_end)
        win = predict_window(
            df,
            lgb_model,
            xgb_model,
            holiday_dates,
            history_cutoff=train_end,
            start=val_start,
            end=val_end,
            segment_models=segment_models,
            recur_w=0.5,
        )
        actual_map = df[
            (df["date"] >= pd.Timestamp(val_start)) & (df["date"] <= pd.Timestamp(val_end))
        ].set_index(["date", "store_nbr", "family"])["sales"]
        win["actual"] = win.apply(
            lambda r: actual_map.get((r["date"], r["store_nbr"], r["family"]), np.nan), axis=1
        )
        win = win.dropna(subset=["actual"]).reset_index(drop=True)

        lgb_list.append(win["pred_lgb"])
        xgb_list.append(win["pred_xgb"])
        act_list.append(win["actual"])
        base_list.append(win["baseline"])
        fold_rows.append(
            {
                "fold": f"{train_end} -> {val_end}",
                "baseline": v4.rmsle(win["actual"], win["baseline"]),
                "lightgbm": v4.rmsle(win["actual"], win["pred_lgb"]),
                "xgboost": v4.rmsle(win["actual"], win["pred_xgb"]),
            }
        )
        if train_end == FOLDS[-1][0]:
            val_df = win.copy()
            val_df["pred"] = (val_df["pred_lgb"] + val_df["pred_xgb"]) / 2

    all_y = np.concatenate(act_list)
    all_lgb = np.concatenate(lgb_list)
    all_xgb = np.concatenate(xgb_list)
    all_base = np.concatenate(base_list)

    best_w, best_score = 0.5, np.inf
    for w in np.arange(0, 1.001, 0.05):
        score = v4.rmsle(all_y, w * all_lgb + (1 - w) * all_xgb)
        if score < best_score:
            best_w, best_score = float(w), score
    val_df["pred"] = best_w * val_df["pred_lgb"] + (1 - best_w) * val_df["pred_xgb"]

    results = {
        "baseline_last_week": v4.rmsle(all_y, all_base),
        "lightgbm": v4.rmsle(all_y, all_lgb),
        "xgboost": v4.rmsle(all_y, all_xgb),
        "ensemble_weighted": best_score,
    }
    print("  折级结果：")
    for row in fold_rows:
        print("  ", row)
    print("  最优权重 w(LightGBM) =", round(best_w, 2))
    return results, best_w, fold_rows, val_df


def train_final(df, params_lgb, params_xgb, holiday_dates):
    """训练最终模型 + Top 组合分层模型 + 特征重要性。"""
    print("5/11 训练最终模型 ...")
    tr = df[df["date"] <= v4.TRAIN_END].dropna(subset=FEATURES)
    lgb_model, xgb_model = train_pair(tr, params_lgb, params_xgb)
    segment_models = train_segment_models(df, v4.TRAIN_END)
    imp = pd.DataFrame({"feature": FEATURES, "importance": lgb_model.feature_importances_})
    imp = imp.sort_values("importance", ascending=False).reset_index(drop=True)
    imp.to_csv(OUT_DIR / "feature_importance.csv", index=False)
    return lgb_model, xgb_model, segment_models, imp


def save_submission(pred_df):
    print("6/11 生成提交文件 ...")
    test_info = pd.read_csv(v4.TEST_CSV)
    test_info["date"] = pd.to_datetime(test_info["date"])
    merged = test_info.merge(pred_df, on=["date", "store_nbr", "family"], how="left")
    merged["sales"] = merged["sales"].fillna(0)
    submission = merged[["id", "sales"]].sort_values("id")
    submission.to_csv(OUT_DIR / "submission.csv", index=False)
    return submission


def main():
    df, holiday_dates = v4.load_data()
    df = v4.add_features(df, holiday_dates)

    numeric_cols = [c for c in FEATURES if c in df.columns]
    corr = df[numeric_cols + ["sales"]].corr(numeric_only=True)["sales"].drop("sales").sort_values()
    corr_df = corr.reset_index()
    corr_df.columns = ["feature", "corr"]
    corr_df.to_csv(OUT_DIR / "feature_sales_correlation.csv", index=False)

    params_lgb, params_xgb = tune_params(df)
    results, best_w, fold_rows, val_df = run_recursive_cv(df, params_lgb, params_xgb, holiday_dates)
    lgb_model, xgb_model, segment_models, imp = train_final(df, params_lgb, params_xgb, holiday_dates)

    win = predict_window(
        df,
        lgb_model,
        xgb_model,
        holiday_dates,
        history_cutoff=v4.TRAIN_END,
        start=v4.TEST_START,
        end=v4.TEST_END,
        segment_models=segment_models,
        recur_w=best_w,
    )
    pred_df = win[["date", "store_nbr", "family"]].copy()
    pred_df["sales"] = best_w * win["pred_lgb"] + (1 - best_w) * win["pred_xgb"]
    submission = save_submission(pred_df)

    v4.make_plots(df, results, best_w, val_df, imp)
    report_path, _ = v4.write_business_report(df, results, best_w, val_df, imp)

    metrics = {
        "optuna_available": OPTUNA_AVAILABLE,
        "params_lgb": params_lgb,
        "params_xgb": params_xgb,
        "model_comparison": results,
        "best_weight_lgb": best_w,
        "best_weight_xgb": 1 - best_w,
        "folds": fold_rows,
        "final_test_rows": len(pred_df),
    }
    with open(OUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("7/11 保存指标与报告 ...")
    print("8/11 完成！")
    print("报告：", report_path)
    print("提交文件：", OUT_DIR / "submission.csv")
    print(submission.head(10))


if __name__ == "__main__":
    main()
