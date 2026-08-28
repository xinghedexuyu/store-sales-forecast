# -*- coding: utf-8 -*-
"""
生成业务分析报告：先列数据，再做分析。
运行：python scripts/generate_business_report.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs_v6"
CSV_PATH = Path(r"C:\Users\xingh\Desktop\sales_analysis.csv")

print("读取数据 ...")
df = pd.read_csv(CSV_PATH)
df["date"] = pd.to_datetime(df["date"])

holidays = pd.read_csv(ROOT / "data" / "holidays_events.csv")
holidays["date"] = pd.to_datetime(holidays["date"])
holidays["transferred"] = holidays["transferred"].map(
    lambda v: 1 if str(v).lower() in ("1", "true", "1.0") else 0
)
holiday_dates = set(holidays.loc[holidays["transferred"] == 0, "date"])
df["is_holiday"] = df["date"].isin(holiday_dates).astype(int)

with open(OUT_DIR / "metrics.json", encoding="utf-8") as f:
    metrics = json.load(f)

comparison = metrics["model_comparison"]
per_family = pd.read_csv(OUT_DIR / "per_family_errors.csv").sort_values("rmsle", ascending=False)
imp = pd.read_csv(OUT_DIR / "feature_importance.csv").head(10)

top_family = df.groupby("family")["sales"].mean().sort_values(ascending=False).head(5)
top_store = df.groupby("store_nbr")["sales"].sum().sort_values(ascending=False).head(5)
holiday = df.groupby("is_holiday")["sales"].mean()
promo = df.groupby(df["onpromotion"] > 0)["sales"].mean()
monthly = df.groupby(df["date"].dt.to_period("M"))["sales"].sum()

lines = [
    "# Store Sales 销量预测项目业务分析报告\n",
    "## 一、数据概览\n",
    "| 指标 | 数值 |",
    "| --- | --- |",
    f"| 销售记录 | {len(df):,} 行 |",
    f"| 日期范围 | {df['date'].min():%Y-%m-%d} 至 {df['date'].max():%Y-%m-%d} |",
    f"| 门店数 | {df['store_nbr'].nunique()} 家 |",
    f"| 商品品类 | {df['family'].nunique()} 个 |",
    f"| 预测期 | 2017-08-16 至 2017-08-31，共 16 天 |\n",
    "## 二、销量结构数据\n",
    "### 平均销量 Top 5 品类\n",
    "| 品类 | 平均销量 |",
    "| --- | --- |",
]
for family, sales in top_family.items():
    lines.append(f"| {family} | {sales:,.1f} |")

lines += [
    "\n### 总销量 Top 5 门店\n",
    "| 门店 | 总销量 |",
    "| --- | --- |",
]
for store, sales in top_store.items():
    lines.append(f"| {store} | {sales:,.0f} |")

lines += [
    "\n### 节假日与促销\n",
    "| 场景 | 平均销量 |",
    "| --- | --- |",
    f"| 普通日 | {holiday.get(0, 0):,.1f} |",
    f"| 节假日 | {holiday.get(1, 0):,.1f} |",
    f"| 无促销 | {promo.get(False, 0):,.1f} |",
    f"| 有促销 | {promo.get(True, 0):,.1f} |",
    f"\n全年销量峰值月份：{monthly.idxmax()}\n",
    "## 三、销量结构分析\n",
    "- 品类高度集中：GROCERY I、BEVERAGES、PRODUCE 三类平均销量显著高于其他品类，库存与补货资源应优先保障高销量品类。",
    "- 门店差异明显：门店 44、45、47、3、49 的总销量领先，说明门店规模和客流量差异很大，适合按门店分层制定目标。",
    "- 节假日有拉动作用：节假日平均销量高于普通日，建议在节前 1-2 周提高畅销品类的备货水位。",
    "- 促销是强驱动因素：促销日平均销量显著高于非促销日，促销排期可以直接作为销量预测的重要输入。",
    "- 季节性明显：12 月是全年销量峰值，年末备货和人力排班应提前规划。",
    "- 高销量组合是重点：高销量门店与高销量品类的组合贡献最大，适合建立重点商品清单和专项跟踪。\n",
    "## 四、模型表现数据\n",
    "| 模型 | RMSLE |",
    "| --- | --- |",
    f"| 上周销量基线 | {comparison['baseline_last_week']:.4f} |",
    f"| LightGBM | {comparison['lightgbm']:.4f} |",
    f"| XGBoost | {comparison['xgboost']:.4f} |",
    f"| 加权融合 | {comparison['ensemble_weighted']:.4f} |",
    f"| 与基线融合 | {comparison.get('blend_with_baseline', comparison['ensemble_weighted']):.4f} |\n",
    "## 五、模型分析\n",
    "- 直接多步预测优于递归预测：为第 1 至第 16 天分别构造特征，避免预测误差随天数累积，验证集 RMSLE 明显低于基线。",
    "- XGBoost 略优于 LightGBM，加权融合（LightGBM 20% + XGBoost 80%）进一步小幅提升。",
    "- 与基线融合未带来提升，说明模型已经掌握了超出“照抄上周”的信息。",
    "- 特征重要性中，历史销量、门店/品类历史均值、促销和节假日相关特征贡献最大，符合业务逻辑。",
    "- 部分品类误差仍然偏高，例如 LINGERIE、GROCERY II、CELEBRATION，后续可针对这些品类补充外部数据或单独建模。",
]

report = "\n".join(lines) + "\n"
out_path = OUT_DIR / "business_report.md"
out_path.write_text(report, encoding="utf-8")
print("已生成：", out_path)
