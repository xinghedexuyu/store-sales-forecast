from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


CSV_PATH = Path(r"C:\Users\xingh\Desktop\sales_analysis.csv")
OUT_DIR = Path(__file__).resolve().parent.parent / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


def main():
    print("正在读取 sales_analysis.csv ...")
    try:
        df = pd.read_csv(CSV_PATH)
    except UnicodeDecodeError:
        df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")

    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.to_period("M")

    print("数据规模（行, 列）：", df.shape)
    print("\n字段类型：")
    print(df.dtypes)
    print("\n缺失值：")
    missing = df.isna().sum()
    print(missing[missing > 0] if (missing > 0).any() else "无缺失值")
    print("\n日期范围：", df["date"].min(), "到", df["date"].max())
    print("门店数量：", df["store_nbr"].nunique())
    print("商品品类数量：", df["family"].nunique())
    print("\n销量统计：")
    print(df["sales"].describe())

    print("\n平均销量最高的 10 个品类：")
    top_family = df.groupby("family")["sales"].mean().sort_values(ascending=False)
    print(top_family.head(10))

    print("\n总销量最高的 10 家门店：")
    top_store = df.groupby("store_nbr")["sales"].sum().sort_values(ascending=False)
    print(top_store.head(10))

    holiday_avg = df.groupby("is_holiday")["sales"].mean()
    print("\n节假日 vs 普通日平均销量：")
    print(holiday_avg)

    print("\n每月总销量（前 12 个月）：")
    monthly = df.groupby("month")["sales"].sum()
    print(monthly.head(12))

    daily = df.groupby("date")["sales"].sum().reset_index()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(daily["date"], daily["sales"], linewidth=1)
    ax.set_title("每日总销量趋势")
    ax.set_xlabel("日期")
    ax.set_ylabel("总销量")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "01_daily_sales_trend.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    top_family.iloc[::-1].head(10).plot(kind="barh", ax=ax, color="#2563EB")
    ax.set_title("平均销量最高的 10 个品类")
    ax.set_xlabel("平均销量")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "02_top_families.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    top_store.iloc[::-1].head(10).plot(kind="barh", ax=ax, color="#F97316")
    ax.set_title("总销量最高的 10 家门店")
    ax.set_xlabel("总销量")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "03_top_stores.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    holiday_avg.rename({0: "普通日", 1: "节假日"}).plot(
        kind="bar", ax=ax, color=["#94A3B8", "#EF4444"]
    )
    ax.set_title("节假日 vs 普通日平均销量")
    ax.set_ylabel("平均销量")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "04_holiday_effect.png", dpi=120)
    plt.close(fig)

    print("\n图片已保存到：", OUT_DIR)


if __name__ == "__main__":
    main()
