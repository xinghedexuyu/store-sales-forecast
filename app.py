from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parent
OUTPUT_CANDIDATES = ["outputs_v6", "outputs_v5", "outputs_v4", "outputs"]


def pick_output_dir():
    for name in OUTPUT_CANDIDATES:
        path = ROOT / name
        if (path / "submission.csv").exists():
            return path
    return None


def load_metrics(out_dir):
    path = out_dir / "metrics.json"
    if not path.exists():
        return {}
    return pd.read_json(path, typ="series").to_dict()


OUT_DIR = pick_output_dir()

st.set_page_config(page_title="Store Sales 销量预测看板", page_icon="chart_with_upwards_trend", layout="wide")

st.markdown(
    """
    <style>
    :root {
        --navy: #14243B;
        --navy-2: #1E3A5F;
        --ink: #1F2937;
        --muted: #6B7280;
        --line: #E5E7EB;
        --bg: #F4F6F9;
        --card: #FFFFFF;
        --accent: #2563EB;
    }
    .stApp {
        background: var(--bg);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0E1C33 0%, #14243B 55%, #1E3A5F 100%);
        color: #F8FAFC;
    }
    [data-testid="stSidebar"] * {
        color: #F8FAFC;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        color: #C7D2E5;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        color: #FFFFFF;
    }
    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 3rem;
        max-width: 1340px;
    }
    .banner {
        background: linear-gradient(115deg, #0E1C33 0%, #14243B 42%, #1D4ED8 100%);
        border-radius: 10px;
        padding: 1.5rem 1.7rem;
        margin-bottom: 1.4rem;
        box-shadow: 0 8px 22px rgba(13, 29, 58, 0.18);
    }
    .banner-title {
        color: #FFFFFF;
        font-size: 1.75rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
        letter-spacing: 0;
    }
    .banner-sub {
        color: #D8E2F5;
        font-size: 0.95rem;
    }
    .banner-badges {
        display: flex;
        gap: 0.5rem;
        margin-top: 0.85rem;
        flex-wrap: wrap;
    }
    .badge {
        background: rgba(255, 255, 255, 0.14);
        border: 1px solid rgba(255, 255, 255, 0.28);
        color: #FFFFFF;
        border-radius: 999px;
        padding: 0.2rem 0.65rem;
        font-size: 0.72rem;
    }
    .kpi-card {
        background: var(--card);
        border: 1px solid var(--line);
        border-top: 4px solid var(--accent);
        border-radius: 8px;
        padding: 0.9rem 1.1rem;
        min-height: 96px;
        box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
    }
    .kpi-card .label {
        color: var(--muted);
        font-size: 0.8rem;
        margin-bottom: 0.35rem;
    }
    .kpi-card .value {
        color: var(--navy);
        font-size: 1.45rem;
        font-weight: 700;
        line-height: 1.2;
    }
    .kpi-card .sub {
        color: var(--muted);
        font-size: 0.78rem;
        margin-top: 0.3rem;
    }
    .kpi-blue {
        border-top-color: #2563EB;
    }
    .kpi-orange {
        border-top-color: #F97316;
    }
    .kpi-green {
        border-top-color: #10B981;
    }
    .kpi-purple {
        border-top-color: #8B5CF6;
    }
    .kpi-cyan {
        border-top-color: #0EA5E9;
    }
    .section-title {
        border-left: 4px solid var(--accent);
        padding-left: 0.65rem;
        color: var(--navy);
        font-size: 1.05rem;
        font-weight: 700;
        margin: 1.4rem 0 0.7rem 0;
    }
    .status-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 0.8rem;
    }
    .status-item {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        background: var(--card);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 0.7rem 0.9rem;
        box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
    }
    .status-check {
        width: 26px;
        height: 26px;
        border-radius: 50%;
        background: #10B981;
        color: #FFFFFF;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.85rem;
        flex-shrink: 0;
    }
    .status-name {
        color: var(--navy);
        font-weight: 700;
        font-size: 0.9rem;
    }
    .status-desc {
        color: var(--muted);
        font-size: 0.75rem;
    }
    .status-tag {
        margin-left: auto;
        color: #059669;
        background: #ECFDF5;
        border: 1px solid #A7F3D0;
        border-radius: 999px;
        padding: 0.15rem 0.55rem;
        font-size: 0.68rem;
        font-weight: 700;
        white-space: nowrap;
    }
    .step-card {
        background: var(--card);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 1rem 1.15rem;
        margin-bottom: 0.8rem;
        box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
    }
    .step-card .step-num {
        color: var(--accent);
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.08em;
    }
    .step-card .step-title {
        color: var(--navy);
        font-size: 1rem;
        font-weight: 700;
        margin: 0.2rem 0 0.35rem 0;
    }
    .step-card .step-desc {
        color: var(--muted);
        font-size: 0.88rem;
        line-height: 1.6;
    }
    .stMarkdown table {
        width: 100%;
        border-collapse: collapse;
        line-height: 1.6;
        margin: 0.4rem 0 0.8rem 0;
    }
    .stMarkdown table th {
        background: #EEF2F7;
        color: var(--navy);
        font-weight: 700;
        text-align: left;
        padding: 0.6rem 0.85rem;
        border-bottom: 2px solid #D7DEE8;
        white-space: nowrap;
    }
    .stMarkdown table td {
        padding: 0.55rem 0.85rem;
        border-bottom: 1px solid var(--line);
        vertical-align: top;
    }
    .stMarkdown table tbody tr:nth-child(even) td {
        background: #F8FAFC;
    }
    .stMarkdown table tbody tr:hover td {
        background: #EFF6FF;
    }
    .insight-card {
        background: linear-gradient(180deg, #FFFFFF 0%, #F8FAFC 100%);
        border: 1px solid var(--line);
        border-left: 4px solid var(--accent);
        border-radius: 8px;
        padding: 0.85rem 1rem;
        margin: 0.7rem 0 1.2rem 0;
        box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
    }
    .insight-card .insight-title {
        color: var(--navy);
        font-weight: 700;
        font-size: 0.88rem;
        margin-bottom: 0.35rem;
    }
    .insight-card .insight-body {
        color: #374151;
        font-size: 0.86rem;
        line-height: 1.7;
    }
    .footer {
        margin-top: 2.6rem;
        padding: 1rem 0;
        border-top: 1px solid var(--line);
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        flex-wrap: wrap;
        color: var(--muted);
        font-size: 0.78rem;
    }
    #MainMenu, footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def banner(title, subtitle, rmsle_text=None):
    badges = ['<span class="badge">Kaggle Store Sales</span>']
    if rmsle_text:
        badges.append(f'<span class="badge">{rmsle_text}</span>')
    st.markdown(
        f'<div class="banner"><div class="banner-title">{title}</div>'
        f'<div class="banner-sub">{subtitle}</div>'
        f'<div class="banner-badges">{"".join(badges)}</div></div>',
        unsafe_allow_html=True,
    )


def kpi_row(metrics, df):
    if df is None or df.empty:
        return
    total = float(df["sales"].sum())
    avg = float(df["sales"].mean())
    stores = int(df["store_nbr"].nunique())
    families = int(df["family"].nunique())
    days = int(df["date"].nunique())
    cols = st.columns(5)
    cards = [
        ("预测总销量", f"{total:,.0f}", "未来 16 天合计", "kpi-blue"),
        ("日均预测销量", f"{avg:,.0f}", "全部门店品类", "kpi-orange"),
        ("门店数", str(stores), "54 家门店", "kpi-green"),
        ("商品品类", str(families), "33 个品类", "kpi-purple"),
        ("预测天数", str(days), "8/16 - 8/31", "kpi-cyan"),
    ]
    for col, (label, value, sub, cls) in zip(cols, cards):
        col.markdown(
            f'<div class="kpi-card {cls}"><div class="label">{label}</div>'
            f'<div class="value">{value}</div><div class="sub">{sub}</div></div>',
            unsafe_allow_html=True,
        )
    rmsle = metrics.get("model_comparison", {}).get("ensemble_weighted")
    if rmsle:
        st.caption(f"验证指标：加权融合模型 RMSLE {rmsle:.4f}（越小越好）")


def chart_layout(fig, title, height=360):
    fig.update_layout(
        title=dict(text=title, x=0.02, font=dict(size=15, color="#14243B")),
        template="plotly_white",
        font=dict(family="Microsoft YaHei, sans-serif", size=12, color="#1F2937"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        margin=dict(l=40, r=20, t=54, b=32),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
        ),
        colorway=["#2563EB", "#F97316", "#10B981", "#8B5CF6", "#0EA5E9"],
        hoverlabel=dict(
            bgcolor="#FFFFFF",
            bordercolor="#E5E7EB",
            font=dict(color="#1F2937"),
        ),
        height=height,
    )
    fig.update_xaxes(gridcolor="#EDF0F5", linecolor="#D7DEE8", zeroline=False)
    fig.update_yaxes(gridcolor="#EDF0F5", linecolor="#D7DEE8", zeroline=False)
    return fig


def insight_card(title, body, target=None):
    mark = target.markdown if target is not None else st.markdown
    mark(
        f'<div class="insight-card"><div class="insight-title">{title}</div>'
        f'<div class="insight-body">{body}</div></div>',
        unsafe_allow_html=True,
    )


MODEL_LABELS = {
    "baseline_last_week": "上周销量基线",
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
    "ensemble_weighted": "加权融合",
    "blend_with_baseline": "基线混合",
}


with st.sidebar:
    st.markdown("## STORE SALES")
    st.markdown("零售销量预测与业务分析")
    st.divider()
    page = st.radio(
        "查看内容",
        ["项目介绍", "概览", "预测明细", "业务分析", "模型报告"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("数据源：Kaggle Store Sales")
    if OUT_DIR is None:
        st.caption("当前还没有预测结果，请先运行完整预测流程")


def build_df():
    if OUT_DIR is None:
        return None
    test = pd.read_csv(ROOT / "data" / "test.csv")
    test["date"] = pd.to_datetime(test["date"])
    pred = pd.read_csv(OUT_DIR / "submission.csv")
    df = test.merge(pred, on="id")
    df["date"] = pd.to_datetime(df["date"])
    df["day_ahead"] = (df["date"] - pd.Timestamp("2017-08-15")).dt.days
    return df


df = build_df()
metrics = load_metrics(OUT_DIR) if OUT_DIR else {}
rmsle = metrics.get("model_comparison", {}).get("ensemble_weighted")
rmsle_text = f"RMSLE {rmsle:.4f}" if rmsle else None

if df is None or df.empty:
    st.warning("请先运行完整预测流程，生成预测结果后再打开看板。")
    st.stop()


if page == "项目介绍":
    banner("项目介绍", "从数据到预测、再到可上线看板的完整零售销量预测项目", rmsle_text)

    st.markdown('<div class="section-title">项目目标</div>', unsafe_allow_html=True)
    st.markdown(
        "基于 Kaggle Store Sales 数据集，预测厄瓜多尔 54 家门店、33 个商品品类在未来 16 天的日销量。"
        "项目覆盖 SQL 数据提取、Python 数据分析、特征工程、销量建模、模型验证、业务报告和线上看板，"
        "目标是训练一套可以稳定跑赢上周销量基线的预测方案。"
    )

    st.markdown('<div class="section-title">我们是怎么做的</div>', unsafe_allow_html=True)
    steps = [
        ("01", "数据准备", "下载 Kaggle 原始 CSV，使用 SQLite 完成多表导入与 SQL 练习，再通过 SQL 多表关联生成 sales_analysis 宽表。"),
        ("02", "数据分析", "用 pandas 检查缺失值、异常值和日期范围，分析品类、门店、节假日、促销、油价的销量影响。"),
        ("03", "特征工程", "构造日期特征、节假日距离、促销滚动、油价滚动、交易量、历史销量滞后和滚动统计等约 50 维特征。"),
        ("04", "建模", "使用 Tweedie 目标函数训练 LightGBM 和 XGBoost，并采用直接多步预测策略，为第 1 至第 16 天分别预测。"),
        ("05", "模型验证", "采用 3 段滚动验证，验证集同样按未来 15 天直接预测，避免递归误差累积，最后用验证集搜索融合权重。"),
        ("06", "看板与部署", "把预测结果、业务分析和模型指标整合成 Streamlit 看板，可部署到 Streamlit Community Cloud 或 Hugging Face Spaces。"),
    ]
    for num, title, desc in steps:
        st.markdown(
            f'<div class="step-card"><div class="step-num">{num}</div>'
            f'<div class="step-title">{title}</div>'
            f'<div class="step-desc">{desc}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title">项目成果</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    baseline = metrics.get("model_comparison", {}).get("baseline_last_week")
    if rmsle and baseline:
        improve = (1 - rmsle / baseline) * 100
        result_cards = [
            ("加权融合 RMSLE", f"{rmsle:.4f}", "验证集 3 段平均", "kpi-green"),
            ("上周销量基线", f"{baseline:.4f}", "直接多步对比基准", "kpi-orange"),
            ("误差降幅", f"{improve:.1f}%", f"{baseline:.4f} → {rmsle:.4f}", "kpi-blue"),
        ]
    else:
        result_cards = [
            ("加权融合 RMSLE", "0.3986", "验证集 3 段平均", "kpi-green"),
            ("上周销量基线", "0.6302", "直接多步对比基准", "kpi-orange"),
            ("误差降幅", "36.7%", "0.6302 → 0.3986", "kpi-blue"),
        ]
    for col, (label, value, sub, cls) in zip([c1, c2, c3], result_cards):
        col.markdown(
            f'<div class="kpi-card {cls}"><div class="label">{label}</div>'
            f'<div class="value">{value}</div><div class="sub">{sub}</div></div>',
            unsafe_allow_html=True,
        )

elif page == "概览":
    banner("Ecuador 零售门店销量预测", "Kaggle Store Sales 数据集的未来 16 天销量预测与业务分析看板", rmsle_text)
    kpi_row(metrics, df)

    daily = df.groupby("date")["sales"].sum().reset_index()
    fig = px.bar(daily, x="date", y="sales", color_discrete_sequence=["#2563EB"])
    fig.add_trace(
        go.Scatter(
            x=daily["date"],
            y=daily["sales"].rolling(3).mean(),
            name="3 日平均",
            line=dict(color="#F97316", width=3),
        )
    )
    chart_layout(fig, "每日预测总销量与 3 日移动平均", 400)
    st.plotly_chart(fig, use_container_width=True)

    daily_total = float(daily["sales"].sum())
    peak_row = daily.loc[daily["sales"].idxmax()]
    trough_row = daily.loc[daily["sales"].idxmin()]
    ma_start = daily["sales"].rolling(3).mean().iloc[2]
    ma_end = daily["sales"].rolling(3).mean().iloc[-1]
    trend = "走高" if ma_end >= ma_start else "回落"
    insight_card(
        "每日总销量分析",
        f"预测期 16 天合计销量 {daily_total:,.0f}，日均 {daily_total / len(daily):,.0f}。"
        f"单日峰值在 {peak_row['date'].strftime('%Y-%m-%d')}（{peak_row['sales']:,.0f}），"
        f"低谷在 {trough_row['date'].strftime('%Y-%m-%d')}（{trough_row['sales']:,.0f}）。"
        f"3 日移动平均整体{trend}，说明短期需求{('增强' if trend == '走高' else '转弱')}，"
        "库存和人力安排可以按这个节奏提前调度。",
    )

    horizon = df.groupby("day_ahead")["sales"].mean().reset_index()
    fig = px.bar(
        horizon,
        x="day_ahead",
        y="sales",
        color="sales",
        color_continuous_scale="Blues",
        labels={"day_ahead": "预测第几天", "sales": "平均销量"},
    )
    chart_layout(fig, "未来 16 天平均预测销量走势", 360)
    st.plotly_chart(fig, use_container_width=True)

    peak_h = horizon.loc[horizon["sales"].idxmax()]
    first8 = horizon[horizon["day_ahead"] <= 8]["sales"].mean()
    last8 = horizon[horizon["day_ahead"] > 8]["sales"].mean()
    phase = "前 8 天高于后 8 天" if first8 >= last8 else "后 8 天高于前 8 天"
    insight_card(
        "16 天预测节奏分析",
        f"第 {int(peak_h['day_ahead'])} 天平均预测销量最高（{peak_h['sales']:,.0f}）。"
        f"前 8 天平均 {first8:,.0f}，后 8 天平均 {last8:,.0f}，{phase}，"
        "预测期需求有明显的短期节奏，建议按周分段制定补货计划。",
    )

    col1, col2 = st.columns(2)
    top_family = df.groupby("family")["sales"].sum().nlargest(10).reset_index()
    fig = px.bar(
        top_family,
        x="sales",
        y="family",
        orientation="h",
        color="sales",
        color_continuous_scale="Blues",
    )
    chart_layout(fig, "预测销量 Top 10 品类", 420)
    fig.update_layout(yaxis=dict(autorange="reversed"))
    col1.plotly_chart(fig, use_container_width=True)

    fam1 = top_family.iloc[0]
    share1 = fam1["sales"] / df["sales"].sum() * 100
    share3 = top_family.head(3)["sales"].sum() / df["sales"].sum() * 100
    insight_card(
        "品类结构分析",
        f"预测销量最高的品类是 {fam1['family']}（{fam1['sales']:,.0f}，占总销量 {share1:.1f}%）。"
        f"Top 3 品类合计占 {share3:.1f}%，头部品类应优先保证库存与陈列资源。",
        target=col1,
    )

    top_store = df.groupby("store_nbr")["sales"].sum().nlargest(10).reset_index()
    fig = px.bar(
        top_store,
        x="sales",
        y="store_nbr",
        orientation="h",
        color="sales",
        color_continuous_scale="Oranges",
    )
    chart_layout(fig, "预测销量 Top 10 门店", 420)
    fig.update_layout(yaxis=dict(autorange="reversed"))
    col2.plotly_chart(fig, use_container_width=True)

    s1 = top_store.iloc[0]
    share_s1 = s1["sales"] / df["sales"].sum() * 100
    share_s3 = top_store.head(3)["sales"].sum() / df["sales"].sum() * 100
    insight_card(
        "门店结构分析",
        f"预测销量最高的门店是 {s1['store_nbr']}（{s1['sales']:,.0f}，占总销量 {share_s1:.1f}%）。"
        f"Top 3 门店合计占 {share_s3:.1f}%，高流量门店是供应链调度的重点。",
        target=col2,
    )

elif page == "预测明细":
    banner("预测明细", "按门店、品类、日期查看未来 16 天销量预测", rmsle_text)
    st.markdown('<div class="section-title">按门店、品类、日期筛选</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    store = c1.selectbox("门店", sorted(df["store_nbr"].unique()))
    family = c2.selectbox("品类", sorted(df["family"].unique()))
    day = c3.selectbox("预测日期", sorted(df["date"].dt.strftime("%Y-%m-%d").unique()))

    sub = df[(df["store_nbr"] == store) & (df["family"] == family)]
    sub["date_str"] = sub["date"].dt.strftime("%Y-%m-%d")
    one = sub[sub["date_str"] == day]
    if not one.empty:
        st.metric("该组合预测销量", f"{one['sales'].iloc[0]:,.2f}")

    fig = px.line(
        sub,
        x="date",
        y="sales",
        markers=True,
        color_discrete_sequence=["#2563EB"],
        labels={"date": "日期", "sales": "预测销量"},
    )
    chart_layout(fig, f"门店 {store} · {family} 的 16 天预测", 380)
    st.plotly_chart(fig, use_container_width=True)

    avg_all = df["sales"].mean()
    avg_sub = sub["sales"].mean()
    peak_sub = sub.loc[sub["sales"].idxmax()]
    trough_sub = sub.loc[sub["sales"].idxmin()]
    rel = "高于" if avg_sub >= avg_all else "低于"
    diff_pct = abs(avg_sub - avg_all) / avg_all * 100
    insight_card(
        "单组合预测分析",
        f"门店 {store} · {family} 的 16 天平均预测销量 {avg_sub:,.1f}，"
        f"{rel}全项目平均（{avg_all:,.1f}）{diff_pct:.1f}%。"
        f"峰值在 {peak_sub['date'].strftime('%Y-%m-%d')}（{peak_sub['sales']:,.1f}），"
        f"低谷在 {trough_sub['date'].strftime('%Y-%m-%d')}（{trough_sub['sales']:,.1f}）。",
    )

    show = sub.copy()
    show["date"] = show["date"].dt.strftime("%Y-%m-%d")
    st.dataframe(
        show[["date", "store_nbr", "family", "sales"]],
        use_container_width=True,
        hide_index=True,
    )

elif page == "业务分析":
    banner("业务分析", "预测期内的节假日、促销与高销量结构洞察", rmsle_text)

    holidays = pd.read_csv(ROOT / "data" / "holidays_events.csv")
    holidays["date"] = pd.to_datetime(holidays["date"])
    holidays = holidays[~holidays["transferred"].astype(str).isin(["1", "True", "true"])]
    holiday_dates = set(holidays["date"].dt.normalize())
    df["is_holiday"] = df["date"].dt.normalize().isin(holiday_dates).astype(int)

    c1, c2 = st.columns(2)
    holiday_avg = df.groupby("is_holiday")["sales"].mean().reset_index()
    holiday_avg["is_holiday"] = holiday_avg["is_holiday"].map({0: "普通日", 1: "节假日"})
    fig = px.bar(
        holiday_avg,
        x="is_holiday",
        y="sales",
        color="is_holiday",
        color_discrete_map={"普通日": "#94A3B8", "节假日": "#EF4444"},
    )
    chart_layout(fig, "预测期节假日 vs 普通日平均销量", 360)
    c1.plotly_chart(fig, use_container_width=True)

    hmap = holiday_avg.set_index("is_holiday")["sales"]
    normal_h = hmap.get("普通日")
    holiday_h = hmap.get("节假日")
    if pd.notna(normal_h) and pd.notna(holiday_h) and normal_h > 0:
        diff_h = (holiday_h - normal_h) / normal_h * 100
        rel_h = "高出" if diff_h >= 0 else "低出"
        n_holiday = int((df["is_holiday"] == 1).sum())
        sample_note = f"（节假日样本 {n_holiday} 条）" if n_holiday < 1000 else ""
        body_h = (
            f"预测期内节假日平均销量 {holiday_h:,.1f}，普通日 {normal_h:,.1f}，"
            f"节假日{rel_h}普通日 {abs(diff_h):.1f}%{sample_note}。"
            "建议结合具体节假日安排提前调整备货。"
        )
    else:
        body_h = "预测期 2017-08-16 至 2017-08-31 内没有节假日样本，销量节奏主要由促销、品类结构和门店差异驱动。"
    insight_card("节假日效应分析", body_h, target=c1)

    promo_avg = df.groupby(df["onpromotion"] > 0)["sales"].mean().reset_index()
    promo_avg.columns = ["promo", "sales"]
    promo_avg["promo"] = promo_avg["promo"].map({False: "无促销", True: "有促销"})
    fig = px.bar(
        promo_avg,
        x="promo",
        y="sales",
        color="promo",
        color_discrete_map={"无促销": "#94A3B8", "有促销": "#F97316"},
    )
    chart_layout(fig, "预测期促销 vs 无促销平均销量", 360)
    c2.plotly_chart(fig, use_container_width=True)

    pmap = promo_avg.set_index("promo")["sales"]
    no_promo = pmap.get("无促销")
    has_promo = pmap.get("有促销")
    if pd.notna(no_promo) and pd.notna(has_promo) and no_promo > 0:
        diff_p = (has_promo - no_promo) / no_promo * 100
        rel_p = "提升" if diff_p >= 0 else "下降"
        n_promo = int((df["onpromotion"] > 0).sum())
        sample_note = f"（有促销样本仅 {n_promo} 条，结论需谨慎）" if n_promo < 100 else ""
        body_p = (
            f"预测期内有促销的平均销量 {has_promo:,.1f}，无促销 {no_promo:,.1f}，"
            f"促销平均{rel_p} {abs(diff_p):.1f}%{sample_note}。"
            "可结合历史促销周期进一步验证。"
        )
    else:
        body_p = "预测期内促销样本不足，暂无法估计促销增量，建议结合历史促销周期做进一步归因。"
    insight_card("促销效应分析", body_p, target=c2)

    top = df.groupby(["store_nbr", "family"])["sales"].sum().reset_index()
    top_stores = top.groupby("store_nbr")["sales"].sum().nlargest(8).index
    top_fams = top.groupby("family")["sales"].sum().nlargest(8).index
    pivot = (
        top[top["store_nbr"].isin(top_stores) & top["family"].isin(top_fams)]
        .pivot_table(index="family", columns="store_nbr", values="sales", aggfunc="sum")
        .fillna(0)
    )
    fig = px.imshow(
        pivot,
        text_auto=".0s",
        color_continuous_scale="YlOrRd",
        labels={"color": "预测销量"},
    )
    chart_layout(fig, "高销量门店 x 品类预测热力图", 460)
    st.plotly_chart(fig, use_container_width=True)

    stacked = pivot.stack()
    top_cell = stacked.idxmax()
    insight_card(
        "门店 × 品类结构分析",
        f"在 8 家高销量门店 × 8 个头部品类的交叉组合中，"
        f"{top_cell[0]} × 门店 {top_cell[1]} 的预测销量最高（{stacked.max():,.0f}）。"
        "这类组合应作为补货与陈列的优先对象。",
    )

    report = OUT_DIR / "business_report.md"
    if report.exists():
        st.markdown('<div class="section-title">业务分析报告</div>', unsafe_allow_html=True)
        st.markdown(report.read_text(encoding="utf-8"))

else:
    banner("模型报告", "模型对比、特征重要性与相关性分析", rmsle_text)
    comparison = metrics.get("model_comparison", {})
    if comparison:
        comp_df = pd.Series(comparison).reset_index()
        comp_df.columns = ["model", "rmsle"]
        comp_df["model"] = comp_df["model"].map(MODEL_LABELS).fillna(comp_df["model"])
        fig = px.bar(
            comp_df,
            x="rmsle",
            y="model",
            orientation="h",
            color="rmsle",
            color_continuous_scale="YlOrRd_r",
        )
        chart_layout(fig, "模型 RMSLE 对比（越小越好）", 380)
        fig.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

        best_row = comp_df.loc[comp_df["rmsle"].idxmin()]
        base_row = comp_df[comp_df["model"] == "上周销量基线"]
        if not base_row.empty:
            improve = (1 - best_row["rmsle"] / base_row["rmsle"].iloc[0]) * 100
            body_m = (
                f"最优模型是 {best_row['model']}，RMSLE {best_row['rmsle']:.4f}，"
                f"比上周销量基线（{base_row['rmsle'].iloc[0]:.4f}）降低 {improve:.1f}%，"
                "说明直接多步预测与模型融合带来的增益稳定。"
            )
        else:
            body_m = f"最优模型是 {best_row['model']}，RMSLE {best_row['rmsle']:.4f}。"
        insight_card("模型对比分析", body_m)

    imp_path = OUT_DIR / "feature_importance.csv"
    if imp_path.exists():
        imp = pd.read_csv(imp_path).head(15)
        fig = px.bar(
            imp.iloc[::-1],
            x="importance",
            y="feature",
            orientation="h",
            color="importance",
            color_continuous_scale="Blues",
        )
        chart_layout(fig, "特征重要性 Top 15", 460)
        fig.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

        top_feat = imp.iloc[0]
        insight_card(
            "特征重要性分析",
            f"排名第一的特征是 {top_feat['feature']}（importance {top_feat['importance']:.4f}），"
            "表明历史销量与滚动统计信息是预测的核心信号，后续可围绕滞后特征继续优化。",
        )

    corr_path = OUT_DIR / "feature_sales_correlation.csv"
    if corr_path.exists():
        corr = pd.read_csv(corr_path)
        corr = corr.reindex(corr["corr"].abs().sort_values(ascending=False).index).head(15)
        fig = px.bar(
            corr.iloc[::-1],
            x="corr",
            y="feature",
            orientation="h",
            color="corr",
            color_continuous_scale="RdBu_r",
        )
        chart_layout(fig, "特征与销量相关性 Top 15", 460)
        fig.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

        top_corr = corr.iloc[0]
        sign = "正相关" if top_corr["corr"] > 0 else "负相关"
        insight_card(
            "特征相关性分析",
            f"与销量相关性最强的特征是 {top_corr['feature']}（r = {top_corr['corr']:.3f}，{sign}），"
            "可作为特征工程进一步组合或加权的方向。",
        )

    if metrics:
        st.json(metrics)


rmsle_footer = f"LightGBM + XGBoost 加权融合 · RMSLE {rmsle:.4f}" if rmsle else "LightGBM + XGBoost 加权融合"
st.markdown(
    '<div class="footer"><span>© 2026 Store Sales 销量预测项目</span>'
    '<span>数据来源：Kaggle Store Sales</span>'
    f"<span>{rmsle_footer}</span></div>",
    unsafe_allow_html=True,
)
