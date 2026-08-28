# Store Sales 销量预测实战项目

一个从业务数据到线上看板的完整零售销量预测项目：基于 Kaggle Store Sales 数据集，预测厄瓜多尔 54 家门店、33 个商品品类未来 16 天的日销量。

- 版本：最终版
- 验证：3 段滚动直接多步验证，加权融合 RMSLE 0.3986
- 对比：上周销量基线 0.6302，预测误差降低约 36.7%
- 模型：LightGBM + XGBoost（Tweedie 目标函数，验证集搜索融合权重）
- 看板：Streamlit 多页交互看板

## 核心成果

| 验证段 | 基线 | LightGBM | XGBoost |
| --- | --- | --- | --- |
| 2017-06-30 -> 07-15 | 0.6090 | 0.3909 | 0.3866 |
| 2017-07-15 -> 07-31 | 0.6179 | 0.4115 | 0.4087 |
| 2017-07-31 -> 08-15 | 0.6625 | 0.4055 | 0.4009 |

整体 RMSLE：基线 0.6302，LightGBM 0.4027，XGBoost 0.3989，加权融合 0.3986。

> 指标为 RMSLE（均方根对数误差），越小越好。

## 项目流程

1. SQL 数据提取：SQLite 多表导入，多表关联生成分析宽表
2. 数据质量检查：缺失值、异常值、重复值、日期范围
3. 探索性分析：品类、门店、节假日、促销、油价
4. 特征工程：日期、节假日距离、促销滚动、油价滚动、滞后与滚动统计等约 50 维特征
5. 建模：LightGBM + XGBoost，直接多步预测第 1-16 天
6. 模型验证：3 段滚动验证，验证集搜索融合权重
7. 看板与部署：Streamlit 看板，可部署到社区云

## 目录结构

```text
store-sales-forecast/
├── app.py                        # Streamlit 看板
├── README.md
├── requirements.txt
├── .gitignore
├── data/                         # Kaggle 数据（大文件不入库）
│   ├── test.csv                  # 预测用测试集（看板依赖）
│   ├── holidays_events.csv       # 节假日（看板依赖）
│   ├── oil.csv / stores.csv
│   └── README.md
├── db/                           # SQLite 数据库（不入库）
├── sql/                          # SQL 教程与练习
├── notebooks/                    # 分析笔记
├── scripts/
│   ├── full_pipeline_v6.py       # 最终版完整流程
│   ├── generate_business_report.py
│   └── load_to_sqlite.py
├── outputs_v6/                   # 最终结果
│   ├── submission.csv            # Kaggle 提交文件
│   ├── business_report.md        # 业务分析报告
│   ├── metrics.json              # 模型指标
│   ├── feature_importance.csv
│   └── feature_sales_correlation.csv
└── plots_v6/                     # 12 张交互图
```

## 本地运行

```powershell
cd store-sales-forecast
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

浏览器打开 http://localhost:8501。

数据说明：仓库已包含看板运行需要的小文件（`test.csv`、`holidays_events.csv`、`oil.csv`、`stores.csv`）；`train.csv`、`transactions.csv`、SQLite 数据库和约 195MB 的中间宽表 `sales_analysis.csv` 不入库，需要完整复现时从 Kaggle 下载数据后运行 `python scripts/full_pipeline_v6.py`。

## 复现完整流程

```powershell
python scripts/full_pipeline_v6.py
```

## 线上部署

### Streamlit Community Cloud

1. 把项目推到 GitHub。
2. 打开 [share.streamlit.io](https://share.streamlit.io)，用 GitHub 账号登录。
3. New app：选择仓库和分支，主文件填 `app.py`，点击 Deploy。
4. 部署后访问 `https://<你的账号>-store-sales-forecast.streamlit.app`。

### Hugging Face Spaces

1. 创建 Space，SDK 选择 Streamlit。
2. 把代码推送到 Space 仓库。
3. 保持 `app.py`、`requirements.txt`、`data/` 小文件和 `outputs_v6/` 在仓库中。

### 推送到 GitHub

```powershell
git init
git add .
git commit -m "Store Sales 销量预测最终版"
git branch -M main
git remote add origin https://github.com/<你的账号>/store-sales-forecast.git
git push -u origin main
```

## 岗位职责对照

| 岗位要求 | 项目里怎么做 |
| --- | --- |
| 业务数据提取、数据处理 | SQL 多表关联 + Python 数据清洗 |
| 需求预测、销量预测 | 未来 16 天门店/品类销量预测 |
| 数据异常分析、专题探究 | 缺失值、异常值、节假日/油价/促销分析 |
| 资料收集、报告输出 | README、业务分析报告、Streamlit 汇报看板 |

## 输出文件说明

| 文件 | 说明 |
| --- | --- |
| `outputs_v6/submission.csv` | Kaggle 标准提交文件，28,512 行 |
| `outputs_v6/business_report.md` | 业务分析报告（先数据、后分析） |
| `outputs_v6/metrics.json` | 参数、分段指标、融合权重 |
| `outputs_v6/feature_importance.csv` | 特征重要性 |
| `outputs_v6/feature_sales_correlation.csv` | 特征与销量相关性 |
| `outputs_v6/per_family_errors.csv` | 分品类误差 |
| `plots_v6/*.html` | 12 张交互式分析图，含综合 Dashboard |
