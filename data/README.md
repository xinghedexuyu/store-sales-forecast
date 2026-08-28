# 数据下载说明

1. 打开 [Store Sales 竞赛页](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data)。
2. 点击 Download All 下载压缩包。
3. 解压后把以下 CSV 放进本目录：

```text
train.csv
test.csv
stores.csv
oil.csv
holidays_events.csv
transactions.csv
sample_submission.csv
```

也可以先安装 Kaggle CLI：

```bash
pip install kaggle
kaggle competitions download -c store-sales-time-series-forecasting
```

CSV 放好后，双击项目根目录的 `import_data.bat` 即可自动生成数据库。

数据文件比较大，`.gitignore` 已配置为不入库。
