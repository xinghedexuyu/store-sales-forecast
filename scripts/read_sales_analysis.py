import sqlite3

import pandas as pd


conn = sqlite3.connect("file:db/store_sales.db?mode=ro", uri=True, timeout=30)
df = pd.read_sql_query("SELECT * FROM sales_analysis", conn)
conn.close()

print("行数、列数：", df.shape)
print("\n字段类型：")
print(df.dtypes)
print("\n前 5 行：")
print(df.head())
print("\n缺失值统计：")
print(df.isna().sum())

print("\n按品类统计平均销量（前 10）：")
print(
    df.groupby("family")["sales"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)
