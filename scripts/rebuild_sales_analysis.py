import sqlite3
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "db" / "store_sales.db"
SQL = ROOT / "sql" / "02-data-preprocessing.sql"
OUT = ROOT / "outputs_v6" / "sales_analysis.csv"

with open(SQL, encoding="utf-8") as f:
    lines = [line for line in f.read().splitlines() if not line.strip().startswith("--")]
sql = "\n".join(lines)

conn = sqlite3.connect(DB, timeout=30)
conn.executescript(sql)
conn.commit()
df = pd.read_sql_query("SELECT * FROM sales_analysis", conn)
conn.close()

df.to_csv(OUT, index=False, encoding="utf-8")
print("已生成：", OUT)
print("行数：", len(df))
print(df["is_holiday"].value_counts().to_dict())
