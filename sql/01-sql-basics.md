# SQL 入门教程：用 Store Sales 数据练习

本教程假设你使用 DB Browser for SQLite，并已运行：

```bash
python scripts/load_to_sqlite.py
```

打开 `db/store_sales.db` 后，就可以照着下面的例子练习。

## 1. 认识表结构

先看每张表有哪些列：

```sql
PRAGMA table_info(train);
PRAGMA table_info(stores);
```

核心表：

- `train`：每天每家门店每个品类的销量
- `stores`：门店信息
- `oil`：每日油价
- `holidays_events`：节假日
- `transactions`：门店每日交易笔数

## 2. 最基础：SELECT 和 LIMIT

```sql
SELECT *
FROM train
LIMIT 10;
```

只选需要的列：

```sql
SELECT date, store_nbr, family, sales
FROM train
LIMIT 10;
```

## 3. WHERE：筛选条件

```sql
SELECT date, store_nbr, family, sales
FROM train
WHERE family = 'DAIRY'
  AND sales > 1000
LIMIT 20;
```

常用条件：

- `=` 等于
- `>`、`>=`、`<`、`<=`
- `AND` / `OR`
- `BETWEEN 100 AND 500`
- `IN ('A', 'B')`
- `LIKE 'A%'` 模糊匹配
- `IS NULL` 判断空值

## 4. ORDER BY 和 LIMIT：排序取前 N

```sql
SELECT date, store_nbr, family, sales
FROM train
ORDER BY sales DESC
LIMIT 10;
```

## 5. GROUP BY：分组汇总

统计每个品类有多少行、总销量、平均销量：

```sql
SELECT family,
       COUNT(*)       AS row_count,
       SUM(sales)     AS total_sales,
       AVG(sales)     AS avg_sales
FROM train
GROUP BY family
ORDER BY total_sales DESC;
```

`HAVING` 用于对分组结果继续筛选：

```sql
SELECT family, AVG(sales) AS avg_sales
FROM train
GROUP BY family
HAVING avg_sales > 500;
```

## 6. JOIN：多表关联

把销量和门店信息关联起来：

```sql
SELECT t.date,
       s.city,
       s.state,
       t.family,
       t.sales
FROM train t
JOIN stores s ON t.store_nbr = s.store_nbr
LIMIT 10;
```

`JOIN` 只保留能匹配上的行；`LEFT JOIN` 保留左边全部行，右边没有就显示 `NULL`。

## 7. CASE WHEN：把数值变成业务分类

```sql
SELECT
    CASE
        WHEN sales >= 1000 THEN 'high'
        WHEN sales >= 300 THEN 'medium'
        ELSE 'low'
    END AS sales_level,
    COUNT(*) AS day_count
FROM train
GROUP BY sales_level;
```

## 8. 日期函数

SQLite 里日期按文本存储，常用：

```sql
SELECT date,
       strftime('%Y', date)  AS year,
       strftime('%m', date)  AS month,
       strftime('%w', date)  AS weekday
FROM train
LIMIT 10;
```

按月汇总：

```sql
SELECT strftime('%Y-%m', date) AS month,
       SUM(sales) AS total_sales
FROM train
GROUP BY month
ORDER BY month;
```

## 9. 三个练习方向

做完 `exercises.sql` 后，建议你结合业务再写三条自己的查询：

1. 节假日对销量的影响
2. 促销强度与销量的关系
3. 油价和销量的月度趋势

这就是岗位描述里的“业务问题 -> 数据问题”。

## 10. 用 Python 执行 SQL

先连接数据库，再用 `pd.read_sql_query()` 把查询结果直接变成 DataFrame：

```python
import sqlite3
import pandas as pd

conn = sqlite3.connect("db/store_sales.db")

df = pd.read_sql_query(
    """
    SELECT family,
           AVG(sales) AS avg_sales,
           SUM(sales) AS total_sales
    FROM train
    GROUP BY family
    ORDER BY total_sales DESC
    """,
    conn,
)

print(df.head())
conn.close()
```

这样 SQL 负责取数和聚合，Python 负责后续的可视化和建模，正好是数据分析师工作的常见组合。
