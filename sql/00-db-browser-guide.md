# DB Browser for SQLite 使用指南

## 1. 安装

1. 打开 https://sqlitebrowser.org/dl/
2. 下载 Windows 64 位安装包
3. 安装完成后打开 DB Browser for SQLite

## 2. 打开练习数据库

项目里已经放了一个演示数据库：

```text
db/store_sales_demo.db
```

1. 点击 **Open Database**
2. 选择 `db/store_sales_demo.db`
3. 左侧 **Database Structure** 可以看到 `train`、`stores`、`oil`、`holidays_events`、`transactions`

## 3. 三个常用页面

| 页面 | 作用 |
| --- | --- |
| Database Structure | 看有哪些表和字段 |
| Browse Data | 直接查看表格数据 |
| Execute SQL | 写 SQL、执行 SQL、看查询结果 |

## 4. 第一次练习

1. 切换到 **Execute SQL** 页面
2. 粘贴这一句：

```sql
SELECT *
FROM train
LIMIT 10;
```

3. 点击 **Run**
4. 下方出现 10 行数据，就成功了
5. 接下来打开 `sql/exercises.sql`，一次粘贴一道题练习

## 5. 常见问题

- 打不开数据库：确认选的是 `db/store_sales_demo.db`
- 没有数据：确认不是新建了一个空数据库
- SQL 报错：一次只执行一条语句，看错误提示里写的是第几行
- 想重新练习：可以点 **File -> New Database** 重新打开

## 6. 换真实数据

下载 Kaggle 数据放入 `data/` 后运行：

```bash
python scripts/load_to_sqlite.py
```

它会生成 `db/store_sales.db`，之后打开这个真实数据库练习即可。
