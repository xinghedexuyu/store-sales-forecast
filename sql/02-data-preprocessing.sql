-- =============================================
-- Store Sales 数据预处理 SQL
-- 在 DB Browser 的 Execute SQL 页面执行
-- 每段前面都有中文说明
-- =============================================

-- 1. 看 train 表有多少行
SELECT COUNT(*) AS row_count
FROM train;

-- 2. 看 train 表有哪些列
PRAGMA table_info(train);

-- 3. 看前 10 行数据，了解数据长什么样
SELECT *
FROM train
LIMIT 10;

-- 4. 检查 train 表每一列有多少缺失值
SELECT
    COUNT(*) AS total_rows,
    SUM(CASE WHEN date IS NULL THEN 1 ELSE 0 END) AS missing_date,
    SUM(CASE WHEN store_nbr IS NULL THEN 1 ELSE 0 END) AS missing_store,
    SUM(CASE WHEN family IS NULL THEN 1 ELSE 0 END) AS missing_family,
    SUM(CASE WHEN sales IS NULL THEN 1 ELSE 0 END) AS missing_sales,
    SUM(CASE WHEN onpromotion IS NULL THEN 1 ELSE 0 END) AS missing_promo
FROM train;

-- 5. 检查 oil 表油价缺失情况
SELECT
    COUNT(*) AS total_days,
    SUM(CASE WHEN dcoilwtico IS NULL THEN 1 ELSE 0 END) AS missing_oil
FROM oil;

-- 6. 检查重复记录：同一个日期+门店+品类是否出现多次
SELECT date, store_nbr, family, COUNT(*) AS cnt
FROM train
GROUP BY date, store_nbr, family
HAVING COUNT(*) > 1;

-- 7. 看 train 日期范围
SELECT MIN(date) AS min_date, MAX(date) AS max_date
FROM train;

-- 8. 看 test 日期范围，确认它是未来的预测区间
SELECT MIN(date) AS min_date, MAX(date) AS max_date
FROM test;

-- 9. 看销量分布：最小值、最大值、平均值
SELECT
    MIN(sales) AS min_sales,
    MAX(sales) AS max_sales,
    AVG(sales) AS avg_sales,
    COUNT(*) AS row_count
FROM train;

-- 10. 找销量为负数的异常记录
SELECT date, store_nbr, family, sales
FROM train
WHERE sales < 0;

-- 11. 看门店数量和品类数量
SELECT
    COUNT(DISTINCT store_nbr) AS store_count,
    COUNT(DISTINCT family) AS family_count
FROM train;

-- 12. 看每天记录数是否一致，检查数据完整性
SELECT date, COUNT(*) AS row_count
FROM train
GROUP BY date
ORDER BY row_count ASC
LIMIT 10;

-- 13. 检查 stores 表里的门店是否都有销售记录
SELECT s.store_nbr, s.city
FROM stores s
LEFT JOIN train t ON s.store_nbr = t.store_nbr
WHERE t.store_nbr IS NULL;

-- 14. 油价缺失填充：用前一天油价填充
WITH oil_filled AS (
    SELECT date,
           dcoilwtico,
           LAG(dcoilwtico) OVER (ORDER BY date) AS prev_oil_price
    FROM oil
)
SELECT date,
       COALESCE(dcoilwtico, prev_oil_price) AS oil_price_filled
FROM oil_filled
ORDER BY date;

-- 15. 创建预处理宽表：把 5 张表拼成一张分析表
DROP TABLE IF EXISTS sales_analysis;
CREATE TABLE sales_analysis AS
SELECT
    t.date,
    t.store_nbr,
    t.family,
    t.sales,
    COALESCE(t.onpromotion, 0) AS onpromotion,
    s.city,
    s.state,
    s.type AS store_type,
    s.cluster,
    COALESCE(o.dcoilwtico, 0) AS oil_price,
    CASE WHEN h.date IS NOT NULL THEN 1 ELSE 0 END AS is_holiday,
    COALESCE(tr.transactions, 0) AS daily_transactions
FROM train t
LEFT JOIN stores s ON t.store_nbr = s.store_nbr
LEFT JOIN oil o ON t.date = o.date
LEFT JOIN (
    SELECT DISTINCT date
    FROM holidays_events
    WHERE transferred = 0
) h ON t.date = h.date
LEFT JOIN transactions tr ON t.date = tr.date AND t.store_nbr = tr.store_nbr;

-- 16. 验证宽表生成成功
SELECT COUNT(*) AS row_count
FROM sales_analysis;

-- 17. 检查宽表里是否还有关键缺失值
SELECT
    COUNT(*) AS total_rows,
    SUM(CASE WHEN sales IS NULL THEN 1 ELSE 0 END) AS missing_sales,
    SUM(CASE WHEN city IS NULL THEN 1 ELSE 0 END) AS missing_city,
    SUM(CASE WHEN state IS NULL THEN 1 ELSE 0 END) AS missing_state
FROM sales_analysis;

-- 18. 查看宽表结果示例
SELECT *
FROM sales_analysis
LIMIT 10;
