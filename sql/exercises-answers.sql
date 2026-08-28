-- 参考答案：先自己写，再对答案

-- 1. 查看 train 表前 10 行
SELECT *
FROM train
LIMIT 10;

-- 2. 找出 2017-08-15 当天销量最高的 10 个“门店 + 品类”
SELECT date, store_nbr, family, sales
FROM train
WHERE date = '2017-08-15'
ORDER BY sales DESC
LIMIT 10;

-- 3. 统计每家门店的总销量，从高到低排序
SELECT store_nbr,
       SUM(sales) AS total_sales
FROM train
GROUP BY store_nbr
ORDER BY total_sales DESC;

-- 4. 统计每个品类的平均销量，只保留平均销量大于 500 的品类
SELECT family,
       AVG(sales) AS avg_sales
FROM train
GROUP BY family
HAVING avg_sales > 500;

-- 5. 关联 stores 和 train，统计 Pichincha 州每个门店的总销量
SELECT s.store_nbr,
       s.city,
       SUM(t.sales) AS total_sales
FROM train t
JOIN stores s ON t.store_nbr = s.store_nbr
WHERE s.state = 'Pichincha'
GROUP BY s.store_nbr, s.city
ORDER BY total_sales DESC;

-- 6. 用 LEFT JOIN 区分“节假日”和“普通日”，对比平均销量
SELECT CASE
           WHEN h.date IS NOT NULL THEN 'holiday'
           ELSE 'normal'
       END AS day_type,
       AVG(t.sales) AS avg_sales
FROM train t
LEFT JOIN holidays_events h
       ON t.date = h.date
      AND h.transferred = 0
GROUP BY day_type;

-- 7. 按月份汇总总销量和平均油价
SELECT strftime('%Y-%m', t.date) AS month,
       SUM(t.sales) AS total_sales,
       AVG(o.dcoilwtico) AS avg_oil_price
FROM train t
LEFT JOIN oil o ON t.date = o.date
GROUP BY month
ORDER BY month;

-- 8. 用 CASE WHEN 区分促销日和非促销日，对比平均销量
SELECT CASE
           WHEN onpromotion > 0 THEN 'with_promotion'
           ELSE 'without_promotion'
       END AS promo_status,
       AVG(sales) AS avg_sales
FROM train
GROUP BY promo_status;

-- 9. 找出全部门店合计销量最高的 3 天
SELECT date,
       SUM(sales) AS total_sales
FROM train
GROUP BY date
ORDER BY total_sales DESC
LIMIT 3;

-- 10. 用窗口函数给每个门店的品类销量排名，看每个门店销量最高的品类
SELECT store_nbr,
       family,
       total_sales,
       sales_rank
FROM (
    SELECT store_nbr,
           family,
           SUM(sales) AS total_sales,
           ROW_NUMBER() OVER (PARTITION BY store_nbr ORDER BY SUM(sales) DESC) AS sales_rank
    FROM train
    GROUP BY store_nbr, family
)
WHERE sales_rank = 1
ORDER BY store_nbr;
