-- ============================================================
-- Task 5: SQL Queries for Telecom Churn Analysis
-- Works with SQLite (used in notebook) or any RDBMS
-- ============================================================

-- ── Query 1: Top 5 Customers by Revenue ──────────────────────
SELECT 
    customerID,
    Contract,
    tenure,
    MonthlyCharges,
    TotalCharges,
    RANK() OVER (ORDER BY TotalCharges DESC) AS revenue_rank
FROM customers
ORDER BY TotalCharges DESC
LIMIT 5;

-- ── Query 2: Monthly Revenue by Tenure Cohort ────────────────
SELECT 
    CASE 
        WHEN tenure BETWEEN 1  AND 6  THEN '01-06 months'
        WHEN tenure BETWEEN 7  AND 12 THEN '07-12 months'
        WHEN tenure BETWEEN 13 AND 24 THEN '13-24 months'
        WHEN tenure BETWEEN 25 AND 48 THEN '25-48 months'
        ELSE '49+ months'
    END AS tenure_cohort,
    COUNT(*)                        AS customer_count,
    ROUND(AVG(MonthlyCharges), 2)   AS avg_monthly_charge,
    ROUND(SUM(TotalCharges), 2)     AS total_revenue,
    ROUND(
        SUM(TotalCharges) * 100.0 / SUM(SUM(TotalCharges)) OVER (), 2
    )                               AS pct_of_total_revenue
FROM customers
GROUP BY tenure_cohort
ORDER BY tenure_cohort;

-- ── Query 3: Churn Rate by Contract & Internet Service ────────
SELECT 
    Contract,
    InternetService,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END)  AS churned,
    ROUND(
        SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2
    )                                                 AS churn_rate_pct,
    ROUND(
        SUM(CASE WHEN Churn = 'Yes' THEN MonthlyCharges ELSE 0 END), 2
    )                                                 AS monthly_revenue_at_risk
FROM customers
GROUP BY Contract, InternetService
ORDER BY churn_rate_pct DESC;

-- ── Query 4: High-Value At-Risk Customers ────────────────────
-- Customers who pay high monthly charges AND are on month-to-month
SELECT 
    customerID,
    MonthlyCharges,
    TotalCharges,
    tenure,
    InternetService,
    PaymentMethod,
    Churn
FROM customers
WHERE 
    Contract      = 'Month-to-month'
    AND MonthlyCharges > 70
    AND tenure     < 24
ORDER BY MonthlyCharges DESC
LIMIT 20;

-- ── Query 5: Payment Method Churn Analysis ───────────────────
SELECT
    PaymentMethod,
    COUNT(*) AS total,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned,
    ROUND(
        AVG(CASE WHEN Churn = 'Yes' THEN 1.0 ELSE 0 END) * 100, 2
    ) AS churn_rate_pct,
    ROUND(AVG(MonthlyCharges), 2) AS avg_monthly_charges
FROM customers
GROUP BY PaymentMethod
ORDER BY churn_rate_pct DESC;

-- ── BONUS: Star Schema Design ─────────────────────────────────
-- 
-- dim_customer (customer_id PK, gender, senior_citizen, has_partner, ...)
-- dim_plan     (plan_id PK, contract_type, internet_svc, phone_svc, ...)
-- dim_date     (date_id PK, year, month, quarter, month_name)
-- fact_billing (billing_id PK, customer_id FK, plan_id FK, date_id FK,
--               monthly_charges, total_charges, is_churned, tenure_months)
--
-- WHY STAR SCHEMA?
--   Optimized for analytical queries (GROUP BY, SUM, AVG)
--   Easy to add new dimensions (e.g., dim_location, dim_campaign)
--   Compatible with BI tools: Tableau, Power BI, Metabase
--   Denormalized = fewer JOINs = faster dashboard queries
