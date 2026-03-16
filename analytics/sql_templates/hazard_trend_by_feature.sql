-- Year-over-year trend for a specific hazard type using the dedicated per-hazard
-- event-count columns in risk_feature_mart_current.
-- Covers the full historical range (2010–2023) unlike hazard_event_summary_current
-- which only has recent years for some hazard types.
SELECT
    year,
    COUNT(DISTINCT county_fips)         AS counties_affected,
    SUM({hazard_col})                   AS total_events
FROM gold_hazard.risk_feature_mart_current
WHERE year BETWEEN {start_year} AND {end_year}
GROUP BY year
ORDER BY year ASC
LIMIT {limit};
