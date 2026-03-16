-- Year-over-year trend for a specific hazard type.
-- Uses hazard_event_summary_current for per-hazard event counts by year.
-- Note: coverage varies by hazard type — wildfire records in this table only
-- cover recent years; flood/tornado have full 2010–2023 history.
SELECT
    year,
    COUNT(DISTINCT county_fips)         AS counties_affected,
    SUM(event_count)                    AS total_events
FROM gold_hazard.hazard_event_summary_current
WHERE hazard_type = '{hazard_type}'
  AND year BETWEEN {start_year} AND {end_year}
GROUP BY year
ORDER BY year ASC
LIMIT {limit};
