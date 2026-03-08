-- Year-over-year trend for a specific hazard type.
-- Uses hazard_event_summary_current (per-hazard) rather than risk_feature_mart
-- (all-hazard aggregates) so results reflect the named hazard's actual event counts.
SELECT
    year,
    COUNT(DISTINCT county_fips)         AS counties_affected,
    SUM(event_count)                    AS total_events
FROM gold_hazard.hazard_event_summary_current
WHERE year BETWEEN {start_year} AND {end_year}
  AND hazard_type = '{hazard_type}'
GROUP BY year
ORDER BY year ASC
LIMIT {limit};
