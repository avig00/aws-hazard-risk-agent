-- Top N counties by event count for a specific hazard type.
-- Uses hazard_event_summary_current (per-hazard table) for accurate filtering.
-- Direct join to county_dim — no CTE needed (diagnostics confirmed no duplicate FIPS).
SELECT
    h.county_fips,
    c.county_name,
    c.state,
    h.hazard_type,
    SUM(h.event_count)              AS total_events,
    SUM(h.total_fatalities)         AS total_fatalities,
    SUM(h.total_injuries)           AS total_injuries,
    COUNT(DISTINCT h.year)          AS years_with_events
FROM gold_hazard.hazard_event_summary_current h
JOIN gold_hazard.county_dim c ON h.county_fips = c.county_fips
WHERE h.hazard_type = '{hazard_type}'
  AND h.year BETWEEN {start_year} AND {end_year}
GROUP BY h.county_fips, c.county_name, c.state, h.hazard_type
HAVING {order_col} > 0
ORDER BY {order_col} DESC
LIMIT {limit};
