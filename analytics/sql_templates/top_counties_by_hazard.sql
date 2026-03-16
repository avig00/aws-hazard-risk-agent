-- Top N counties by event count for a specific hazard type.
-- Uses hazard_event_summary_current (per-hazard table) for accurate filtering.
-- Deduplicates county_dim (one row per FIPS) before joining to avoid duplicate
-- county rows when county_dim contains multiple state values for the same FIPS.
WITH county AS (
    -- Keep only rows where county_name contains the state name, which filters out
    -- corrupt rows (e.g. county_name="Macon County, Georgia" but state="Alaska").
    -- Then deduplicate to one row per FIPS in case multiple valid rows remain.
    SELECT county_fips,
           MIN(county_name) AS county_name,
           MIN(state)       AS state
    FROM gold_hazard.county_dim
    WHERE LOWER(county_name) LIKE '%' || LOWER(state) || '%'
    GROUP BY county_fips
)
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
JOIN county c ON h.county_fips = c.county_fips
WHERE h.hazard_type = '{hazard_type}'
  AND h.year BETWEEN {start_year} AND {end_year}
GROUP BY h.county_fips, c.county_name, c.state, h.hazard_type
ORDER BY {order_col} DESC
LIMIT {limit};
