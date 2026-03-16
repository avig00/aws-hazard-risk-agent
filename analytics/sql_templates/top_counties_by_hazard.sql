-- Top N counties by event count for a specific hazard type.
-- Uses hazard_event_summary_current (per-hazard table) for accurate filtering.
-- Note: per-hazard fatality breakdowns are not stored in this table;
--       total_events is the best available proxy for hazard exposure.
SELECT
    h.county_fips,
    d.county_name,
    d.state,
    h.hazard_type,
    SUM(h.event_count)              AS total_events,
    COUNT(DISTINCT h.year)          AS years_with_events
FROM gold_hazard.hazard_event_summary_current h
JOIN gold_hazard.county_dim d ON h.county_fips = d.county_fips
WHERE h.hazard_type = '{hazard_type}'
  AND h.year BETWEEN {start_year} AND {end_year}
GROUP BY h.county_fips, d.county_name, d.state, h.hazard_type
ORDER BY total_events DESC
LIMIT {limit};
