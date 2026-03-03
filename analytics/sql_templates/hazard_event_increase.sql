-- Counties with the largest absolute increase in a specific hazard's event count
-- between two time periods.  Queries hazard_event_summary (per-hazard, per-year)
-- so results are filtered to the exact hazard type the user asked about,
-- unlike risk_feature_mart which stores only all-hazard aggregates.
WITH period_a AS (
    SELECT
        h.county_fips,
        SUM(h.event_count)                              AS total_events_a
    FROM gold_hazard.hazard_event_summary_current h
    WHERE h.year BETWEEN {period_a_start} AND {period_a_end}
      AND h.hazard_type = '{hazard_type}'
    GROUP BY h.county_fips
),
period_b AS (
    SELECT
        h.county_fips,
        SUM(h.event_count)                              AS total_events_b
    FROM gold_hazard.hazard_event_summary_current h
    WHERE h.year BETWEEN {period_b_start} AND {period_b_end}
      AND h.hazard_type = '{hazard_type}'
    GROUP BY h.county_fips
)
SELECT
    a.county_fips,
    d.county_name,
    d.state,
    a.total_events_a                                    AS events_early_period,
    b.total_events_b                                    AS events_recent_period,
    (b.total_events_b - a.total_events_a)               AS absolute_increase,
    ROUND(
        CAST(b.total_events_b - a.total_events_a AS double)
        / NULLIF(a.total_events_a, 0) * 100, 1
    )                                                   AS pct_increase
FROM period_a a
JOIN period_b b ON a.county_fips = b.county_fips
JOIN gold_hazard.county_dim d ON a.county_fips = d.county_fips
ORDER BY absolute_increase DESC
LIMIT {limit};
