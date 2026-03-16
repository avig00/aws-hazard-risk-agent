-- Counties with the largest absolute increase in NOAA events between two periods
WITH period_a AS (
    SELECT
        r.county_fips,
        d.county_name,
        d.state,
        AVG(r.noaa_event_count) AS avg_events_a
    FROM gold_hazard.risk_feature_mart_current r
    JOIN gold_hazard.county_dim d ON r.county_fips = d.county_fips
    WHERE r.year BETWEEN {period_a_start} AND {period_a_end}
    GROUP BY r.county_fips, d.county_name, d.state
),
period_b AS (
    SELECT
        county_fips,
        AVG(noaa_event_count) AS avg_events_b
    FROM gold_hazard.risk_feature_mart_current
    WHERE year BETWEEN {period_b_start} AND {period_b_end}
    GROUP BY county_fips
)
SELECT
    a.county_fips,
    a.county_name,
    a.state,
    ROUND(a.avg_events_a, 2)                            AS avg_events_early,
    ROUND(b.avg_events_b, 2)                            AS avg_events_recent,
    ROUND(b.avg_events_b - a.avg_events_a, 2)           AS absolute_increase
FROM period_a a
JOIN period_b b ON a.county_fips = b.county_fips
ORDER BY absolute_increase DESC
LIMIT {limit};
