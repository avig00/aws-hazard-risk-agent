-- Year-over-year hazard event and damage trend
SELECT
    year,
    COUNT(DISTINCT county_fips)                     AS counties_affected,
    SUM(noaa_event_count)                           AS total_events,
    SUM(noaa_total_fatalities)                      AS total_fatalities,
    ROUND(AVG(noaa_avg_property_damage), 2)         AS avg_property_damage,
    SUM(fema_declaration_count)                     AS total_fema_declarations,
    ROUND(AVG(nri_eal_score), 2)                    AS avg_expected_loss
FROM gold_hazard.risk_feature_mart_current
WHERE year BETWEEN {start_year} AND {end_year}
GROUP BY year
ORDER BY year ASC;
