-- Year-over-year hazard event trend for a specific hazard type
SELECT
    year,
    COUNT(DISTINCT county_fips)          AS counties_affected,
    SUM(total_events)                    AS total_events,
    SUM(total_fatalities)                AS total_fatalities,
    ROUND(AVG(fema_property_damage), 2)  AS avg_property_damage,
    SUM(fema_claim_count)                AS total_claims
FROM hazard_gold.risk_feature_mart
WHERE year BETWEEN {start_year} AND {end_year}
  AND ({hazard_type} = 'all' OR hazard_type_primary = '{hazard_type}')
GROUP BY year
ORDER BY year ASC;
