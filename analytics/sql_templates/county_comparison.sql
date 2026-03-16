-- Side-by-side metric comparison for a list of counties
SELECT
    r.county_fips,
    d.county_name,
    d.state,
    r.year,
    r.noaa_event_count,
    r.noaa_total_fatalities,
    r.fema_declaration_count,
    ROUND(r.fema_total_damage, 2)           AS fema_total_damage,
    ROUND(r.nri_eal_score, 2)               AS expected_loss,
    ROUND(r.nri_risk_score, 4)             AS risk_score,
    ROUND(r.nri_sovi_score, 4)             AS vulnerability,
    ROUND(r.nri_resl_score, 4)             AS resilience
FROM gold_hazard.risk_feature_mart_current r
JOIN gold_hazard.county_dim d ON r.county_fips = d.county_fips
WHERE r.county_fips IN ({county_fips_list})
  AND r.year BETWEEN {start_year} AND {end_year}
ORDER BY r.county_fips, r.year ASC
LIMIT {limit};
