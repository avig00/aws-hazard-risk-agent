-- Top N counties by average NRI Expected Annual Loss (EAL) score within a single state.
-- Used when the question compares a specific county to others in the same state.
SELECT
    r.county_fips,
    d.county_name,
    d.state,
    ROUND(AVG(r.nri_eal_score), 4)      AS avg_eal_score,
    ROUND(AVG(r.nri_risk_score), 4)     AS avg_risk_score,
    ROUND(AVG(r.nri_sovi_score), 4)     AS avg_vulnerability,
    ROUND(AVG(r.nri_resl_score), 4)     AS avg_resilience,
    COUNT(*)                            AS years_on_record
FROM gold_hazard.risk_feature_mart_current r
JOIN gold_hazard.county_dim d ON r.county_fips = d.county_fips
WHERE r.year BETWEEN {start_year} AND {end_year}
  AND LOWER(d.state) = '{state_name}'
GROUP BY r.county_fips, d.county_name, d.state
ORDER BY avg_eal_score DESC
LIMIT {limit};
