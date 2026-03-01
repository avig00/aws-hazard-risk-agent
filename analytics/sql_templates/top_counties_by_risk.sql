-- Top N counties by average NRI Expected Annual Loss (EAL) score
SELECT
    r.county_fips,
    d.county_name,
    d.state,
    ROUND(AVG(r.nri_eal_score), 2)      AS avg_expected_loss,
    ROUND(AVG(r.nri_risk_score), 4)     AS avg_risk_score,
    ROUND(AVG(r.nri_sovi_score), 4)     AS avg_vulnerability,
    ROUND(AVG(r.nri_resl_score), 4)     AS avg_resilience,
    COUNT(*)                            AS years_on_record
FROM gold_hazard.risk_feature_mart r
JOIN gold_hazard.county_dim d ON r.county_fips = d.county_fips
WHERE r.year BETWEEN {start_year} AND {end_year}
GROUP BY r.county_fips, d.county_name, d.state
ORDER BY avg_expected_loss DESC
LIMIT {limit};
