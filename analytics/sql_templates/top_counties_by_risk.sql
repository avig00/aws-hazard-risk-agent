-- Top N counties by average predicted risk score
SELECT
    county_fips,
    county_name,
    state,
    ROUND(AVG(NRI_ExpectedLoss), 2)        AS avg_expected_loss,
    ROUND(AVG(NRI_Exposure), 4)            AS avg_exposure,
    ROUND(AVG(NRI_SocialVulnerability), 4) AS avg_vulnerability,
    COUNT(*)                               AS years_on_record
FROM hazard_gold.risk_feature_mart
WHERE year BETWEEN {start_year} AND {end_year}
GROUP BY county_fips, county_name, state
ORDER BY avg_expected_loss DESC
LIMIT {limit};
