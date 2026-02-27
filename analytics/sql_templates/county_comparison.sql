-- Side-by-side metric comparison for a list of counties
SELECT
    county_fips,
    county_name,
    state,
    year,
    total_events,
    total_fatalities,
    fema_claim_count,
    ROUND(fema_property_damage, 2)       AS property_damage,
    ROUND(NRI_ExpectedLoss, 2)           AS expected_loss,
    ROUND(NRI_Exposure, 4)              AS exposure,
    ROUND(NRI_SocialVulnerability, 4)   AS vulnerability,
    ROUND(NRI_CommunityResilience, 4)   AS resilience
FROM hazard_gold.risk_feature_mart
WHERE county_fips IN ({county_fips_list})
  AND year BETWEEN {start_year} AND {end_year}
ORDER BY county_fips, year ASC
LIMIT {limit};
