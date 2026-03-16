-- States ranked by total FEMA declaration count across all counties
SELECT
    d.state,
    SUM(r.fema_declaration_count)               AS total_declarations,
    COUNT(DISTINCT r.county_fips)               AS counties_affected,
    SUM(r.fema_declaration_count) / NULLIF(COUNT(DISTINCT r.county_fips), 0)
                                                AS avg_declarations_per_county,
    ROUND(SUM(r.fema_total_damage), 2)          AS total_fema_damage
FROM gold_hazard.risk_feature_mart_current r
JOIN gold_hazard.county_dim d ON r.county_fips = d.county_fips
WHERE r.year BETWEEN {start_year} AND {end_year}
GROUP BY d.state
ORDER BY total_declarations DESC
LIMIT {limit};
