# Data dictionary
**data/synthetic/demand_history.csv** - `month` (first day of month, Oct 2021 - Mar 2026), `module_demand_gw` (synthetic monthly module demand, GW).
**data/synthetic/suppliers.csv** - 24 suppliers: `lead_time_cv` (coefficient of variation of lead time), `geo_risk` (0-1), `price_vol` (price volatility, fraction), `single_source` (single-source dependence, fraction), `otif_gap` (1 - on-time-in-full, fraction).
**data/synthetic/component_risk.csv** - 1-5 risk scores per component on geopolitical, tariff, price, concentration.
**data/synthetic/scheduling_inputs.csv** - weekly line capacity (MW, Line 2 reduced in weeks 5-7), cumulative ALMM and export order books (MW), weekly domestic and imported cell receipts (MW-equivalent).
**data/processed/forecast_24m.csv** - FY27-FY28 monthly module demand and cell requirement (GW, 3% shrinkage).
**data/processed/deficit_sensitivity.csv** - FY27 deficit for 3 capacity ramps x 3 utilisation levels (Table 6).
**data/processed/supplier_clusters.csv** - suppliers with cluster, tier label and PCA scores.
**data/processed/schedule_*.csv** - MW produced by line, product and week under FIFO and LP.
**data/processed/kpi_summary.csv** - ALMM fill, export fill, utilisation, 12-week margin (Table 7).
