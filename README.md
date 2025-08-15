# Quick guide to the scripts here

Run the scripts in this directory to generate plots and analyze hydrogen production profiles for a microgrid system. The scripts read data from CSV files, process it, and create visualizations comparing different configurations (DC vs AC microgrid).

## sweep_multi_effs.py

Performs comprehensive parameter sweeps for DC and AC microgrid configurations, varying PV efficiency (90-98%), wind losses (4.9-17.5%), and battery availability losses (2.1-14.7%). Runs the H2Integrate model for each parameter combination while holding other parameters at baseline values. Outputs: `pv_efficiency_sweep_results_dc/ac.pkl`, `wind_loss_sweep_results_dc/ac.pkl`, `battery_availability_sweep_results_dc/ac.pkl` containing LCOE, LCOH, and hydrogen production data.

## plot_sweep_results.py

Visualizes efficiency sweep results from pickle files, creating 2-subplot comparison plots showing cost reduction percentages for PV and wind efficiency sweeps. Compares DC vs AC microgrid performance with labeled curves and statistical summaries. Outputs: `sweep_results_cost_reduction.png` and `sweep_results_comparison.png` showing hydrogen cost reduction vs efficiency parameters.

## plot_site.py

Creates a geographical map visualization showing the study site location in Northern Texas (35.2°N, 101.9°W) using Cartopy mapping tools. Generates a simple site map with state boundaries and coastlines for context. Outputs: `site_map.png` showing the wind farm location on a regional map.

## plot_hydrogen_profiles.py

Generates detailed hydrogen production time series plots comparing DC and AC microgrid configurations. Plots July monthly profiles and first week of July hourly profiles from sweep baseline results. Shows production differences between microgrid types over time with parameter annotations. Outputs: `hydrogen_production_july_comparison.png` and `hydrogen_production_july_week1_comparison.png`.

## run_optiwindnet.py
Optimizes wind farm cable layout using OptiWindNet for a 90-turbine wind farm (9×10 grid, 7D spacing) with 5 substations. Performs discrete cable sizing analysis comparing LVDC, MVAC, and MVDC technologies with different cable capacities (3-9 turbines). Calculates total cable costs and optimal routing. Outputs: `turbine_layout.png` showing wind farm layout with color-coded cable sizing and cost analysis summary.