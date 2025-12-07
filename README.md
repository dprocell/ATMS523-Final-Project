# Predicting Significant Hail in the Central Great Plains: A Comparison of Atmospheric Instability Models


### In this repository, we use ERA5 data to analyze how CAPE, wind shear, and freezing level affect significant hail formation using different ML models.



#### Data Included
* era5_raw folder
    * Contains raw era5 reanalysis data in .nc format from Copernicus: https://cds.climate.copernicus.eu/
    * Data fields:
        * CAPE (J/kg)
        * Shear: U and V wind components at 500mb and 850mb (m/s)
        * freezing level (m)
    * Spatial resolution: 1 x 1 degree spatial resolution
    * Region: 33-42°N, 103-94°W
    * Time: 00Z
    * Months: April - September
    * Years: 2020-2021
* hail_reports.csv
    * Source: NOAA Storm Prediction Center: https://www.spc.noaa.gov/wcm/data/
    * Fields:
        * date (YYYY-MM-DD)
        * slat
        * slon
        * mag (hail size in inches)
        * mo (Month)
    * Region: 33-42°N, 103-94°W
    * Time: 00Z
    * Months: April - September
    * Years: 2020-2021
* final_dataset_boxed.csv
    * final cleaned dataset with hail samples, generated non hail samples, and CAPE=0 filtered out
    * 5285 samples total, Hail: 2757 (51.2%), Non-hail: 2580 (48.8%)
    * Columns
        * date (YYYY-MM-DD)
        * lat
        * lon
        * hail_occurred (binary target variable: 1=hail, 0=no hail)
        * cape (J/kg)
        * shear (0-6 km wind shear magnitude (m/s), calculated as √[(U₅₀₀-U₈₅₀)² + (V₅₀₀-V₈₅₀)²])
        * freezing_level (meters above sea level)
    * Date range: April 2, 2020 to September 29, 2021
    * Spatial coverage: Hail Alley (33-42°N, 103-94°W)

 
  #### Analysis Script
  To run: `python HailAnaylsis.py`


  #### Figures
  * all_models_metrics_comparison.png
  * all_models_roc_comparison.png
  * coefficient_comparison.png
  * confusion_matrices.png
  * feature_distributions.png
  * figures/metrics_comparison.png
  * figures/pca_biplot.png
  * pca_loadings_variance.png
  * rf_feature_importances.png
  * rf_learning_curve.png
  * rf_partial_dependence.png
  * rf_sample_tree.png
  * roc_pr_curves.png

      


