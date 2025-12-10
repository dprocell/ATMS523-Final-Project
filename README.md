# Predicting Significant Hail in the Central Great Plains: A Comparison of Atmospheric Instability Models


### In this repository, we use ERA5 data to analyze how CAPE, wind shear, and freezing level affect significant hail formation using different ML models.

---
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
* final_dataset_boxed.csv (00Z for atmoshperic variable extraction)
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
* final_dataset_boxed_daily_max_cape.csv (Maximum daily CAPE extraction)
    * Final cleaned dataset, same as final_dataset_boxed.csv, but using daily maximum CAPE instead of fixed time of 00Z
    * Atmospheric conditions extracted at time of maximum CAPE each day
    * Assumes hail occurs at same time as maximum CAPE, instead of guessing that hail occurs around 6pm CDT each day
    * Different approach, neither is perfect because hail reports from NOAA have no time fields, just date



<img width="330" height="370" alt="box_methodology" src="https://github.com/user-attachments/assets/99534b40-746c-4d80-8b4f-12056a9ef676" />  

<img width="450" height="270" alt="box_concept" src="https://github.com/user-attachments/assets/bc73ce6c-ebd4-4377-9e06-0056132cecd5" />

 ---
  #### Analysis Script
  To run: `python HailAnaylsis.py`

---
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


---
#### Data Download script
To download your own data:
* run `python boxed_data_download.py` for fixed 00Z hail atmoshperic data + hail (initial approach)
* run `python boxed_datadownload_daily_max_cape.py` for maximum CAPE atmospheric data + hail (secondary approach)
This assumes you have your CDS API key set up on your machine: https://cds.climate.copernicus.eu/how-to-api

---
#### Data Cited

##### ERA5 Reanalysis (Atmospheric Data):

Hersbach, H., Bell, B., Berrisford, P., Hirahara, S., Horányi, A., Muñoz-Sabater, J., 
    Nicolas, J., Peubey, C., Radu, R., Schepers, D., Simmons, A., Soci, C., Abdalla, 
    S., Abellan, X., Balsamo, G., Bechtold, P., Biavati, G., Bidlot, J., Bonavita, M., 
    De Chiara, G., Dahlgren, P., Dee, D., Diamantakis, M., Dragani, R., Flemming, J., 
    Forbes, R., Fuentes, M., Geer, A., Haimberger, L., Healy, S., Hogan, R. J., Hólm, 
    E., Janisková, M., Keeley, S., Laloyaux, P., Lopez, P., Lupu, C., Radnoti, G., de 
    Rosnay, P., Rozum, I., Vamborg, F., Villaume, S., and Thépaut, J.-N. (2020). The 
    ERA5 global reanalysis. Quarterly Journal of the Royal Meteorological Society, 
    146(730), 1999-2049. https://doi.org/10.1002/qj.3803

Hersbach, H., Bell, B., Berrisford, P., Biavati, G., Horányi, A., Muñoz Sabater, J., 
    Nicolas, J., Peubey, C., Radu, R., Rozum, I., Schepers, D., Simmons, A., Soci, C., 
    Dee, D., Thépaut, J-N. (2023). ERA5 hourly data on single levels from 1940 to 
    present. Copernicus Climate Change Service (C3S) Climate Data Store (CDS). 
    https://doi.org/10.24381/cds.adbb2d47

##### NOAA Storm Prediction Center Severe Weather Reports (Hail Data)
NOAA National Weather Service Storm Prediction Center (2024). Severe Weather Database Files (1950-present). National Oceanic and Atmospheric Administration. Retrieved from https://www.spc.noaa.gov/wcm/
