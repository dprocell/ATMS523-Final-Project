# Predicting Significant Hail in the Central Great Plains: A Comparison of Atmospheric Instability Models


### In this repository, we use ERA5 data to analyze how CAPE, wind shear, and freezing level affect significant hail formation using different ML models.

#### Data Included:
- era5_raw folder
    - folder containing raw era5 reanalysis data 
- daily_hail.csv
    - datset with the following fields: date,hail_reports,hail_occurred
- era5_processed
    - dataset with the following fields: date,cape,shear,freezing_level
- final_dataset.csv
    - final cleaned dataset with the following fields: date,hail_occurred,cape,shear,freezing_level


