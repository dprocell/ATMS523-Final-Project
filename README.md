# Predicting Significant Hail in the Central Great Plains: A Comparison of Atmospheric Instability Models


### In this repository, we use ERA5 data to analyze how CAPE, wind shear, and freezing level affect significant hail formation using different ML models.

#### Data Included:
- era5_raw folder
    - folder containing raw era5 reanalysis data in .nc format
    - Fields: Time, lat, lon, variable (CAPE, shear, lifted index, freezing level)
    - 1 x 1 degree resolution
    - 18Z temporal resolution, April - September 2020-2021, daily data
- daily_hail.csv
    - Fields: date, hail_reports, hail_occurred
    - Years 1955-2023, daily hail reports. Contains only days WITH hail
    - Filtered:Geographically for "Hail Alley", Time of study period: April - September 2020-2021, Hail size: significant hail is 1 inch in diameter or larger
    - Source: NOAA SPC: https://www.spc.noaa.gov/wcm/data/1955-2023_hail.csv
- era5_processed.csv
    - Fields: date,cape,shear,freezing_level
    - consolidate 100 grid points per day to 1 per day - regional mean values
    - atmospheric data for each day, regardless of hail or not
    - spatial averaging introduces error by diluting the "extremes" part of severe weather
- final_dataset.csv
    - final cleaned dataset with the following fields: date,hail_occurred,cape,shear,freezing_level
    - binary of 0 or 1 for hail/ no hail
    - contains data for all days, not just hail days
      


