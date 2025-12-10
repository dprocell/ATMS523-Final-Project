"""
Download and preprocess hail and ERA5 data with box averaging approach
This version creates location-specific samples instead of regional averages
"""

import pandas as pd
import numpy as np
import xarray as xr
import cdsapi
import os
from datetime import datetime

os.makedirs('data', exist_ok=True)
os.makedirs('data/era5_raw', exist_ok=True)

# Download hail reports for 2020 and 2021

hail_dfs = []
for year in [2020, 2021]:
    url = f"https://www.spc.noaa.gov/wcm/data/{year}_hail.csv"
    df = pd.read_csv(url)
    hail_dfs.append(df)

# Combine years
hail_df = pd.concat(hail_dfs, ignore_index=True)

# Parse existing date column (it's in YYYY-MM-DD format already)
hail_df['date'] = pd.to_datetime(hail_df['date'])

# Filter for Hail Alley (33-42°N, -103 to -94°W)
hail_df = hail_df[
    (hail_df['slat'] >= 33) & (hail_df['slat'] <= 42) &
    (hail_df['slon'] >= -103) & (hail_df['slon'] <= -94)
]

# Filter for April-September
hail_df = hail_df[
    (hail_df['mo'] >= 4) & (hail_df['mo'] <= 9)
]

# Filter for significant hail 
hail_df = hail_df[hail_df['mag'] >= 1.0]
hail_df.to_csv('data/hail_reports.csv', index=False)
try:
    c = cdsapi.Client()
except Exception as e:
    print("ERROR: Could not initialize CDS API client")

region = [42, -103, 33, -94]
for year in [2020, 2021]:
    for month in range(4, 10):  # April through September
        
        filename = f'data/era5_raw/era5_single_{year}_{month:02d}.nc'
        
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'variable': [
                    'convective_available_potential_energy',
                    'zero_degree_level',
                    'total_column_water_vapour',
                ],
                'year': str(year),
                'month': f'{month:02d}',
                'day': [f'{d:02d}' for d in range(1, 32)],
                'time': '00:00',
                'area': region,
                'grid': [1.0, 1.0],
                'format': 'netcdf',
            },
            filename
        )

# Download pressure level variables for wind shear calculation
for year in [2020, 2021]:
    for month in range(4, 10):
        
        filename = f'data/era5_raw/era5_winds_{year}_{month:02d}.nc'
        
        c.retrieve(
            'reanalysis-era5-pressure-levels',
            {
                'product_type': 'reanalysis',
                'variable': [
                    'u_component_of_wind',
                    'v_component_of_wind',
                ],
                'pressure_level': ['500', '850'],
                'year': str(year),
                'month': f'{month:02d}',
                'day': [f'{d:02d}' for d in range(1, 32)],
                'time': '00:00',
                'area': region,
                'grid': [1.0, 1.0],
                'format': 'netcdf',
            },
            filename
        )

expected_files = []
for year in [2020, 2021]:
    for month in range(4, 10):
        expected_files.append(f'data/era5_raw/era5_single_{year}_{month:02d}.nc')
        expected_files.append(f'data/era5_raw/era5_winds_{year}_{month:02d}.nc')


# Load all ERA5 data
single_files = [f'data/era5_raw/era5_single_{year}_{month:02d}.nc' 
                for year in [2020, 2021] for month in range(4, 10)]
winds_files = [f'data/era5_raw/era5_winds_{year}_{month:02d}.nc' 
               for year in [2020, 2021] for month in range(4, 10)]

ds_single = xr.open_mfdataset(single_files, combine='by_coords')
ds_winds = xr.open_mfdataset(winds_files, combine='by_coords')

# Handle vairable names
var_names_single = list(ds_single.variables.keys())
var_names_winds = list(ds_winds.variables.keys())

cape_var = 'cape' if 'cape' in var_names_single else None
freezing_var = None
for name in ['deg0l', 'zero_degree_level', 'z0']:
    if name in var_names_single:
        freezing_var = name
        break

level_dim = None
for dim in ds_winds.dims.keys():
    if 'level' in dim.lower() or 'pressure' in dim.lower() or 'isobaric' in dim.lower():
        level_dim = dim
        break

def extract_box_average(lat, lon, date, box_size=0.5):
    """
    Extract atmospheric conditions averaged over a box around a location
    
    Parameters:
    -----------
    lat, lon : float
        Center location
    date : datetime
        Date to extract
    box_size : float
        Half-width of box in degrees (0.5° ≈ 55 km)
    
    Returns:
    --------
    dict with atmospheric variables, or None if data not available
    """

    lat_min = lat - box_size
    lat_max = lat + box_size
    lon_min = lon - box_size
    lon_max = lon + box_size
    
    date_dt64 = pd.to_datetime(date).tz_localize(None)
    # Use 00Z of the NEXT day (hail reports are dated on the day they occur,
    # but 00Z is technically the next calendar day in UTC)
    target_time = date_dt64 + pd.Timedelta(days=1)  # 00Z next day
    subset_single_time = ds_single.sel(valid_time=target_time, method='nearest')
    subset_winds_time = ds_winds.sel(valid_time=target_time, method='nearest')

    subset_single = subset_single_time.sel(
        latitude=slice(lat_max, lat_min), 
        longitude=slice(lon_min, lon_max)
    )
    
    subset_winds = subset_winds_time.sel(
        latitude=slice(lat_max, lat_min),
        longitude=slice(lon_min, lon_max)
    )
    
    # Check if we got any data
    if subset_single.dims.get('latitude', 0) == 0 or subset_single.dims.get('longitude', 0) == 0:
        return None
    
    # Average over the box
    box_mean_single = subset_single.mean(dim=['latitude', 'longitude'])
    box_mean_winds = subset_winds.mean(dim=['latitude', 'longitude'])
    
    cape = float(box_mean_single[cape_var].values)
    
    if freezing_var:
        freezing_level = float(box_mean_single[freezing_var].values)
    else:
        freezing_level = np.nan
    
    u_500 = float(box_mean_winds.sel({level_dim: 500})['u'].values)
    v_500 = float(box_mean_winds.sel({level_dim: 500})['v'].values)
    u_850 = float(box_mean_winds.sel({level_dim: 850})['u'].values)
    v_850 = float(box_mean_winds.sel({level_dim: 850})['v'].values)
    shear = np.sqrt((u_500 - u_850)**2 + (v_500 - v_850)**2)
    
    return {
        'cape': cape,
        'shear': shear,
        'freezing_level': freezing_level
    }
        
# Hail reports
hail_df = pd.read_csv('data/hail_reports.csv')
hail_df['date'] = pd.to_datetime(hail_df['date'])

hail_samples = []
failed_count = 0

for idx, row in hail_df.iterrows():
    atmos = extract_box_average(row['slat'], row['slon'], row['date'])
    
    if atmos is not None:
        hail_samples.append({
            'date': row['date'],
            'lat': row['slat'],
            'lon': row['slon'],
            'hail_occurred': 1,
            **atmos
        })
    else:
        failed_count += 1

# 15 non hail samples per day to balance dataset
hail_samples_df = pd.DataFrame(hail_samples)
hail_dates = hail_df['date'].unique()
non_hail_samples = []
samples_per_day = 15 

LAT_MIN, LAT_MAX = 33, 42
LON_MIN, LON_MAX = -103, -94

for date_idx, date in enumerate(hail_dates):
    hail_locs_today = hail_df[hail_df['date'] == date][['slat', 'slon']].values
    
    # Generate random non hail locations
    attempts = 0
    generated = 0
    
    while generated < samples_per_day and attempts < samples_per_day * 20:
        attempts += 1
        
        # Random location in hail alley
        rand_lat = np.random.uniform(LAT_MIN, LAT_MAX)
        rand_lon = np.random.uniform(LON_MIN, LON_MAX)
        
        # Check distance from all hail reports on this day
        # Only keep if 50km or more from any hail report
        distances = np.sqrt(
            (hail_locs_today[:, 0] - rand_lat)**2 + 
            (hail_locs_today[:, 1] - rand_lon)**2
        )
        
        if distances.min() > 0.5: 
            atmos = extract_box_average(rand_lat, rand_lon, date)
            
            if atmos is not None:
                non_hail_samples.append({
                    'date': date,
                    'lat': rand_lat,
                    'lon': rand_lon,
                    'hail_occurred': 0,
                    **atmos
                })
                generated += 1

non_hail_samples_df = pd.DataFrame(non_hail_samples)

# Combine hail and non hail samples
final_df = pd.concat([hail_samples_df, non_hail_samples_df], ignore_index=True)

# Check if we have any data
if len(final_df) == 0:
    print(" You have no data!!")

# Remove hail with CAPE = 0
cape_zero_hail = ((final_df['hail_occurred'] == 1) & (final_df['cape'] == 0)).sum()
print("Found", cape_zero_hail, "hail samples with CAPE = 0. Will be removed.")

final_df = final_df[~((final_df['hail_occurred'] == 1) & (final_df['cape'] == 0))]
final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
final_df = final_df.dropna()
final_df.to_csv('data/final_dataset_boxed.csv', index=False)

print(final_df.head(10))