"""
Download and preprocess hail and ERA5 data with box averaging approach
Extracts atmospheric conditions at time of daily maximum CAPE
"""

import pandas as pd
import numpy as np
import xarray as xr
import cdsapi
import os

os.makedirs('data', exist_ok=True)
os.makedirs('data/era5_raw', exist_ok=True)

# Download hail reports for 2020 and 2021
hail_dfs = []
for year in [2020, 2021]:
    url = f"https://www.spc.noaa.gov/wcm/data/{year}_hail.csv"
    df = pd.read_csv(url)
    hail_dfs.append(df)

hail_df = pd.concat(hail_dfs, ignore_index=True)
hail_df['date'] = pd.to_datetime(hail_df['date'])
hail_df = hail_df[
    (hail_df['slat'] >= 33) & (hail_df['slat'] <= 42) &
    (hail_df['slon'] >= -103) & (hail_df['slon'] <= -94)
]
hail_df = hail_df[(hail_df['mo'] >= 4) & (hail_df['mo'] <= 9)]
hail_df = hail_df[hail_df['mag'] >= 1.0]
hail_df.to_csv('data/hail_reports.csv', index=False)

c = cdsapi.Client()
region = [42, -103, 33, -94]

# Download all hours, not just 00Z
for year in [2020, 2021]:
    for month in range(4, 10):
        filename = f'data/era5_raw/era5_single_{year}_{month:02d}.nc'
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'variable': ['convective_available_potential_energy', 'zero_degree_level', 'total_column_water_vapour'],
                'year': str(year),
                'month': f'{month:02d}',
                'day': [f'{d:02d}' for d in range(1, 32)],
                'time': [f'{h:02d}:00' for h in range(24)],
                'area': region,
                'grid': [1.0, 1.0],
                'format': 'netcdf',
            },
            filename
        )

for year in [2020, 2021]:
    for month in range(4, 10):
        filename = f'data/era5_raw/era5_winds_{year}_{month:02d}.nc'
        c.retrieve(
            'reanalysis-era5-pressure-levels',
            {
                'product_type': 'reanalysis',
                'variable': ['u_component_of_wind', 'v_component_of_wind'],
                'pressure_level': ['500', '850'],
                'year': str(year),
                'month': f'{month:02d}',
                'day': [f'{d:02d}' for d in range(1, 32)],
                'time': [f'{h:02d}:00' for h in range(24)],
                'area': region,
                'grid': [1.0, 1.0],
                'format': 'netcdf',
            },
            filename
        )

single_files = [f'data/era5_raw/era5_single_{year}_{month:02d}.nc' 
                for year in [2020, 2021] for month in range(4, 10)]
winds_files = [f'data/era5_raw/era5_winds_{year}_{month:02d}.nc' 
               for year in [2020, 2021] for month in range(4, 10)]

ds_single = xr.open_mfdataset(single_files, combine='by_coords')
ds_winds = xr.open_mfdataset(winds_files, combine='by_coords')

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
    """Extract atmospheric conditions at time of daily maximum CAPE"""
    lat_min = lat - box_size
    lat_max = lat + box_size
    lon_min = lon - box_size
    lon_max = lon + box_size
    
    date_start = pd.to_datetime(date).replace(hour=0, minute=0, second=0, microsecond=0)
    date_end = date_start + pd.Timedelta(days=1)
    
    try:
        subset_day = ds_single.sel(valid_time=slice(date_start, date_end))
        subset_day_spatial = subset_day.sel(
            latitude=slice(lat_max, lat_min),
            longitude=slice(lon_min, lon_max)
        )
        
        if subset_day_spatial.dims.get('latitude', 0) == 0:
            return None
        
        box_mean_day = subset_day_spatial.mean(dim=['latitude', 'longitude'])
        cape_timeseries = box_mean_day[cape_var].values
        
        if len(cape_timeseries) == 0 or np.all(np.isnan(cape_timeseries)):
            return None
        
        max_cape_idx = np.nanargmax(cape_timeseries)
        target_time = pd.to_datetime(box_mean_day.valid_time.values[max_cape_idx])
        
    except Exception:
        return None
    
    try:
        subset_single_time = ds_single.sel(valid_time=target_time, method='nearest')
        subset_winds_time = ds_winds.sel(valid_time=target_time, method='nearest')
    except Exception:
        return None

    subset_single = subset_single_time.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))
    subset_winds = subset_winds_time.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))
    
    if subset_single.dims.get('latitude', 0) == 0 or subset_single.dims.get('longitude', 0) == 0:
        return None
    
    box_mean_single = subset_single.mean(dim=['latitude', 'longitude'])
    box_mean_winds = subset_winds.mean(dim=['latitude', 'longitude'])
    
    cape = float(box_mean_single[cape_var].values)
    freezing_level = float(box_mean_single[freezing_var].values) if freezing_var else np.nan
    
    u_500 = float(box_mean_winds.sel({level_dim: 500})['u'].values)
    v_500 = float(box_mean_winds.sel({level_dim: 500})['v'].values)
    u_850 = float(box_mean_winds.sel({level_dim: 850})['u'].values)
    v_850 = float(box_mean_winds.sel({level_dim: 850})['v'].values)
    shear = np.sqrt((u_500 - u_850)**2 + (v_500 - v_850)**2)
    
    return {'cape': cape, 'shear': shear, 'freezing_level': freezing_level}
        
# Extract hail samples
hail_df = pd.read_csv('data/hail_reports.csv')
hail_df['date'] = pd.to_datetime(hail_df['date'])

hail_samples = []
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

# Generate non-hail samples
hail_dates = hail_df['date'].unique()
non_hail_samples = []
LAT_MIN, LAT_MAX = 33, 42
LON_MIN, LON_MAX = -103, -94

for date in hail_dates:
    hail_locs_today = hail_df[hail_df['date'] == date][['slat', 'slon']].values
    generated = 0
    attempts = 0
    
    while generated < 15 and attempts < 300:
        attempts += 1
        rand_lat = np.random.uniform(LAT_MIN, LAT_MAX)
        rand_lon = np.random.uniform(LON_MIN, LON_MAX)
        
        distances = np.sqrt((hail_locs_today[:, 0] - rand_lat)**2 + (hail_locs_today[:, 1] - rand_lon)**2)
        
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

final_df = pd.concat([pd.DataFrame(hail_samples), pd.DataFrame(non_hail_samples)], ignore_index=True)
final_df = final_df[~((final_df['hail_occurred'] == 1) & (final_df['cape'] == 0))]
final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
final_df = final_df.dropna()
final_df.to_csv('data/final_dataset_daily_max_cape.csv', index=False)
