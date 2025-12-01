# ATMS 523 Final Project


# Author: Dara Procell
# Date: December 12, 2025
# Description: This script analyzes severe weather data to predict hail occurrences using logistic regression.


import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve, precision_recall_curve, auc
import seaborn as sns

# Set up directories
import os
os.makedirs('data', exist_ok=True)
os.makedirs('figures', exist_ok=True)


# Download hail reports
url = "https://www.spc.noaa.gov/wcm/data/1950-2023_actual_tornadoes.csv"  
# Note: SPC has separate hail files, use: https://www.spc.noaa.gov/wcm/data/1955-2023_hail.csv

# Load hail data
hail_df = pd.read_csv("https://www.spc.noaa.gov/wcm/data/1955-2023_hail.csv")

# Filter for your region and time period
hail_df['date'] = pd.to_datetime(hail_df[['yr', 'mo', 'dy']])

# Filter: Hail Alley bounds (33-42°N, -103 to -94°W)
hail_df = hail_df[
    (hail_df['slat'] >= 33) & (hail_df['slat'] <= 42) &
    (hail_df['slon'] >= -103) & (hail_df['slon'] <= -94)
]

# Filter: 2020-2021, April-September
hail_df = hail_df[
    (hail_df['yr'].isin([2020, 2021])) &
    (hail_df['mo'] >= 4) & (hail_df['mo'] <= 9)
]

# Filter: Significant hail only (≥1 inch)
hail_df = hail_df[hail_df['sz'] >= 1.0]

# Aggregate to daily binary: any hail that day = 1
daily_hail = hail_df.groupby('date').size().reset_index(name='hail_reports')
daily_hail['hail_occurred'] = 1


### **Step 3: Set Up Copernicus CDS API**

# You'll need to register and get an API key:

# 1. Go to: https://cds.climate.copernicus.eu/
# 2. Register for free account
# 3. Get your API key from your profile
# 4. Create a file `~/.cdsapirc` with:
# ```
# url: https://cds.climate.copernicus.eu/api/v2
# key: YOUR_UID:YOUR_API_KEY