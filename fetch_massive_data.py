import requests
import pandas as pd
import os
import time
import io

def fetch_chunk(start, end):
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    params = {
        "format": "csv",
        "starttime": f"{start}-01-01",
        "endtime": f"{end}-12-31",
        "minlatitude": 32.0,  # Genişletilmiş Bölge (Yunanistan'dan İran'a)
        "maxlatitude": 45.0,
        "minlongitude": 20.0,
        "maxlongitude": 55.0,
        "minmagnitude": 2.5
    }
    try:
        r = requests.get(url, params=params, timeout=60)
        if r.status_code == 200:
            return pd.read_csv(io.StringIO(r.text))
        else:
            return pd.DataFrame()
    except Exception as e:
        print(f"Error for {start}-{end}: {e}")
        return pd.DataFrame()

# Genişletilmiş Zaman Dilimleri (1900-2025)
years = [
    (1900, 1960), (1961, 1980), (1981, 1995), 
    (1996, 2005), (2006, 2015), (2016, 2025)
]
all_dfs = []

# Also include the local database.csv
local_df = pd.read_csv(r"C:\Users\Ersel\Downloads\archive\database.csv")
local_df = local_df[local_df['Type'] == 'Earthquake']

for start, end in years:
    print(f"Fetching {start}-{end}...")
    df = fetch_chunk(start, end)
    if not df.empty:
        df = df.rename(columns={'time': 'Date', 'latitude': 'Latitude', 'longitude': 'Longitude', 'depth': 'Depth', 'mag': 'Magnitude'})
        # Gerekli sütunlar varsa ekle
        cols = ['Date', 'Latitude', 'Longitude', 'Depth', 'Magnitude']
        if all(c in df.columns for c in cols):
            all_dfs.append(df[cols])
    time.sleep(1)

if all_dfs:
    final_df = pd.concat(all_dfs)
    # database.csv ile birleştir
    try:
        local_df = pd.read_csv(r"C:\Users\Ersel\Downloads\archive\database.csv")
        local_v = local_df[['Date', 'Latitude', 'Longitude', 'Depth', 'Magnitude']]
        final_df = pd.concat([final_df, local_v])
    except: pass
    
    final_df = final_df.drop_duplicates(subset=['Latitude', 'Longitude', 'Magnitude'])
    final_df.to_csv("huge_turkey_dataset.csv", index=False)
    print(f"Successfully created huge_turkey_dataset.csv with {len(final_df)} records.")
else:
    print("No data fetched.")
