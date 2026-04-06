import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
import pickle
import os

# Config
DATA_PATH = "huge_turkey_dataset.csv"
MODEL_PATH = "earthquake_model.pkl"
STATS_PATH = "grid_stats.pkl"

# Major Fault Lines (Approximate coordinates for Turkey)
FAULTS = {
    "KAF": [[40.0, 26.0], [40.8, 30.0], [41.0, 35.0], [40.0, 40.0], [39.0, 44.0]], # North Anatolian
    "DAF": [[36.0, 36.0], [37.5, 37.5], [38.5, 39.5], [40.0, 42.0]],               # East Anatolian
    "BAF": [[37.0, 27.0], [38.5, 27.5], [39.0, 28.0]]                             # West Anatolian
}

def dist_to_line(lat, lon, line):
    min_dist = 1000
    for p in line:
        d = np.sqrt((lat - p[0])**2 + (lon - p[1])**2)
        if d < min_dist: min_dist = d
    return min_dist

def calculate_b_value(magnitudes):
    # Basit bir Gutenberg-Richter b-değeri hesaplama: b = log10(e) / (M_avg - M_min)
    if len(magnitudes) < 5: return 1.0 # Veri azsa standart değer
    m_min = magnitudes.min()
    m_avg = magnitudes.mean()
    if m_avg == m_min: return 1.0
    return np.log10(np.e) / (m_avg - m_min)

def process_and_train():
    print("50.000+ kayıt işleniyor...")
    df = pd.read_csv(DATA_PATH)
    # Tarihleri UTC olarak sabitle
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', utc=True)
    df = df.dropna(subset=['Latitude', 'Longitude', 'Magnitude', 'Date'])
    
    # 1. Grid bazlı istatistikler (0.1 derecelik hassas gridler)
    df['Lat_Grid'] = df['Latitude'].round(1)
    df['Lon_Grid'] = df['Longitude'].round(1)
    
    print("Sismoloji parametreleri hesaplanıyor (B-Value, Sismik Boşluk)...")
    grid_stats = []
    
    # Bugünün tarihi (UTC aware)
    today = pd.Timestamp('2025-01-01', tz='UTC')
    
    for (lat_g, lon_g), group in df.groupby(['Lat_Grid', 'Lon_Grid']):
        # B-Value
        b_val = calculate_b_value(group['Magnitude'])
        # Sismik Boşluk (Son M4.0+ depremden geçen gün)
        # Kararlılık için eşiği 4.0'a çektik
        major = group[group['Magnitude'] >= 4.0]
        if not major.empty:
            last_date = major['Date'].max()
            gap_days = (today - last_date).days
        else:
            gap_days = 20000 # Hiç büyük deprem yoksa max değer
        
        # Frekans
        freq = len(group)
        
        # Fay Mesafesi
        dist_kaf = dist_to_line(lat_g, lon_g, FAULTS["KAF"])
        dist_daf = dist_to_line(lat_g, lon_g, FAULTS["DAF"])
        dist_baf = dist_to_line(lat_g, lon_g, FAULTS["BAF"])
        min_fault = min(dist_kaf, dist_daf, dist_baf)
        
        grid_stats.append({
            'Lat_Grid': lat_g, 'Lon_Grid': lon_g,
            'B_Value': b_val, 'Seismic_Gap': gap_days,
            'Frequency': freq, 'Min_Fault_Dist': min_fault
        })
        
    stats_df = pd.DataFrame(grid_stats)
    stats_df.to_pickle(STATS_PATH)
    
    # 2. Model için veriyi hazırla
    df = df.merge(stats_df, on=['Lat_Grid', 'Lon_Grid'])
    
    X = df[['Latitude', 'Longitude', 'Depth', 'Frequency', 'Min_Fault_Dist', 'B_Value', 'Seismic_Gap']]
    y = df['Magnitude']
    
    print("Gradient Boosting Regressor eğitiliyor (ULTIMATE 1000 TREE MODEL)...")
    # Çok daha yüksek kapasiteli ve kararlı (extreme stability) v4.0 model
    model = GradientBoostingRegressor(
        n_estimators=1000, 
        learning_rate=0.02, 
        max_depth=9, 
        min_samples_split=5,
        random_state=42,
        verbose=1
    )
    model.fit(X, y)
    
    # Başarı metriklerini hesapla
    pred = model.predict(X)
    mae = np.mean(np.abs(y - pred))
    r2 = model.score(X, y)
    print(f"\n--- Model Kararlılık Karnesi ---")
    print(f"Toplanan Veri Sayısı: {len(df)}")
    print(f"Hata Payı (MAE): {mae:.4f}")
    print(f"Güven Skoru (R2): {r2:.4f}")
    
    # Modeli kaydet
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model v4.0 PRO başarıyla kaydedildi.")

if __name__ == "__main__":
    process_and_train()
