import requests
import pandas as pd
from datetime import datetime, timedelta
import os

DATA_PATH = "huge_turkey_dataset.csv"

def fetch_afad_micro():
    print("AFAD Canlı Veri Servisine Bağlanılıyor (Mikro-Sismisite)...")
    # AFAD Public Event API (Last 20,000 events)
    url = "https://deprem.afad.gov.tr/EventData/GetEventsByFilter"
    
    # Payload for the last 6 months (as a sample of micro-data)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)
    
    payload = {
        "start": start_date.strftime("%Y-%m-%dT%H:%M:%S"),
        "end": end_date.strftime("%Y-%m-%dT%H:%M:%S"),
        "minMag": 0.5, # MIKRO-DEPREMLER (Çok hassas)
        "maxMag": 10.0,
        "minLat": 34.0, "maxLat": 43.0,
        "minLon": 25.0, "maxLon": 45.0
    }
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        data = response.json()
        print(f"AFAD'dan {len(data)} adet yeni mikro-deprem verisi çekildi.")
        
        afad_df = pd.DataFrame(data)
        # Rename columns to match our dataset
        # AFAD columns: eventId, eventDate, latitude, longitude, depth, magnitude, type, location
        new_df = pd.DataFrame({
            'Date': afad_df['eventDate'],
            'Time': afad_df['eventDate'],
            'Latitude': afad_df['latitude'],
            'Longitude': afad_df['longitude'],
            'Depth': afad_df['depth'],
            'Magnitude': afad_df['magnitude']
        })
        
        if os.path.exists(DATA_PATH):
            old_df = pd.read_csv(DATA_PATH)
            merged_df = pd.concat([old_df, new_df]).drop_duplicates(subset=['Latitude', 'Longitude', 'Date'])
            merged_df.to_csv(DATA_PATH, index=False)
            print(f"Veri seti güncellendi. Toplam kayıt: {len(merged_df)}")
        else:
            new_df.to_csv(DATA_PATH, index=False)
            
    except Exception as e:
        print(f"AFAD API Hatası: {e}")

if __name__ == "__main__":
    fetch_afad_micro()
