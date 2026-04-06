from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
import pickle
import os
import numpy as np
from scipy.spatial import KDTree

# Absolute directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = FastAPI(title="DepremVizyon 5.0 Ultimate")

static_dir = os.path.join(BASE_DIR, "static")
if not os.path.exists(static_dir): os.makedirs(static_dir)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

MODEL_PATH = os.path.join(BASE_DIR, "earthquake_model.pkl")
STATS_PATH = os.path.join(BASE_DIR, "grid_stats.pkl")
DATA_PATH = os.path.join(BASE_DIR, "huge_turkey_dataset.csv")

# Global resources
model = None
grid_stats = None
spatial_index = None

def load_resources():
    global model, grid_stats, spatial_index
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f: model = pickle.load(f)
    if os.path.exists(STATS_PATH):
        grid_stats = pd.read_pickle(STATS_PATH)
        # Create a KDTree for extremely fast spatial lookup (Interpolation)
        if grid_stats is not None:
            coords = grid_stats[['Lat_Grid', 'Lon_Grid']].values
            spatial_index = KDTree(coords)

load_resources()

class PredictionRequest(BaseModel):
    latitude: float
    longitude: float

@app.get("/")
def read_index():
    path = os.path.join(BASE_DIR, "index.html")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f: return HTMLResponse(content=f.read())
    return "Index not found."

# Major Fault Lines in Turkey
FAULTS_GEO = {
    "Kuzey Anadolu Fay Hattı (KAF)": [[39.5, 26.5], [41.0, 31.0], [41.0, 36.0], [39.5, 41.0], [39.0, 44.0]],
    "Doğu Anadolu Fay Hattı (DAF)": [[36.5, 36.0], [38.0, 38.5], [39.0, 41.0]],
    "Batı Anadolu Fay Hattı (BAF)": [[37.0, 27.0], [38.5, 27.5], [40.0, 27.0]]
}

@app.get("/faults")
def get_faults():
    return FAULTS_GEO

@app.get("/historical")
def get_historical():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce', utc=True)
        recent = df.sort_values(by='Date', ascending=False).head(200)
        return recent.fillna(0).to_dict(orient='records')
    return []

@app.get("/heatmap")
def get_heatmap():
    if grid_stats is None: return []
    max_f = grid_stats['Frequency'].max() or 1
    stats = grid_stats.copy()
    # Risk-weighted intensity for better visualization
    stats['Intensity'] = (stats['Frequency'] / max_f) * 0.5 + (1.0 / stats['B_Value'].clip(0.1, 2.0)) * 0.5
    return stats[['Lat_Grid', 'Lon_Grid', 'Intensity']].values.tolist()

@app.post("/predict")
def predict(req: PredictionRequest):
    if model is None or grid_stats is None or spatial_index is None: 
        raise HTTPException(status_code=500, detail="Resources not loaded")
    
    # AKILLI MEKANSAL İNTERPOLASYON (Smart Spatial Interpolation)
    # Tıklanan noktaya en yakın 5 grid verisini bul ve mesafe ağırlıklı ortalama al
    distances, indices = spatial_index.query([req.latitude, req.longitude], k=5)
    
    # Ağırlıklandırma (1/d)
    weights = 1.0 / np.maximum(distances, 0.01)
    weights /= weights.sum()
    
    neighbors = grid_stats.iloc[indices]
    
    # Özellikleri interpolasyonla hesapla
    freq = np.sum(neighbors['Frequency'].values * weights)
    min_f = np.sum(neighbors['Min_Fault_Dist'].values * weights)
    b_val = np.sum(neighbors['B_Value'].values * weights)
    gap = np.sum(neighbors['Seismic_Gap'].values * weights)
    
    # Model Tahmini: [Lat, Lon, Depth, Freq, Min_Fault, B_Val, Gap]
    features = [[req.latitude, req.longitude, 10.0, freq, min_f, b_val, gap]]
    pred = model.predict(features)[0]
    
    # Kararlı Risk Analizi (v5.0 Interpolated)
    norm_f = min(10, freq / 8.0)
    norm_m = min(10, (pred / 8.5) * 10)
    norm_b = min(10, (1.1 / max(0.2, b_val)) * 6)
    norm_g = min(10, (gap / 18000.0) * 10)
    
    risk = round((norm_f*0.2 + norm_m*0.3 + norm_b*0.2 + norm_g*0.3), 1)
    
    return {
        "latitude": req.latitude, "longitude": req.longitude,
        "predicted_magnitude": round(float(pred), 2),
        "risk_score": risk,
        "frequency": round(float(freq), 1),
        "b_value": round(float(b_val), 2),
        "seismic_gap": int(gap),
        "fault_dist": round(float(min_f), 4),
        "interpolation_confidence": round(float(100 - (distances[0]*50)), 1) # Yakınlık güven yüzdesi
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
