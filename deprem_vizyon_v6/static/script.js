const map = L.map('map', { zoomControl: false }).setView([38.9, 35.2], 6.5);

// Tiles
const darkTiles = L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    attribution: '&copy; CARTO',
    subdomains: 'abcd',
    maxZoom: 20
}).addTo(map);

const osmTiles = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; OpenStreetMap'
});

let heatmapLayer = null;
let currentMarker = null;
let historicalLayer = L.layerGroup();
let faultLayer = L.layerGroup().addTo(map);

// UI Elements
const latDisplay = document.querySelector('#lat-display .value');
const lonDisplay = document.querySelector('#lon-display .value');
const magVal = document.querySelector('#mag-val');
const riskVal = document.querySelector('#risk-val');
const riskFill = document.querySelector('#risk-fill');
const gapVal = document.getElementById('gap-val');
const bVal = document.getElementById('b-val');
const freqVal = document.getElementById('freq-val');
const faultVal = document.getElementById('fault-val');
const recentList = document.querySelector('#recent-list');

async function initApp() {
    setTimeout(() => { map.invalidateSize(); }, 500);
    loadHeatmap();
    loadRecentData();
    loadFaultLines();
}

async function loadFaultLines() {
    try {
        const response = await fetch('/faults');
        const data = await response.json();
        faultLayer.clearLayers();
        
        for (const [name, coords] of Object.entries(data)) {
            L.polyline(coords, {
                color: '#ff5252',
                weight: 3,
                opacity: 0.8,
                dashArray: '5, 10'
            }).bindTooltip(name, { sticky: true }).addTo(faultLayer);
        }
    } catch (e) { console.error("Fault loading error:", e); }
}

async function loadHeatmap() {
    try {
        const response = await fetch('/heatmap');
        const data = await response.json();
        if (Array.isArray(data) && data.length > 0) {
            heatmapLayer = L.heatLayer(data, {
                radius: 18, blur: 12, max: 0.8,
                gradient: { 0.2: 'cyan', 0.5: 'yellow', 0.8: 'orange', 1: 'red' }
            }).addTo(map);
        }
    } catch (e) { console.error("Heatmap loading error:", e); }
}

async function loadRecentData() {
    try {
        const response = await fetch('/historical');
        const data = await response.json();
        
        recentList.innerHTML = '';
        data.slice(0, 30).forEach(dp => {
            const item = document.createElement('li');
            item.className = 'recent-item';
            const dateStr = dp.Date ? dp.Date.split('T')[0] : (dp.Date || 'N/A');
            item.innerHTML = `
                <div class="recent-info">
                    <div class="recent-date">${dateStr}</div>
                    <div class="recent-loc">${dp.Latitude.toFixed(2)}, ${dp.Longitude.toFixed(2)}</div>
                </div>
                <div class="recent-mag">${dp.Magnitude.toFixed(1)}</div>
            `;
            recentList.appendChild(item);
        });

        historicalLayer.clearLayers();
        data.forEach(dp => {
            L.circleMarker([dp.Latitude, dp.Longitude], {
                radius: dp.Magnitude * 1.5,
                fillColor: "#ff5252",
                color: "transparent",
                fillOpacity: 0.12
            }).addTo(historicalLayer);
        });
    } catch (e) { console.error("Recent data error:", e); }
}

map.on('click', async (e) => {
    const { lat, lng } = e.latlng;
    if (currentMarker) map.removeLayer(currentMarker);
    currentMarker = L.marker([lat, lng]).addTo(map);
    latDisplay.textContent = lat.toFixed(4);
    lonDisplay.textContent = lng.toFixed(4);
    
    try {
        const res = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ latitude: lat, longitude: lng })
        });
        const data = await res.json();
        updateUI(data);
    } catch (e) { console.error("Prediction error:", e); }
});

function updateUI(data) {
    animateValue(magVal, data.predicted_magnitude, 1.0);
    animateValue(riskVal, data.risk_score, 1.0);
    
    // Advanced Stats (Interpolated)
    gapVal.textContent = data.seismic_gap;
    bVal.textContent = data.b_value;
    freqVal.textContent = data.frequency;
    faultVal.textContent = data.fault_dist.toFixed(3);
    
    // Confidence Meter Update
    const confPct = document.getElementById('conf-pct');
    const confFill = document.getElementById('conf-fill');
    if (confPct && data.interpolation_confidence) {
        confPct.textContent = data.interpolation_confidence;
        confFill.style.width = `${data.interpolation_confidence}%`;
        confFill.style.background = data.interpolation_confidence > 70 ? '#00e5ff' : '#ff9800';
    }
    
    const riskPct = (data.risk_score / 10) * 100;
    riskFill.style.width = `${riskPct}%`;
    riskFill.style.background = getRiskColor(data.risk_score);
    
    // Style adjustments
    gapVal.style.color = data.seismic_gap > 5000 ? '#f44336' : '#4caf50';
    bVal.style.color = data.b_value < 0.9 ? '#f44336' : '#2196f3';
}

function animateValue(obj, end, duration) {
    let start = parseFloat(obj.textContent) || 0;
    let startTime = null;
    const step = (now) => {
        if (!startTime) startTime = now;
        const progress = Math.min((now - startTime) / (duration * 1000), 1);
        obj.innerHTML = (progress * (end - start) + start).toFixed(1);
        if (progress < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
}

function getRiskColor(risk) {
    if (risk < 3.5) return "#4caf50";
    if (risk < 6.5) return "#ffeb3b";
    if (risk < 8.5) return "#ff9800";
    return "#f44336";
}

document.getElementById('toggle-heatmap').addEventListener('click', function() {
    if (heatmapLayer && map.hasLayer(heatmapLayer)) {
        map.removeLayer(heatmapLayer);
        this.classList.remove('active');
    } else if (heatmapLayer) {
        heatmapLayer.addTo(map);
        this.classList.add('active');
    }
});

document.getElementById('toggle-faults').addEventListener('click', function() {
    if (map.hasLayer(faultLayer)) {
        map.removeLayer(faultLayer);
        this.classList.remove('active');
    } else {
        faultLayer.addTo(map);
        this.classList.add('active');
    }
});

document.getElementById('show-historical').addEventListener('click', function() {
    if (map.hasLayer(historicalLayer)) {
        map.removeLayer(historicalLayer);
        this.classList.remove('active');
    } else {
        historicalLayer.addTo(map);
        this.classList.add('active');
    }
});

// Run
initApp();
