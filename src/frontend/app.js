let maps = {};
const ALGORITHMS = ['ALNS', 'DQN', 'DQN+ALNS', 'OR-Tools'];
const MAP_IDS = { 'ALNS': 'map-alns', 'DQN': 'map-dqn', 'DQN+ALNS': 'map-hybrid', 'OR-Tools': 'map-ortools' };
let currentSolutionData = [];

// Coordinate Transformation (Solomon 100x100 -> HCMC Lat/Lng)
const HCMC_CENTER = [10.762622, 106.660172];
const SCALE_FACTOR = 0.0008; // Adjust to spread points appropriately

function transformCoords(x, y) {
    return [
        HCMC_CENTER[0] + (x - 50) * SCALE_FACTOR, 
        HCMC_CENTER[1] + (y - 50) * SCALE_FACTOR
    ];
}

document.addEventListener('DOMContentLoaded', async () => {
    initMaps();
    await loadInstanceList();
    setupEvents();
});

function initMaps() {
    ALGORITHMS.forEach(algo => {
        const id = MAP_IDS[algo];
        if(!document.getElementById(id)) return;
        
        // CartoDB Positron (Minimalist B&W)
        const map = L.map(id, { zoomControl: false }).setView(HCMC_CENTER, 13);
        L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
            attribution: '', maxZoom: 20
        }).addTo(map);
        
        maps[algo] = map;
    });
}

async function loadInstanceList() {
    const select = document.getElementById('instance-select');
    try {
        const res = await fetch('/api/instances');
        const list = await res.json();
        select.innerHTML = '';
        list.forEach(item => {
            const opt = document.createElement('option');
            opt.value = item;
            opt.textContent = item;
            select.appendChild(opt);
        });
    } catch (e) {
        setStatus('Failed to load instances', true);
    }
}

function setupEvents() {
    document.getElementById('btn-load').addEventListener('click', loadInstancePreview);
    document.getElementById('btn-run').addEventListener('click', runBenchmark);
    document.getElementById('btn-clear').addEventListener('click', clearAllMaps);
    
    document.getElementById('btn-toggle-table').addEventListener('click', () => {
        renderTable();
        document.getElementById('result-modal').style.display = 'flex';
    });
    
    document.querySelector('.close-modal').addEventListener('click', () => {
        document.getElementById('result-modal').style.display = 'none';
    });
}

function clearAllMaps() {
    ALGORITHMS.forEach(algo => {
        const map = maps[algo];
        map.eachLayer(layer => {
            if (layer instanceof L.Path || layer instanceof L.Marker || layer instanceof L.CircleMarker) {
                map.removeLayer(layer);
            }
        });
        document.getElementById(`meta-${getMetaId(algo)}`).textContent = '';
    });
    document.getElementById('btn-toggle-table').style.display = 'none';
    setStatus('Maps cleared');
}

// --- NEW: Load Instance Preview ---
async function loadInstancePreview() {
    const instance = document.getElementById('instance-select').value;
    if (!instance) return setStatus('Select an instance first', true);

    setStatus('Loading map data...');
    clearAllMaps(); // Clear previous data

    try {
        const res = await fetch('/api/load_instance', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ instance: instance })
        });
        
        if (!res.ok) throw new Error('API Error');
        const data = await res.json();

        // Draw dots on ALL maps
        ALGORITHMS.forEach(algo => {
            const map = maps[algo];
            drawNodes(map, data.depot, data.customers);
        });

        setStatus(`Loaded ${instance} (${data.customers.length} customers)`);
    } catch (e) {
        setStatus('Error loading instance: ' + e.message, true);
    }
}

function drawNodes(map, depot, customers) {
    const depotPos = transformCoords(depot.lat, depot.lng);
    const bounds = L.latLngBounds([depotPos]);

    // Draw Depot
    L.circleMarker(depotPos, {
        radius: 6, color: '#000', fillColor: '#000', fillOpacity: 1
    }).addTo(map).bindPopup('<b>DEPOT</b>');

    // Draw Customers
    customers.forEach(c => {
        const pos = transformCoords(c.lat, c.lng);
        bounds.extend(pos);
        L.circleMarker(pos, {
            radius: 3, color: '#888', fillColor: '#fff', weight: 1, fillOpacity: 1
        }).addTo(map).bindPopup(`Cust: ${c.id}<br>Demand: ${c.demand}`);
    });

    map.fitBounds(bounds, { padding: [30, 30] });
}

// --- Run Benchmark ---
async function runBenchmark() {
    const instance = document.getElementById('instance-select').value;
    const maxVehicles = parseInt(document.getElementById('max-vehicles').value);
    const selectedAlgos = Array.from(document.querySelectorAll('input[type="checkbox"]:checked')).map(cb => cb.value);

    if (!instance) return setStatus('Please select an instance', true);
    if (!selectedAlgos.length) return setStatus('Select algorithms', true);

    const btn = document.getElementById('btn-run');
    btn.disabled = true;
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> RUNNING...';
    setStatus('Solving...');

    try {
        const res = await fetch('/api/run_comparison', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                instance: instance,
                algorithms: selectedAlgos,
                max_vehicles: maxVehicles
            })
        });

        const data = await res.json();
        currentSolutionData = data.solutions;
        
        // Clear maps to redraw solutions (or keep dots and just draw lines? Let's redraw for clean layers)
        clearAllMaps();

        data.solutions.forEach(sol => {
            if (sol.error) {
                console.error(sol.algorithm, sol.error);
                return;
            }
            drawSolution(sol);
        });

        document.getElementById('btn-toggle-table').style.display = 'block';
        setStatus('Benchmark Completed');
    } catch (e) {
        setStatus('Execution failed: ' + e.message, true);
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="fa-solid fa-play"></i> RUN BENCHMARK';
    }
}

function drawSolution(sol) {
    const map = maps[sol.algorithm];
    if (!map) return;

    const depotPos = transformCoords(sol.depot.lat, sol.depot.lng);
    const bounds = L.latLngBounds([depotPos]);

    // Redraw Depot
    L.circleMarker(depotPos, { radius: 6, color: '#000', fillColor: '#000', fillOpacity: 1 }).addTo(map);

    const colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#f032e6'];

    sol.routes.forEach((route, i) => {
        const color = colors[i % colors.length];
        const latlngs = [depotPos]; // Start at depot

        route.nodes.forEach(node => {
            const pos = transformCoords(node.lat, node.lng);
            latlngs.push(pos);
            bounds.extend(pos);
            
            // Draw Customer Node (Colored)
            L.circleMarker(pos, {
                radius: 4, color: color, fillColor: '#fff', weight: 2, fillOpacity: 1
            }).addTo(map).bindPopup(`Cust: ${node.id}`);
        });

        latlngs.push(depotPos); // Return to depot

        // Draw Route
        L.polyline(latlngs, { color: color, weight: 2.5, opacity: 0.8 }).addTo(map);
    });

    map.fitBounds(bounds, { padding: [20, 20] });

    // Update Meta Stats
    const metaId = getMetaId(sol.algorithm);
    const el = document.getElementById(`meta-${metaId}`);
    if(el) el.innerHTML = `<i class="fa-solid fa-truck"></i> ${sol.vehicles} | <i class="fa-solid fa-road"></i> ${sol.distance.toFixed(1)} | <i class="fa-regular fa-clock"></i> ${sol.time.toFixed(2)}s`;
}

function getMetaId(algoName) {
    return algoName.toLowerCase().replace('dqn+alns', 'hybrid').replace('or-tools', 'ortools');
}

function setStatus(msg, isError = false) {
    const el = document.getElementById('status-bar');
    el.textContent = msg;
    el.style.color = isError ? '#d32f2f' : '#666';
}

function renderTable() {
    const container = document.getElementById('table-container');
    let html = `<table><thead><tr><th>Algorithm</th><th>Vehicles</th><th>Distance</th><th>Time (s)</th></tr></thead><tbody>`;
    
    currentSolutionData.forEach(row => {
        if(row.error) {
            html += `<tr><td>${row.algorithm}</td><td colspan="3" style="color:red">Failed: ${row.error}</td></tr>`;
            return;
        }
        html += `<tr>
            <td><strong>${row.algorithm}</strong></td>
            <td>${row.vehicles}</td>
            <td>${row.distance.toFixed(2)}</td>
            <td>${row.time.toFixed(3)}</td>
        </tr>`;
    });
    html += '</tbody></table>';
    container.innerHTML = html;
}
