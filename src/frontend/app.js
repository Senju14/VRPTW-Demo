let maps = {};
const ALGORITHMS = ['ALNS', 'DQN', 'DQN+ALNS', 'OR-Tools'];
const MAP_IDS = { 'ALNS': 'map-alns', 'DQN': 'map-dqn', 'DQN+ALNS': 'map-hybrid', 'OR-Tools': 'map-ortools' };
let currentSolutionData = {}; 

const HCMC_CENTER = [10.762622, 106.660172];
const SCALE_FACTOR = 0.0008;

function transformCoords(x, y) {
    if (x === undefined || y === undefined) return HCMC_CENTER;
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
        
        const map = L.map(id, { zoomControl: false }).setView(HCMC_CENTER, 13);
        L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
            attribution: '', maxZoom: 20
        }).addTo(map);
        maps[algo] = map;
    });

    // Click map to show metrics
    document.querySelectorAll('.map-cell').forEach(cell => {
        cell.addEventListener('click', () => {
            const algo = cell.getAttribute('data-algo');
            updateMetricsPanel(algo);
            
            // Highlight selected map
            document.querySelectorAll('.map-cell').forEach(c => c.style.borderColor = '#e0e0e0');
            cell.style.borderColor = '#000';
        });
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
        if(list.length > 0) select.value = list[0];
    } catch (e) {
        setStatus('Failed to load instances', true);
    }
}

function setupEvents() {
    document.getElementById('btn-load').addEventListener('click', loadInstancePreview);
    document.getElementById('btn-run').addEventListener('click', runBenchmark);
    document.getElementById('btn-clear').addEventListener('click', clearAllMaps);
    
    // Toggle Table logic
    document.getElementById('btn-toggle-table').addEventListener('click', () => {
        renderTable();
        document.getElementById('result-modal').style.display = 'flex';
    });
    
    document.querySelector('.close-modal').addEventListener('click', () => {
        document.getElementById('result-modal').style.display = 'none';
    });
    
    // Close modal when clicking outside
    window.onclick = function(event) {
        const modal = document.getElementById('result-modal');
        if (event.target == modal) {
            modal.style.display = "none";
        }
    }
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
        
        const cell = document.querySelector(`.map-cell[data-algo="${algo}"]`);
        const existingErr = cell.querySelector('.map-overlay-error');
        if(existingErr) existingErr.remove();
        cell.style.borderColor = '#e0e0e0';
    });
    
    document.getElementById('metrics-panel').style.display = 'none';
    document.getElementById('btn-toggle-table').style.display = 'none';
    document.getElementById('instance-info').style.display = 'none';
    setStatus('Maps cleared');
}

async function loadInstancePreview() {
    const instance = document.getElementById('instance-select').value;
    if (!instance) return setStatus('Select an instance first', true);

    setStatus('Loading map data...');
    clearAllMaps();

    try {
        const res = await fetch('/api/load_instance', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ instance: instance })
        });
        
        const data = await res.json();
        
        // Update Info Panel
        document.getElementById('instance-info').style.display = 'block';
        document.getElementById('instance-name').textContent = instance;
        document.getElementById('total-customers').textContent = data.customers.length;
        document.getElementById('vehicle-capacity').textContent = data.capacity;

        ALGORITHMS.forEach(algo => {
            const map = maps[algo];
            drawNodes(map, data.depot, data.customers);
        });

        setStatus(`Loaded ${instance}`);
    } catch (e) {
        setStatus('Error loading instance: ' + e.message, true);
    }
}

function drawNodes(map, depot, customers) {
    if (!depot) return;
    const depotPos = transformCoords(depot.lat, depot.lng);
    const bounds = L.latLngBounds([depotPos]);

    L.circleMarker(depotPos, {
        radius: 6, color: '#000', fillColor: '#000', fillOpacity: 1
    }).addTo(map).bindPopup('<b>DEPOT</b>');

    customers.forEach(c => {
        if(!c) return;
        const pos = transformCoords(c.lat, c.lng);
        bounds.extend(pos);
        L.circleMarker(pos, {
            radius: 3, color: '#888', fillColor: '#fff', weight: 1, fillOpacity: 1
        }).addTo(map);
    });

    map.fitBounds(bounds, { padding: [30, 30] });
}

async function runBenchmark() {
    const instance = document.getElementById('instance-select').value;
    const maxVehicles = parseInt(document.getElementById('max-vehicles').value);
    const selectedAlgos = Array.from(document.querySelectorAll('input[type="checkbox"]:checked')).map(cb => cb.value);

    if (!instance) return setStatus('Select instance', true);
    if (!selectedAlgos.length) return setStatus('Select algorithms', true);

    const btn = document.getElementById('btn-run');
    btn.disabled = true;
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> RUNNING...';
    setStatus('Solving... This may take a while.');

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
        
        currentSolutionData = {};
        clearAllMaps();
        // Keep Info Panel visible
        document.getElementById('instance-info').style.display = 'block';

        data.solutions.forEach(sol => {
            currentSolutionData[sol.algorithm] = sol;
            if (sol.error) {
                showErrorOnMap(sol.algorithm, sol.error);
            } else {
                drawSolution(sol);
            }
        });

        // Show Table Button
        document.getElementById('btn-toggle-table').style.display = 'block';

        // Auto show metrics for first result
        const firstSuccess = data.solutions.find(s => !s.error);
        if(firstSuccess) updateMetricsPanel(firstSuccess.algorithm);

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
    if (!map || !sol.depot || typeof sol.depot.lat === 'undefined') return;

    const depotPos = transformCoords(sol.depot.lat, sol.depot.lng);
    const bounds = L.latLngBounds([depotPos]);
    L.circleMarker(depotPos, { radius: 6, color: '#000', fillColor: '#000', fillOpacity: 1 }).addTo(map);

    const colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#f032e6'];

    sol.routes.forEach((route, i) => {
        const color = colors[i % colors.length];
        const latlngs = [depotPos];

        if (route.nodes) {
            route.nodes.forEach(node => {
                if (!node || typeof node.lat === 'undefined') return;
                const pos = transformCoords(node.lat, node.lng);
                latlngs.push(pos);
                bounds.extend(pos);
                L.circleMarker(pos, {
                    radius: 4, color: color, fillColor: '#fff', weight: 2, fillOpacity: 1
                }).addTo(map);
            });
        }
        latlngs.push(depotPos);
        L.polyline(latlngs, { color: color, weight: 2.5, opacity: 0.8 }).addTo(map);
    });

    map.fitBounds(bounds, { padding: [20, 20] });

    const metaId = getMetaId(sol.algorithm);
    const el = document.getElementById(`meta-${metaId}`);
    if(el) el.innerHTML = `<i class="fa-solid fa-truck"></i> ${sol.vehicles} | <i class="fa-solid fa-road"></i> ${sol.distance.toFixed(1)}`;
}

function updateMetricsPanel(algo) {
    const sol = currentSolutionData[algo];
    const panel = document.getElementById('metrics-panel');
    
    if (!sol || sol.error) {
        panel.style.display = 'none';
        return;
    }
    panel.style.display = 'block';
    document.getElementById('metric-algo').textContent = algo;
    document.getElementById('total-distance').textContent = sol.distance.toFixed(2);
    document.getElementById('total-time').textContent = sol.time.toFixed(2) + 's';
    document.getElementById('vehicles-used').textContent = sol.vehicles;
}

function renderTable() {
    const container = document.getElementById('table-container');
    let html = `<table><thead><tr><th>Algorithm</th><th>Vehicles</th><th>Distance</th><th>Time (s)</th></tr></thead><tbody>`;
    
    // Sort keys to maintain order ALNS -> DQN -> Hybrid -> OR
    const order = ['ALNS', 'DQN', 'DQN+ALNS', 'OR-Tools'];
    
    order.forEach(algo => {
        const row = currentSolutionData[algo];
        if(!row) return;

        if(row.error) {
            html += `<tr><td><strong>${row.algorithm}</strong></td><td colspan="3" style="color:red">Failed</td></tr>`;
        } else {
            html += `<tr>
                <td><strong>${row.algorithm}</strong></td>
                <td>${row.vehicles}</td>
                <td>${row.distance.toFixed(2)}</td>
                <td>${row.time.toFixed(3)}</td>
            </tr>`;
        }
    });
    html += '</tbody></table>';
    container.innerHTML = html;
}

function showErrorOnMap(algo, msg) {
    const cell = document.querySelector(`.map-cell[data-algo="${algo}"]`);
    if(cell) {
        const div = document.createElement('div');
        div.className = 'map-overlay-error';
        div.innerHTML = `<i class="fa-solid fa-circle-exclamation" style="font-size:30px; margin-bottom:10px;"></i><span>${msg}</span>`;
        cell.appendChild(div);
    }
}

function getMetaId(algoName) {
    return algoName.toLowerCase().replace('dqn+alns', 'hybrid').replace('or-tools', 'ortools');
}

function setStatus(msg, isError = false) {
    const el = document.getElementById('status-bar');
    el.textContent = msg;
    el.style.color = isError ? '#d32f2f' : '#666';
}
