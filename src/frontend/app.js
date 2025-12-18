let maps = [];
let instances = [];
let currentResults = null;
let logs = [];

document.addEventListener('DOMContentLoaded', async () => {
    await loadInstances();
    initMaps();
    setupListeners();
    log('System ready');
});

function log(msg) {
    const time = new Date().toLocaleTimeString('en-US', { hour12: false });
    logs.push(`[${time}] ${msg}`);
    if (logs.length > 100) logs.shift();
    
    const output = document.getElementById('log-output');
    output.innerHTML = logs.map(l => `<div class="log-entry">${l}</div>`).join('');
    output.scrollTop = output.scrollHeight;
}

function status(msg, type = 'info') {
    const bar = document.getElementById('status-bar');
    bar.textContent = msg;
    bar.style.color = type === 'error' ? 'var(--error)' : 
                      type === 'success' ? 'var(--success)' : 'var(--text)';
}

async function loadInstances() {
    try {
        log('Loading instances...');
        const res = await fetch('/api/instances');
        instances = await res.json();
        
        const select = document.getElementById('instance-select');
        const preview = document.getElementById('preview-select');
        
        const options = instances.map(i => `<option value="${i}">${i.toUpperCase()}</option>`).join('');
        select.innerHTML = options;
        preview.innerHTML = '<option value="">Choose...</option>' + options;
        
        log(`Loaded ${instances.length} instances`);
    } catch (err) {
        log(`Error: ${err.message}`);
        status('Failed to load instances', 'error');
    }
}

function initMaps() {
    log('Initializing maps...');
    for (let i = 0; i < 4; i++) {
        const map = L.map(`map-${i}`, {
            center: [0, 0],
            zoom: 13,
            zoomControl: true,
            attributionControl: false
        });
        
        L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png', {
            maxZoom: 19
        }).addTo(map);
        
        maps.push(map);
    }
}

function setupListeners() {
    document.getElementById('run-btn').addEventListener('click', runComparison);
    document.getElementById('clear-btn').addEventListener('click', clearMaps);
    document.getElementById('preview-btn').addEventListener('click', loadPreview);
    document.getElementById('show-table-btn').addEventListener('click', showTable);
    document.getElementById('close-modal').addEventListener('click', () => {
        document.getElementById('table-modal').classList.remove('active');
    });
    
    document.querySelectorAll('.expand-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const idx = e.target.dataset.index;
            const item = e.target.closest('.grid-item');
            
            if (item.classList.contains('expanded')) {
                item.classList.remove('expanded');
                e.target.textContent = '⛶';
            } else {
                document.querySelectorAll('.grid-item').forEach(g => {
                    g.classList.remove('expanded');
                    g.querySelector('.expand-btn').textContent = '⛶';
                });
                item.classList.add('expanded');
                e.target.textContent = '◧';
            }
            
            setTimeout(() => maps[idx].invalidateSize(), 100);
        });
    });
}

function clearMaps() {
    maps.forEach(map => {
        map.eachLayer(layer => {
            if (layer instanceof L.Polyline || layer instanceof L.CircleMarker) {
                map.removeLayer(layer);
            }
        });
        map.setView([0, 0], 2);
    });
    
    ['v', 'd', 't'].forEach(prefix => {
        for (let i = 0; i < 4; i++) {
            document.getElementById(`${prefix}-${i}`).textContent = '-';
        }
    });
    
    log('Maps cleared');
    status('Maps cleared');
}

async function loadPreview() {
    const inst = document.getElementById('preview-select').value;
    if (!inst) return;
    
    log(`Loading preview: ${inst}`);
    
    try {
        const res = await fetch('/api/load_preview', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ instance: inst })
        });
        
        const data = await res.json();
        
        maps.forEach(map => {
            map.eachLayer(layer => {
                if (layer instanceof L.CircleMarker) map.removeLayer(layer);
            });
            
            const depot = [data.depot.lat, data.depot.lng];
            const customers = data.customers.map(c => [c.lat, c.lng]);
            
            map.fitBounds([depot, ...customers], { padding: [40, 40] });
            
            L.circleMarker(depot, {
                radius: 8,
                fillColor: '#2563eb',
                color: '#ffffff',
                weight: 2,
                fillOpacity: 1
            }).addTo(map).bindPopup('<b>Depot</b>');
            
            customers.forEach((c, i) => {
                L.circleMarker(c, {
                    radius: 4,
                    fillColor: '#94a3b8',
                    color: '#ffffff',
                    weight: 1,
                    fillOpacity: 1
                }).addTo(map).bindPopup(`Customer ${i + 1}`);
            });
        });
        
        log(`Preview loaded: ${data.customers.length} customers`);
        status(`Loaded ${inst.toUpperCase()}`);
    } catch (err) {
        log(`Preview error: ${err.message}`);
    }
}

async function runComparison() {
    const select = document.getElementById('instance-select');
    const selected = Array.from(select.selectedOptions).map(o => o.value);
    const algos = Array.from(document.querySelectorAll('.checkbox-label input:checked')).map(c => c.value);
    const maxVeh = parseInt(document.getElementById('max-vehicles').value) || 15;
    
    if (!selected.length) {
        status('Select instances', 'error');
        return;
    }
    
    if (!algos.length) {
        status('Select algorithms', 'error');
        return;
    }
    
    log(`Starting: ${selected.length} instances × ${algos.length} algorithms`);
    status('Running comparison...');
    document.getElementById('run-btn').disabled = true;
    
    try {
        const res = await fetch('/api/run_comparison', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                instances: selected,
                algorithms: algos,
                max_vehicles: maxVeh
            })
        });
        
        currentResults = await res.json();
        displayResults();
        log(`Completed: ${currentResults.solutions.length} solutions`);
        status('Comparison complete', 'success');
    } catch (err) {
        log(`Error: ${err.message}`);
        status('Comparison failed', 'error');
    } finally {
        document.getElementById('run-btn').disabled = false;
    }
}

function displayResults() {
    clearMaps();
    
    const algoMap = {
        'ALNS': 0,
        'DQN': 1,
        'DQN+ALNS': 2,
        'OR-TOOLS': 3
    };
    
    currentResults.solutions.forEach(sol => {
        const idx = algoMap[sol.algorithm];
        if (idx !== undefined) visualize(sol, idx);
    });
}

function visualize(sol, idx) {
    const map = maps[idx];
    
    const depot = [sol.depot.lat, sol.depot.lng];
    const allPts = [depot, ...sol.routes.flatMap(r => r.nodes.map(n => [n.lat, n.lng]))];
    
    map.fitBounds(allPts, { padding: [30, 30] });
    
    L.circleMarker(depot, {
        radius: 8,
        fillColor: '#2563eb',
        color: '#ffffff',
        weight: 2,
        fillOpacity: 1
    }).addTo(map).bindPopup('<b>Depot</b>');
    
    const colors = ['#ef4444', '#3b82f6', '#10b981', '#f59e0b', '#8b5cf6', '#06b6d4'];
    
    sol.routes.forEach((route, i) => {
        const color = colors[i % colors.length];
        const pts = [depot, ...route.nodes.map(n => [n.lat, n.lng]), depot];
        
        L.polyline(pts, {
            color: color,
            weight: 3,
            opacity: 0.7
        }).addTo(map);
        
        route.nodes.forEach((node, j) => {
            L.circleMarker([node.lat, node.lng], {
                radius: 5,
                fillColor: '#ffffff',
                color: color,
                weight: 2,
                fillOpacity: 1
            }).addTo(map).bindPopup(`Route ${i + 1}<br>Stop ${j + 1}`);
        });
    });
    
    document.getElementById(`v-${idx}`).textContent = sol.vehicles;
    document.getElementById(`d-${idx}`).textContent = sol.distance.toFixed(1);
    document.getElementById(`t-${idx}`).textContent = sol.time.toFixed(2) + 's';
}

function showTable() {
    if (!currentResults) {
        status('No results yet', 'error');
        return;
    }
    
    let html = '<table><thead><tr>';
    html += '<th>Instance</th><th>Algorithm</th><th>Vehicles</th><th>Distance</th><th>Time (s)</th>';
    html += '</tr></thead><tbody>';
    
    currentResults.table.forEach(row => {
        html += `<tr>
            <td>${row.instance}</td>
            <td>${row.algorithm}</td>
            <td>${row.vehicles}</td>
            <td>${row.distance.toFixed(2)}</td>
            <td>${row.time.toFixed(2)}</td>
        </tr>`;
    });
    
    html += '</tbody></table>';
    document.getElementById('table-wrapper').innerHTML = html;
    document.getElementById('table-modal').classList.add('active');
}
