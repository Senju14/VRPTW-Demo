/**
 * VRPTW Planner - Complete JavaScript with all fixes
 * - Depot on land (Quận 7, không ở sông)
 * - Click to add customers
 * - Winner/loser borders
 * - Loading indicator
 * - Robust parsing
 */

const API = {
    instances: '/api/instances',
    load: '/api/load',
    solve: '/api/solve',
    parsePaste: '/api/parse_paste',
    solveCustom: '/api/solve_custom'
};

// Quận 7, HCMC - trên đất, không ở sông
const CONFIG = {
    center: [10.7340, 106.7220],  // Quận 7 - Phú Mỹ Hưng area
    scale: 0.0008,
    colors: ['#1a73e8', '#ea4335', '#34a853', '#fbbc04', '#9334e6', '#e91e63', '#00bcd4', '#795548']
};

const STORAGE_KEY = 'vrptw_history';

// State
let maps = {};
let prodMap = null;
let currentData = null;
let prodCustomers = [];
let lastProdSolution = null;
let isAddMode = false;
let depotMarker = null;

// ===== INIT =====
document.addEventListener('DOMContentLoaded', async () => {
    initTabs();
    initMaps();
    await loadInstances();
    setupEvents();
    loadHistory();
});

// ===== LOADING INDICATOR =====
function showLoading(text = 'Processing...') {
    const bar = document.getElementById('loading-bar');
    bar.querySelector('.loading-text').textContent = text;
    bar.style.display = 'flex';
}

function hideLoading() {
    document.getElementById('loading-bar').style.display = 'none';
}

// ===== TABS =====
function initTabs() {
    document.querySelectorAll('.nav-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            const target = tab.dataset.tab;

            document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

            tab.classList.add('active');
            document.getElementById(`tab-${target}`).classList.add('active');

            setTimeout(() => {
                if (target === 'benchmark') {
                    Object.values(maps).forEach(m => m.invalidateSize());
                } else if (target === 'production' && prodMap) {
                    prodMap.invalidateSize();
                }
            }, 100);
        });
    });
}

// ===== MAPS =====
function initMaps() {
    // Benchmark maps - offset để tránh sông
    ['ALNS', 'Hybrid'].forEach(algo => {
        const id = `map-${algo.toLowerCase()}`;
        const el = document.getElementById(id);
        if (el) maps[algo] = createMap(id);
    });

    // Production map
    const prodEl = document.getElementById('map-prod');
    if (prodEl) {
        prodMap = createMap('map-prod');

        // Depot marker - draggable
        depotMarker = L.marker(CONFIG.center, {
            draggable: true,
            icon: L.divIcon({
                className: 'depot-icon',
                html: '<span class="material-icons" style="font-size:28px;color:#202124;">warehouse</span>',
                iconSize: [28, 28],
                iconAnchor: [14, 28]
            })
        }).addTo(prodMap).bindPopup('<b>Depot</b><br>Kéo để di chuyển');

        // Click to add
        prodMap.on('click', onMapClick);
    }
}

function createMap(id) {
    const map = L.map(id).setView(CONFIG.center, 14);
    L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
        attribution: '© OpenStreetMap',
        maxZoom: 19
    }).addTo(map);
    return map;
}

// Solomon coords -> LatLng (offset để tránh sông)
function toLatLng(x, y) {
    return [
        CONFIG.center[0] + (x - 50) * CONFIG.scale * 0.8,
        CONFIG.center[1] + (y - 50) * CONFIG.scale * 0.8 + 0.01  // Shift east slightly
    ];
}

// ===== CLICK TO ADD =====
function onMapClick(e) {
    if (!isAddMode) return;

    const id = prodCustomers.length + 1;
    const customer = {
        id,
        lat: e.latlng.lat,
        lng: e.latlng.lng,
        address: `Customer ${id}`,
        demand: 10,
        ready_time: 0,
        due_time: 1000,
        service_time: 10
    };

    prodCustomers.push(customer);

    L.circleMarker([customer.lat, customer.lng], {
        radius: 8,
        color: '#1a73e8',
        fillColor: '#fff',
        fillOpacity: 1,
        weight: 2
    }).addTo(prodMap).bindPopup(`<b>${customer.address}</b><br>Demand: ${customer.demand}`);

    renderCustomerList();
    setStatus(`Added customer ${id}`);
}

// ===== BENCHMARK =====
async function loadInstances() {
    try {
        const res = await fetch(API.instances);
        const list = await res.json();
        document.getElementById('instance-select').innerHTML =
            list.map(n => `<option value="${n}">${n}</option>`).join('');
        setStatus('Ready');
    } catch (e) {
        setStatus('Load failed', true);
    }
}

async function loadInstance() {
    const instance = document.getElementById('instance-select').value;
    if (!instance) return setStatus('Select instance', true);

    showLoading('Loading instance...');
    clearBenchmarkMaps();

    try {
        const res = await fetch(API.load, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ instance })
        });

        currentData = await res.json();
        currentData.instance = instance;

        document.getElementById('info-card').style.display = 'block';
        document.getElementById('info-name').textContent = instance;
        document.getElementById('info-customers').textContent = currentData.customers.length;
        document.getElementById('info-capacity').textContent = currentData.capacity;

        Object.values(maps).forEach(map => drawNodes(map, currentData));
        setStatus(`Loaded ${instance}`);
    } catch (e) {
        setStatus('Error: ' + e.message, true);
    } finally {
        hideLoading();
    }
}

async function solveBenchmark() {
    if (!currentData) return setStatus('Load data first', true);

    const maxVehicles = parseInt(document.getElementById('max-vehicles').value) || 25;

    showLoading('Running ALNS + Hybrid...');

    try {
        const res = await fetch(API.solve, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                instance: currentData.instance,
                algorithms: ['ALNS', 'Hybrid'],
                max_vehicles: maxVehicles
            })
        });

        const data = await res.json();

        // Show results
        document.getElementById('results-card').style.display = 'block';
        document.getElementById('results-content').innerHTML = buildResultsTable(data.solutions);

        // Draw solutions and set winner/loser
        const alns = data.solutions.find(s => s.algorithm === 'ALNS');
        const hybrid = data.solutions.find(s => s.algorithm === 'Hybrid');

        let winner = null;
        if (alns && hybrid && !alns.error && !hybrid.error) {
            winner = hybrid.distance <= alns.distance ? 'Hybrid' : 'ALNS';
        }

        data.solutions.forEach(sol => {
            if (!sol.error && maps[sol.algorithm]) {
                drawSolution(maps[sol.algorithm], sol);
                updateMeta(sol.algorithm, sol);

                // Set winner/loser border
                const panel = document.getElementById(`panel-${sol.algorithm.toLowerCase()}`);
                panel.classList.remove('winner', 'loser');
                if (winner) {
                    panel.classList.add(sol.algorithm === winner ? 'winner' : 'loser');
                }
            }
        });

        saveToHistory(currentData.instance, data.solutions);
        setStatus('Complete');
    } catch (e) {
        setStatus('Error: ' + e.message, true);
    } finally {
        hideLoading();
    }
}

function clearBenchmarkMaps() {
    Object.values(maps).forEach(map => {
        map.eachLayer(layer => {
            if (layer instanceof L.Path || layer instanceof L.CircleMarker) {
                map.removeLayer(layer);
            }
        });
    });

    // Clear winner/loser classes
    document.getElementById('panel-alns').classList.remove('winner', 'loser');
    document.getElementById('panel-hybrid').classList.remove('winner', 'loser');

    document.getElementById('results-card').style.display = 'none';
    document.getElementById('meta-alns').textContent = '';
    document.getElementById('meta-hybrid').textContent = '';
}

// ===== PRODUCTION =====
async function parsePaste() {
    const text = document.getElementById('paste-area').value;
    if (!text.trim()) return setStatus('Enter data', true);

    showLoading('Parsing data...');

    try {
        const res = await fetch(API.parsePaste, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text })
        });

        const data = await res.json();

        // Distribute customers around depot (Quận 7 area)
        const depotPos = depotMarker.getLatLng();
        prodCustomers = data.customers.map((c, i) => {
            const angle = (i / data.customers.length) * 2 * Math.PI;
            const radius = 0.005 + Math.random() * 0.008;
            return {
                ...c,
                lat: depotPos.lat + Math.cos(angle) * radius,
                lng: depotPos.lng + Math.sin(angle) * radius
            };
        });

        renderCustomerList();
        drawProdCustomers();
        setStatus(`Parsed ${prodCustomers.length} customers`);
    } catch (e) {
        setStatus('Error: ' + e.message, true);
    } finally {
        hideLoading();
    }
}

function renderCustomerList() {
    const card = document.getElementById('cust-card');
    const list = document.getElementById('cust-list');
    const count = document.getElementById('cust-count');

    count.textContent = prodCustomers.length;

    if (!prodCustomers.length) {
        card.style.display = 'none';
        return;
    }

    card.style.display = 'block';
    list.innerHTML = prodCustomers.map(c => `
        <div class="cust-item">
            <span>#${c.id} ${(c.address || '').substring(0, 18)}...</span>
            <span>${c.demand || 10}kg</span>
        </div>
    `).join('');
}

function drawProdCustomers() {
    // Clear markers except depot
    prodMap.eachLayer(layer => {
        if (layer instanceof L.CircleMarker || layer instanceof L.Polyline) {
            prodMap.removeLayer(layer);
        }
    });

    const depotPos = depotMarker.getLatLng();
    const bounds = L.latLngBounds([[depotPos.lat, depotPos.lng]]);

    prodCustomers.forEach(c => {
        const pos = [c.lat, c.lng];
        bounds.extend(pos);
        L.circleMarker(pos, {
            radius: 7,
            color: '#1a73e8',
            fillColor: '#fff',
            fillOpacity: 1,
            weight: 2
        }).addTo(prodMap).bindPopup(`<b>${c.address || 'Customer ' + c.id}</b><br>Demand: ${c.demand || 10}`);
    });

    prodMap.fitBounds(bounds, { padding: [50, 50] });
}

async function planRoutes() {
    if (!prodCustomers.length) return setStatus('Parse or add customers first', true);

    const capacity = parseFloat(document.getElementById('prod-capacity').value) || 100;
    const maxVehicles = parseInt(document.getElementById('prod-vehicles').value) || 10;
    const depotPos = depotMarker.getLatLng();

    showLoading('Planning routes...');

    try {
        const res = await fetch(API.solveCustom, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                depot: { lat: depotPos.lat, lng: depotPos.lng },
                customers: prodCustomers.map(c => ({
                    id: c.id,
                    lat: c.lat,
                    lng: c.lng,
                    demand: c.demand || 10,
                    ready_time: c.ready_time || 0,
                    due_time: c.due_time || 1000,
                    service_time: c.service_time || 10
                })),
                capacity,
                max_vehicles: maxVehicles,
                algorithms: ['Hybrid']
            })
        });

        const data = await res.json();
        lastProdSolution = data.solutions[0];

        if (lastProdSolution && !lastProdSolution.error) {
            drawProdRoutes(lastProdSolution);
            document.getElementById('prod-meta').textContent =
                `${lastProdSolution.vehicles} vehicles • ${lastProdSolution.distance.toFixed(1)} dist`;

            saveToHistory('Custom', data.solutions);
        }

        setStatus('Routes planned');
    } catch (e) {
        setStatus('Error: ' + e.message, true);
    } finally {
        hideLoading();
    }
}

function drawProdRoutes(sol) {
    // Clear routes only
    prodMap.eachLayer(layer => {
        if (layer instanceof L.Polyline) prodMap.removeLayer(layer);
    });

    const depotPos = depotMarker.getLatLng();

    sol.routes.forEach((route, i) => {
        const color = CONFIG.colors[i % CONFIG.colors.length];
        const latlngs = [[depotPos.lat, depotPos.lng]];

        route.nodes.forEach(n => {
            latlngs.push([n.lat, n.lng]);
        });

        latlngs.push([depotPos.lat, depotPos.lng]);
        L.polyline(latlngs, { color, weight: 3, opacity: 0.8 }).addTo(prodMap);
    });
}

// ===== DRAWING =====
function drawNodes(map, data) {
    const depotPos = toLatLng(data.depot.lat, data.depot.lng);
    const bounds = L.latLngBounds([depotPos]);

    L.circleMarker(depotPos, {
        radius: 10, color: '#202124', fillColor: '#202124', fillOpacity: 1
    }).addTo(map).bindPopup('<b>Depot</b>');

    data.customers.forEach(c => {
        const pos = toLatLng(c.lat, c.lng);
        bounds.extend(pos);
        L.circleMarker(pos, {
            radius: 5, color: '#5f6368', fillColor: '#fff', fillOpacity: 1
        }).addTo(map);
    });

    map.fitBounds(bounds, { padding: [30, 30] });
}

function drawSolution(map, sol) {
    map.eachLayer(layer => {
        if (layer instanceof L.Polyline) map.removeLayer(layer);
    });

    const depotPos = toLatLng(sol.depot.lat, sol.depot.lng);

    sol.routes.forEach((route, i) => {
        const color = CONFIG.colors[i % CONFIG.colors.length];
        const latlngs = [depotPos];

        route.nodes.forEach(n => {
            latlngs.push(toLatLng(n.lat, n.lng));
        });

        latlngs.push(depotPos);
        L.polyline(latlngs, { color, weight: 3, opacity: 0.8 }).addTo(map);
    });
}

function updateMeta(algo, sol) {
    const el = document.getElementById(`meta-${algo.toLowerCase()}`);
    if (el) el.textContent = `${sol.vehicles} vehicles • ${sol.distance.toFixed(1)} dist`;
}

function buildResultsTable(sols) {
    return `
        <table style="width:100%;font-size:13px;border-collapse:collapse;">
            <tr style="border-bottom:1px solid #ddd;">
                <th style="text-align:left;padding:8px;">Algo</th>
                <th style="text-align:right;padding:8px;">V</th>
                <th style="text-align:right;padding:8px;">Dist</th>
            </tr>
            ${sols.map(s => `
                <tr>
                    <td style="padding:8px;">${s.algorithm}</td>
                    <td style="text-align:right;padding:8px;">${s.error ? '-' : s.vehicles}</td>
                    <td style="text-align:right;padding:8px;">${s.error ? 'Error' : s.distance.toFixed(1)}</td>
                </tr>
            `).join('')}
        </table>
    `;
}

// ===== HISTORY =====
function loadHistory() {
    const history = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]');
    const tbody = document.getElementById('history-tbody');
    const empty = document.getElementById('history-empty');
    const table = document.getElementById('history-table');

    if (!history.length) {
        tbody.innerHTML = '';
        empty.style.display = 'block';
        table.style.display = 'none';
        return;
    }

    empty.style.display = 'none';
    table.style.display = 'table';

    tbody.innerHTML = history.map(h => {
        const best = h.solutions.reduce((a, b) =>
            ((a.distance || Infinity) < (b.distance || Infinity) ? a : b), h.solutions[0]);
        const winnerClass = h.winner === 'Hybrid' ? 'winner-hybrid' : 'winner-alns';

        return `
            <tr>
                <td>${h.timestamp}</td>
                <td>${h.instance}</td>
                <td class="${winnerClass}">${h.winner || '-'}</td>
                <td>${best.vehicles || '-'}</td>
                <td>${best.distance ? best.distance.toFixed(1) : '-'}</td>
            </tr>
        `;
    }).join('');
}

function saveToHistory(instance, sols) {
    const history = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]');

    const alns = sols.find(s => s.algorithm === 'ALNS');
    const hybrid = sols.find(s => s.algorithm === 'Hybrid');

    let winner = null;
    if (alns && hybrid && !alns.error && !hybrid.error) {
        winner = hybrid.distance <= alns.distance ? 'Hybrid' : 'ALNS';
    } else if (hybrid && !hybrid.error) {
        winner = 'Hybrid';
    }

    history.unshift({
        id: Date.now(),
        timestamp: new Date().toLocaleString('vi-VN'),
        instance,
        solutions: sols,
        winner
    });

    if (history.length > 50) history.pop();
    localStorage.setItem(STORAGE_KEY, JSON.stringify(history));
    loadHistory();
}

function clearHistory() {
    if (confirm('Clear history?')) {
        localStorage.removeItem(STORAGE_KEY);
        loadHistory();
    }
}

// ===== UTILITIES =====
function setStatus(msg, isError = false) {
    const text = document.getElementById('status-text');
    const icon = document.querySelector('.status-icon');
    text.textContent = msg;
    icon.style.color = isError ? '#ea4335' : '';
}

// ===== EVENTS =====
function setupEvents() {
    // Benchmark
    document.getElementById('btn-load').addEventListener('click', loadInstance);
    document.getElementById('btn-solve').addEventListener('click', solveBenchmark);
    document.getElementById('btn-clear').addEventListener('click', () => {
        clearBenchmarkMaps();
        document.getElementById('info-card').style.display = 'none';
        setStatus('Cleared');
    });

    // Production
    document.getElementById('btn-parse').addEventListener('click', parsePaste);
    document.getElementById('btn-plan').addEventListener('click', planRoutes);

    // Add mode toggle
    document.getElementById('btn-add-mode').addEventListener('click', function () {
        isAddMode = !isAddMode;
        this.classList.toggle('active', isAddMode);
        this.innerHTML = isAddMode
            ? '<span class="material-icons">check</span> Adding Mode ON'
            : '<span class="material-icons">add_location</span> Click Map to Add';
        setStatus(isAddMode ? 'Click map to add customers' : 'Add mode off');
    });

    // History
    document.getElementById('btn-clear-history').addEventListener('click', clearHistory);
}
