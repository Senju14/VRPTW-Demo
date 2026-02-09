/**
 * VRPTW Planner - Advanced Research Script
 * Features: Multi-criteria ranking, Gantt Charts, Real Road Routing.
 */

const API = {
    instances: '/api/instances',
    load: '/api/load',
    solve: '/api/solve',
    parsePaste: '/api/parse_paste'
};

const CONFIG = {
    center: [10.793297, 106.632957], // District 1/3 HCMC Center
    scale: 0.0008,
    colors: ['#1a73e8', '#ea4335', '#34a853', '#fbbc04', '#9334e6', '#e91e63', '#00bcd4', '#795548']
};

const STORAGE_KEY = 'vrptw_history_v4';

// Global State
let maps = {};
let prodMap = null;
let currentBenchmarkData = null;
let prodCustomers = [];
let isAddMode = false;
let depotMarker = null;
let lastSolution = null;
let isRestoring = false;

// deterministic random for stable mock geocoding
function getStableCoord(address, seedOffset) {
    let hash = 0;
    for (let i = 0; i < address.length; i++) {
        hash = ((hash << 5) - hash) + address.charCodeAt(i);
        hash |= 0;
    }
    const rand = Math.abs(Math.sin(hash + seedOffset));
    return (rand - 0.5) * 0.02;
}

// ===== INITIALIZATION =====
document.addEventListener('DOMContentLoaded', async () => {
    initTabs();
    initMaps();
    initModals();
    await loadInstances();
    setupEvents();
    loadHistory();
});

// ===== UI COMPONENTS =====
function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `<span class="material-icons">info</span> <span>${message}</span>`;
    container.appendChild(toast);
    setTimeout(() => toast.remove(), 4000);
}

function showModal(title, message, onOk = null, onCancel = null) {
    const modal = document.getElementById('custom-modal');
    document.getElementById('modal-title').textContent = title;
    document.getElementById('modal-message').textContent = message;
    const okBtn = document.getElementById('modal-btn-ok');
    const cancelBtn = document.getElementById('modal-btn-cancel');
    cancelBtn.style.display = onCancel ? 'inline-flex' : 'none';
    okBtn.onclick = () => { modal.style.display = 'none'; if (onOk) onOk(); };
    cancelBtn.onclick = () => { modal.style.display = 'none'; if (onCancel) onCancel(); };
    modal.style.display = 'flex';
}

function initModals() {
    document.getElementById('modal-close-icon').onclick = () => document.getElementById('custom-modal').style.display = 'none';
}

// ===== TABS MANAGEMENT =====
function initTabs() {
    document.querySelectorAll('.nav-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            const target = tab.dataset.tab;
            document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            tab.classList.add('active');
            document.getElementById(`tab-${target}`).classList.add('active');
            if (target === 'dashboard') updateDashboard(lastSolution);
            setTimeout(() => {
                if (target === 'benchmark') Object.values(maps).forEach(m => m.invalidateSize());
                else if (target === 'production' && prodMap) prodMap.invalidateSize();
            }, 100);
        });
    });
}

// ===== MAP OPERATIONS =====
function initMaps() {
    ['ALNS', 'Hybrid'].forEach(algo => {
        const id = `map-${algo.toLowerCase()}`;
        if (document.getElementById(id)) maps[algo] = createMap(id);
    });
    if (document.getElementById('map-prod')) {
        prodMap = createMap('map-prod');
        depotMarker = L.marker(CONFIG.center, {
            draggable: true,
            icon: L.divIcon({
                className: 'depot-icon',
                html: '<span class="material-icons" style="font-size:32px;color:#202124;">warehouse</span>',
                iconSize: [32, 32],
                iconAnchor: [16, 32]
            })
        }).addTo(prodMap).bindPopup('<b>Main Hub (Depot)</b>');
        prodMap.on('click', onMapClick);
    }
}

function createMap(id) {
    const map = L.map(id).setView(CONFIG.center, 14);
    L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
        attribution: '© VRPTW Planner', maxZoom: 19
    }).addTo(map);
    return map;
}

function toLatLng(x, y, isBenchmark = true) {
    if (!isBenchmark) return [x, y];
    return [
        CONFIG.center[0] + (x - 40) * CONFIG.scale,
        CONFIG.center[1] + (y - 40) * CONFIG.scale
    ];
}

// ===== BENCHMARK FLOW =====
async function loadInstances() {
    try {
        const res = await fetch(API.instances);
        const list = await res.json();
        document.getElementById('instance-select').innerHTML = list.map(n => `<option value="${n}">${n}</option>`).join('');
    } catch (e) { showToast('Server offline', 'error'); }
}

async function loadInstanceToWorkspace() {
    const instance = document.getElementById('instance-select').value;
    if (!instance) return;
    setLoadingStatus(true, 'load', 'Loading...');
    try {
        const res = await fetch(API.load, {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ instance })
        });
        currentBenchmarkData = await res.json();
        currentBenchmarkData.instance = instance;
        document.getElementById('info-card').style.display = 'block';
        document.getElementById('info-name').textContent = instance;
        document.getElementById('info-customers').textContent = currentBenchmarkData.customers.length;
        document.getElementById('info-capacity').textContent = currentBenchmarkData.capacity;
        Object.values(maps).forEach(map => {
            clearMap(map);
            drawNodes(map, currentBenchmarkData, true);
        });
        setLoadingStatus(false, 'load', 'Success!');
    } catch (e) {
        showModal('Error', 'Failed to load instance');
        setLoadingStatus(false, 'load');
    }
}

async function runBenchmark() {
    if (!currentBenchmarkData) return showModal('Info', 'Load a dataset first.');
    const maxV = parseInt(document.getElementById('max-vehicles').value);
    const timeLimit = parseInt(document.getElementById('time-limit').value);
    setLoadingStatus(true, 'solve', 'Solving...');
    try {
        const res = await fetch(API.solve, {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                instance: currentBenchmarkData.instance,
                algorithms: ['ALNS', 'Proposed'],
                max_vehicles: maxV,
                time_limit: timeLimit
            })
        });
        const data = await res.json();
        processResults(data.solutions, true);
        setLoadingStatus(false, 'solve', 'Done!');
    } catch (e) {
        showModal('Error', 'Solver failed');
        setLoadingStatus(false, 'solve');
    }
}

function processResults(solutions, isBenchmark) {
    const valid = solutions.filter(s => !s.error);
    valid.sort((a, b) => (a.vehicles - b.vehicles) || (a.distance - b.distance) || (a.time - b.time));
    const winner = valid[0];
    lastSolution = winner;

    displayResultsUI(solutions, winner);

    solutions.forEach(sol => {
        const mapKey = sol.algorithm === 'Proposed' ? 'Hybrid' : sol.algorithm;
        const map = maps[mapKey] || prodMap;
        if (map && !sol.error) {
            clearMap(map);
            drawNodes(map, isBenchmark ? currentBenchmarkData : { depot: sol.depot, customers: prodCustomers }, isBenchmark);
            drawSolution(map, sol, isBenchmark);
            updateMapMeta(sol, winner);
        }
    });

    if (!isRestoring) {
        saveToHistory(isBenchmark ? currentBenchmarkData.instance : 'Custom', solutions);
    }
    updateDashboard(winner);
}

function displayResultsUI(solutions, winner) {
    const card = document.getElementById('results-card');
    card.style.display = 'block';
    const content = document.getElementById('results-content');
    content.innerHTML = `
        <table class="data-table">
            <thead>
                <tr><th>Solver</th><th>Veh</th><th>Dist</th><th>Time</th></tr>
            </thead>
            <tbody>
                ${solutions.map(s => `
                    <tr class="${s.algorithm === winner.algorithm ? 'winner-row' : ''}">
                        <td>${s.algorithm}</td>
                        <td>${s.error ? '-' : s.vehicles}</td>
                        <td>${s.error ? '-' : s.distance.toFixed(1)}</td>
                        <td>${s.time.toFixed(1)}s</td>
                    </tr>
                `).join('')}
            </tbody>
        </table>
    `;
}

// ===== CUSTOM PROBLEM =====
function onMapClick(e) {
    if (!isAddMode) return;
    const id = prodCustomers.length + 1;
    prodCustomers.push({ id, lat: e.latlng.lat, lng: e.latlng.lng, demand: 10, ready_time: 0, due_time: 1440, service_time: 15, address: `Stop ${id}` });
    L.circleMarker(e.latlng, { radius: 6, color: '#1a73e8', fillColor: '#fff', fillOpacity: 1, weight: 2 }).addTo(prodMap);
    renderProdList();
}

async function parseAndLoadPasteData() {
    const text = document.getElementById('paste-area').value;
    setLoadingStatus(true, 'parse', 'Parsing...');
    try {
        const res = await fetch(API.parsePaste, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text }) });
        const data = await res.json();
        if (data.customers.length) {
            depotMarker.setLatLng([data.customers[0].lat || CONFIG.center[0], data.customers[0].lng || CONFIG.center[1]]);
            prodCustomers = data.customers.slice(1).map((c, i) => {
                const lat = c.lat || (CONFIG.center[0] + getStableCoord(c.address, 1));
                const lng = c.lng || (CONFIG.center[1] + getStableCoord(c.address, 2));
                return { ...c, id: i + 1, lat, lng };
            });
            drawProdNodes();
            renderProdList();
            setLoadingStatus(false, 'parse', 'Ready!');
        }
    } catch (e) {
        showToast('Parse failed', 'error');
        setLoadingStatus(false, 'parse');
    }
}

async function solveCustom() {
    if (prodCustomers.length < 1) return;
    setLoadingStatus(true, 'plan', 'Solving...');
    try {
        const depotPos = depotMarker.getLatLng();
        const res = await fetch(API.solve, {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                algorithms: ['Proposed'],
                max_vehicles: parseInt(document.getElementById('prod-vehicles').value),
                time_limit: 15,
                depot: { id: 0, lat: depotPos.lat, lng: depotPos.lng },
                customers: prodCustomers
            })
        });
        const data = await res.json();
        processResults(data.solutions, false);
        setLoadingStatus(false, 'plan', 'Done!');
        showToast('Route Optimized!', 'success');
    } catch (e) {
        showModal('Error', 'Optimization failed');
        setLoadingStatus(false, 'plan');
    }
}

// ===== DASHBOARD & ANALYTICS =====
function updateDashboard(sol) {
    if (!sol) return;
    document.getElementById('dash-instance-info').textContent = sol.algorithm + ' Solution';
    document.getElementById('dash-distance').textContent = `${sol.distance.toFixed(1)} km`;
    document.getElementById('dash-vehicles').textContent = sol.vehicles;
    document.getElementById('dash-runtime').textContent = `${sol.time.toFixed(2)}s`;

    renderTimelineAxis(sol);
    renderGanttChart(sol);
    renderVehicleMetrics(sol);
}

function renderTimelineAxis(sol) {
    const axis = document.getElementById('timeline-axis');
    const maxTime = Math.max(...sol.routes.flatMap(r => r.nodes.map(n => n.end_service))) || 1440;
    const timeScale = 100 / maxTime;

    const markers = [];
    const interval = maxTime > 600 ? 180 : 60; // 3h or 1h markers

    for (let t = 0; t <= maxTime; t += interval) {
        const hour = Math.floor(t / 60);
        const label = hour + 'h';
        markers.push(`<div class="axis-marker" style="left:${t * timeScale}%">${label}</div>`);
    }
    axis.innerHTML = markers.join('');
}

function renderGanttChart(sol) {
    const container = document.getElementById('gantt-chart');
    if (!sol.routes || !sol.routes.length) return;

    const maxTime = Math.max(...sol.routes.flatMap(r => r.nodes.map(n => n.end_service))) || 1440;
    const timeScale = 100 / maxTime;

    container.innerHTML = sol.routes.map(route => {
        let prevEnd = 0;
        return `
            <div class="timeline-row">
                <div class="timeline-label">Vehicle ${route.vehicle_id}</div>
                <div class="timeline-bars">
                    ${route.nodes.map(node => {
            const travelW = (node.arrival_time - prevEnd) * timeScale;
            const waitW = (node.start_service - node.arrival_time) * timeScale;
            const serviceW = (node.end_service - node.start_service) * timeScale;
            const travelL = prevEnd * timeScale;
            const waitL = node.arrival_time * timeScale;
            const serviceL = node.start_service * timeScale;

            prevEnd = node.end_service;

            return `
                            <div class="timeline-bar bar-travel" style="left:${travelL}%; width:${travelW}%;"></div>
                            <div class="timeline-bar bar-wait" style="left:${waitL}%; width:${waitW}%;"></div>
                            <div class="timeline-bar bar-service" style="left:${serviceL}%; width:${serviceW}%;"></div>
                        `;
        }).join('')}
                </div>
            </div>
        `;
    }).join('');
}

function renderVehicleMetrics(sol) {
    const list = document.getElementById('vehicle-list');
    list.innerHTML = sol.routes.map(r => `
        <div class="vehicle-metric-item">
            <span class="v-id">Vehicle ${r.vehicle_id}</span>
            <span class="v-details">${r.nodes.length} stops | ${r.distance.toFixed(1)} km</span>
        </div>
    `).join('');
}

// ===== HISTORY & STORAGE =====
function saveToHistory(instance, solutions) {
    const hist = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]');
    hist.unshift({ id: Date.now(), time: new Date().toLocaleString(), instance, solutions });
    localStorage.setItem(STORAGE_KEY, JSON.stringify(hist.slice(0, 50)));
    loadHistory();
}

function loadHistory() {
    const hist = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]');
    const tbody = document.getElementById('history-tbody');
    const empty = document.getElementById('history-empty');
    if (!hist.length) { empty.style.display = 'block'; tbody.innerHTML = ''; return; }
    empty.style.display = 'none';
    tbody.innerHTML = hist.map(h => {
        const best = [...h.solutions].filter(s => !s.error).sort((a, b) => (a.vehicles - b.vehicles) || (a.distance - b.distance))[0] || h.solutions[0];
        return `<tr>
            <td>${h.time}</td><td>${h.instance}</td><td>${best.algorithm}</td>
            <td>${best.vehicles}</td><td>${best.distance.toFixed(1)}</td><td>${best.time.toFixed(1)}s</td>
            <td style="display:flex; gap:8px;">
                <button class="btn btn-view btn-icon" title="View Result" onclick="viewHistoryItem(${h.id})"><span class="material-icons">visibility</span></button>
                <button class="btn btn-delete btn-icon" title="Delete Entry" onclick="deleteHistoryItem(${h.id})"><span class="material-icons">delete</span></button>
            </td>
        </tr>`;
    }).join('');
}

function viewHistoryItem(id) {
    const hist = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]');
    const item = hist.find(h => h.id === id);
    if (!item) return;

    isRestoring = true;
    const isBench = item.instance !== 'Custom';
    processResults(item.solutions, isBench);
    isRestoring = false;

    // Switch to Benchmark or Custom tab
    const tab = isBench ? 'benchmark' : 'production';
    document.querySelector(`.nav-tab[data-tab="${tab}"]`).click();
    showToast(`Restored results for ${item.instance}`);
}

function deleteHistoryItem(id) {
    showModal('Delete Entry', 'Are you sure you want to delete this execution log?', () => {
        let hist = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]');
        hist = hist.filter(h => h.id !== id);
        localStorage.setItem(STORAGE_KEY, JSON.stringify(hist));
        loadHistory();
        showToast('Entry deleted');
    }, () => { });
}

// ===== UTILS =====
function setLoadingStatus(isLoading, type, label) {
    const btn = document.getElementById(`btn-${type}`);
    if (!btn) return;
    btn.disabled = isLoading;

    // Save original icon and text
    if (isLoading && !btn.dataset.original) {
        btn.dataset.original = btn.innerHTML;
    }

    if (isLoading) {
        btn.innerHTML = `<span class="loading-spinner"></span> ${label}`;
    } else {
        // Show success label briefly if provided
        if (label) {
            btn.innerHTML = `<span class="material-icons">check</span> ${label}`;
            setTimeout(() => {
                btn.innerHTML = btn.dataset.original;
                delete btn.dataset.original;
            }, 2000);
        } else {
            btn.innerHTML = btn.dataset.original;
            delete btn.dataset.original;
        }
    }
}

function clearMap(map) { map.eachLayer(l => (l instanceof L.Path || l instanceof L.CircleMarker) && map.removeLayer(l)); }

function drawNodes(map, data, isBenchmark) {
    const depot = toLatLng(data.depot.lat, data.depot.lng, isBenchmark);
    L.circleMarker(depot, { radius: 10, color: '#202124', fillColor: '#202124', fillOpacity: 1 }).addTo(map);
    data.customers.forEach(c => L.circleMarker(toLatLng(c.lat, c.lng, isBenchmark), { radius: 5, color: '#666', weight: 1 }).addTo(map));
    map.fitBounds(L.latLngBounds([depot, ...data.customers.map(c => toLatLng(c.lat, c.lng, isBenchmark))]), { padding: [30, 30] });
}

function drawSolution(map, sol, isBenchmark) {
    const depot = toLatLng(sol.depot.lat, sol.depot.lng, isBenchmark);
    sol.routes.forEach((r, i) => {
        let latlngs;
        if (r.geometry && r.geometry.length > 0) {
            latlngs = r.geometry;
        } else {
            latlngs = [depot, ...r.nodes.map(n => toLatLng(n.lat, n.lng, isBenchmark)), depot];
        }

        const color = CONFIG.colors[i % 8];
        const weight = isBenchmark ? 3 : 4;
        const opacity = isBenchmark ? 0.7 : 0.8;

        L.polyline(latlngs, { color, weight, opacity, lineJoin: 'round' }).addTo(map);
    });
}

function updateMapMeta(sol, winner) {
    const key = sol.algorithm === 'Proposed' ? 'hybrid' : sol.algorithm.toLowerCase();
    const el = document.getElementById(`meta-${key}`);
    if (el) el.textContent = `${sol.vehicles} vehicles | ${sol.distance.toFixed(1)} km | ${sol.time.toFixed(1)}s`;

    const panel = document.getElementById(`panel-${key}`);
    if (panel) {
        panel.classList.remove('winner', 'loser');
        panel.classList.add(sol.algorithm === winner.algorithm ? 'winner' : 'loser');
    }
}

function drawProdNodes() {
    clearMap(prodMap);
    prodCustomers.forEach(c => L.circleMarker([c.lat, c.lng], { radius: 6, color: '#1a73e8', fillColor: '#fff', fillOpacity: 1, weight: 2 }).addTo(prodMap));
}

function renderProdList() {
    const summary = document.getElementById('prod-summary');
    if (prodCustomers.length > 0) {
        summary.style.display = 'block';
        summary.innerHTML = `<span class="material-icons" style="font-size:16px; vertical-align:middle;">check_circle</span> 1 Depot + ${prodCustomers.length} Customers loaded`;
    } else {
        summary.style.display = 'none';
    }
}

function setupEvents() {
    document.getElementById('btn-load').onclick = loadInstanceToWorkspace;
    document.getElementById('btn-solve').onclick = runBenchmark;
    document.getElementById('btn-clear').onclick = () => {
        showModal('Clear Workspace', 'Are you sure you want to clear all current results?', () => {
            clearBenchmarkDisplay();
            showToast('Workspace cleared');
        }, () => { });
    };
    document.getElementById('btn-parse').onclick = parseAndLoadPasteData;
    document.getElementById('btn-plan').onclick = solveCustom;
    document.getElementById('btn-add-mode').onclick = function () {
        isAddMode = !isAddMode;
        this.classList.toggle('active', isAddMode);
        this.innerHTML = isAddMode ? 'CANCEL' : '<span class="material-icons">add_location</span> Manual Entry';
    };
    document.getElementById('btn-clear-history').onclick = () => {
        showModal('Clear History', 'Delete all saved execution logs?', () => {
            localStorage.removeItem(STORAGE_KEY);
            loadHistory();
            showToast('History cleared');
        }, () => { });
    };
}

function clearBenchmarkDisplay() {
    currentBenchmarkData = null;
    lastSolution = null;
    document.getElementById('results-card').style.display = 'none';

    // Clear Map Meta and classes
    ['alns', 'hybrid'].forEach(key => {
        const el = document.getElementById(`meta-${key}`);
        if (el) el.textContent = 'Waiting for solve...';
        const panel = document.getElementById(`panel-${key}`);
        if (panel) panel.classList.remove('winner', 'loser');
    });

    Object.values(maps).forEach(clearMap);
}
