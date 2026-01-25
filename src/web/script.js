/**
 * VRPTW Solver Frontend Application
 */

const API = {
    instances: '/api/instances',
    load: '/api/load',
    solve: '/api/solve'
};

const CONFIG = {
    center: [10.762622, 106.660172],
    scale: 0.0008,
    colors: ['#1a73e8', '#ea4335', '#34a853', '#fbbc04', '#9334e6', '#e91e63', '#00bcd4']
};

const STORAGE_KEY = 'vrptw_history';

let maps = {};
let solutionData = {};

// Coordinate transform
function toLatLng(x, y) {
    return [
        CONFIG.center[0] + (x - 50) * CONFIG.scale,
        CONFIG.center[1] + (y - 50) * CONFIG.scale
    ];
}

// Initialize
document.addEventListener('DOMContentLoaded', async () => {
    initMaps();
    await loadInstances();
    setupEvents();
    loadHistory();
});

function initMaps() {
    ['ALNS', 'Hybrid'].forEach(algo => {
        const id = `map-${algo.toLowerCase()}`;
        const el = document.getElementById(id);
        if (!el) return;

        const map = L.map(id, { zoomControl: false }).setView(CONFIG.center, 13);
        L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
            attribution: '', maxZoom: 19
        }).addTo(map);

        maps[algo] = map;
    });
}

async function loadInstances() {
    try {
        const res = await fetch(API.instances);
        const list = await res.json();

        const select = document.getElementById('instance-select');
        select.innerHTML = list.map(name =>
            `<option value="${name}">${name}</option>`
        ).join('');

        setStatus('Ready');
    } catch (e) {
        setStatus('Failed to load instances', true);
    }
}

function setupEvents() {
    document.getElementById('btn-load').addEventListener('click', loadInstance);
    document.getElementById('btn-solve').addEventListener('click', solve);
    document.getElementById('btn-clear').addEventListener('click', clearMaps);
    document.getElementById('btn-clear-history').addEventListener('click', clearHistory);
}

async function loadInstance() {
    const instance = document.getElementById('instance-select').value;
    if (!instance) return setStatus('Select an instance', true);

    setStatus('Loading...');
    clearMaps();

    try {
        const res = await fetch(API.load, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ instance })
        });
        const data = await res.json();

        // Update info card
        const infoCard = document.getElementById('info-card');
        infoCard.style.display = 'block';
        document.getElementById('info-name').textContent = instance;
        document.getElementById('info-customers').textContent = data.customers.length;
        document.getElementById('info-capacity').textContent = data.capacity;

        // Draw on both maps
        Object.keys(maps).forEach(algo => {
            drawNodes(maps[algo], data.depot, data.customers);
        });

        setStatus(`Loaded ${instance}`);
    } catch (e) {
        setStatus('Error: ' + e.message, true);
    }
}

function drawNodes(map, depot, customers) {
    const depotPos = toLatLng(depot.lat, depot.lng);
    const bounds = L.latLngBounds([depotPos]);

    // Depot
    L.circleMarker(depotPos, {
        radius: 8, color: '#202124', fillColor: '#202124', fillOpacity: 1, weight: 2
    }).addTo(map).bindPopup('<b>Depot</b>');

    // Customers
    customers.forEach(c => {
        const pos = toLatLng(c.lat, c.lng);
        bounds.extend(pos);
        L.circleMarker(pos, {
            radius: 4, color: '#5f6368', fillColor: '#ffffff', fillOpacity: 1, weight: 1.5
        }).addTo(map);
    });

    map.fitBounds(bounds, { padding: [30, 30] });
}

async function solve() {
    const instance = document.getElementById('instance-select').value;
    const maxVehicles = parseInt(document.getElementById('max-vehicles').value);
    const algorithms = Array.from(document.querySelectorAll('input[type="checkbox"]:checked'))
        .map(cb => cb.value);

    if (!instance) return setStatus('Select an instance', true);
    if (!algorithms.length) return setStatus('Select at least one algorithm', true);

    const btn = document.getElementById('btn-solve');
    btn.disabled = true;
    btn.innerHTML = '<span class="material-icons">hourglass_empty</span> Running...';
    setStatus('Solving...');

    try {
        const res = await fetch(API.solve, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ instance, algorithms, max_vehicles: maxVehicles })
        });
        const data = await res.json();

        solutionData = {};
        clearMaps();
        document.getElementById('info-card').style.display = 'block';

        data.solutions.forEach(sol => {
            solutionData[sol.algorithm] = sol;
            if (!sol.error) {
                drawSolution(sol);
            }
        });

        showResults(data.solutions);
        saveToHistory(instance, data.solutions);
        setStatus('Complete');
    } catch (e) {
        setStatus('Error: ' + e.message, true);
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<span class="material-icons">play_arrow</span> Run Comparison';
    }
}

function drawSolution(sol) {
    const map = maps[sol.algorithm];
    if (!map || !sol.depot) return;

    const depotPos = toLatLng(sol.depot.lat, sol.depot.lng);
    const bounds = L.latLngBounds([depotPos]);

    // Depot
    L.circleMarker(depotPos, {
        radius: 8, color: '#202124', fillColor: '#202124', fillOpacity: 1, weight: 2
    }).addTo(map);

    // Routes
    sol.routes.forEach((route, i) => {
        const color = CONFIG.colors[i % CONFIG.colors.length];
        const latlngs = [depotPos];

        route.nodes.forEach(node => {
            const pos = toLatLng(node.lat, node.lng);
            latlngs.push(pos);
            bounds.extend(pos);

            L.circleMarker(pos, {
                radius: 5, color: color, fillColor: '#ffffff', fillOpacity: 1, weight: 2
            }).addTo(map);
        });

        latlngs.push(depotPos);
        L.polyline(latlngs, { color, weight: 2.5, opacity: 0.8 }).addTo(map);
    });

    map.fitBounds(bounds, { padding: [20, 20] });

    // Update meta
    const metaId = `meta-${sol.algorithm.toLowerCase()}`;
    const metaEl = document.getElementById(metaId);
    if (metaEl) {
        metaEl.textContent = `${sol.vehicles} vehicles | ${sol.distance.toFixed(1)} dist`;
    }
}

function showResults(solutions) {
    const card = document.getElementById('results-card');
    const content = document.getElementById('results-content');
    card.style.display = 'block';

    content.innerHTML = buildResultsTable(solutions);
}

function buildResultsTable(solutions) {
    let html = '<table class="results-table"><tr><th>Algorithm</th><th>Vehicles</th><th>Distance</th><th>Time</th></tr>';

    solutions.forEach(sol => {
        if (sol.error) {
            html += `<tr><td>${sol.algorithm}</td><td colspan="3" style="color:#d93025">${sol.error}</td></tr>`;
        } else {
            html += `<tr>
                <td>${sol.algorithm}</td>
                <td>${sol.vehicles}</td>
                <td>${sol.distance.toFixed(2)}</td>
                <td>${sol.time.toFixed(2)}s</td>
            </tr>`;
        }
    });

    html += '</table>';
    return html;
}

function clearMaps() {
    Object.values(maps).forEach(map => {
        map.eachLayer(layer => {
            if (layer instanceof L.Path || layer instanceof L.CircleMarker) {
                map.removeLayer(layer);
            }
        });
    });

    document.getElementById('meta-alns').textContent = '';
    document.getElementById('meta-hybrid').textContent = '';
    document.getElementById('results-card').style.display = 'none';
    solutionData = {};
}

function setStatus(msg, isError = false) {
    const el = document.getElementById('status');
    el.textContent = msg;
    el.className = isError ? 'status-bar error' : 'status-bar';
}

// History functions
function getHistory() {
    try {
        return JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]');
    } catch {
        return [];
    }
}

function saveHistory(history) {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(history));
}

function saveToHistory(instance, solutions) {
    const history = getHistory();

    const alns = solutions.find(s => s.algorithm === 'ALNS');
    const hybrid = solutions.find(s => s.algorithm === 'Hybrid');

    const entry = {
        id: Date.now(),
        instance,
        timestamp: new Date().toLocaleString(),
        solutions,
        winner: (hybrid && alns && !hybrid.error && !alns.error)
            ? (hybrid.distance < alns.distance ? 'Hybrid' : 'ALNS')
            : null
    };

    history.unshift(entry);

    // Keep only last 20 entries
    if (history.length > 20) {
        history.pop();
    }

    saveHistory(history);
    loadHistory();
}

function loadHistory() {
    const history = getHistory();
    const container = document.getElementById('history-list');

    if (history.length === 0) {
        container.innerHTML = '<div class="history-empty">No runs yet</div>';
        return;
    }

    container.innerHTML = history.map(entry => {
        const winClass = entry.winner === 'Hybrid' ? 'winner' : (entry.winner === 'ALNS' ? 'loser' : '');
        return `
            <div class="history-item ${winClass}" data-id="${entry.id}">
                <div class="history-item-title">${entry.instance}</div>
                <div class="history-item-meta">${entry.timestamp}</div>
            </div>
        `;
    }).join('');

    // Add click handlers
    container.querySelectorAll('.history-item').forEach(item => {
        item.addEventListener('click', () => showHistoryEntry(parseInt(item.dataset.id)));
    });
}

function showHistoryEntry(id) {
    const history = getHistory();
    const entry = history.find(h => h.id === id);

    if (!entry) return;

    const card = document.getElementById('results-card');
    const content = document.getElementById('results-content');
    card.style.display = 'block';

    content.innerHTML = `
        <div style="margin-bottom: 8px; font-weight: 500;">${entry.instance} - ${entry.timestamp}</div>
        ${buildResultsTable(entry.solutions)}
    `;
}

function clearHistory() {
    if (confirm('Clear all history?')) {
        localStorage.removeItem(STORAGE_KEY);
        loadHistory();
    }
}
