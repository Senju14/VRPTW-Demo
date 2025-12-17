let maps = {};
let layers = {};
let currentInstance = null;
let comparisonData = null;
const mapCanvas = L.canvas();

const SOLVERS = ['alns', 'dqn', 'dqn_alns', 'ortools'];
const COLORS = ['#111111', '#666666', '#999999', '#333333']; 

function initMaps() {
    SOLVERS.forEach(solver => {
        const map = L.map(`map-${solver}`, {
            crs: L.CRS.Simple, minZoom: -5, zoomControl: false, attributionControl: false
        });
        if (solver === 'ortools') L.control.zoom({ position: 'bottomright' }).addTo(map);
        
        map.on('move', () => {
            SOLVERS.forEach(s => {
                if (maps[s] && s !== solver) maps[s].setView(map.getCenter(), map.getZoom(), { animate: false });
            });
        });
        maps[solver] = map;
        layers[solver] = L.layerGroup().addTo(map);
    });
}

function plotInstance(instance) {
    SOLVERS.forEach(s => layers[s].clearLayers());
    const nodes = [instance.depot, ...instance.customers];
    const bounds = L.latLngBounds(nodes.map(n => [n.y, n.x]));

    SOLVERS.forEach(solver => {
        const layer = layers[solver];
        L.rectangle([[instance.depot.y-2, instance.depot.x-2], [instance.depot.y+2, instance.depot.x+2]], {
            color: '#000', weight: 0, fillOpacity: 1, renderer: mapCanvas
        }).bindTooltip(`<b>DEPOT</b>`).addTo(layer);

        instance.customers.forEach(c => {
            L.circleMarker([c.y, c.x], {
                radius: 3, color: 'transparent', fillColor: '#666', fillOpacity: 0.8, renderer: mapCanvas
            }).bindTooltip(`<b>C${c.id}</b><br>Dem: ${c.demand}`).addTo(layer);
        });
        maps[solver].fitBounds(bounds, { padding: [50, 50] });
    });
}

function plotRoutes(routes, instance, solver) {
    if (!maps[solver] || !routes) return;
    const layer = layers[solver];
    const nodeMap = {};
    [instance.depot, ...instance.customers].forEach(n => nodeMap[n.id] = n);

    routes.forEach((route, i) => {
        if (route.length < 2) return;
        const latlngs = route.map(id => [nodeMap[id].y, nodeMap[id].x]);
        L.polyline(latlngs, {
            color: COLORS[i % COLORS.length], weight: 2, opacity: 0.8, renderer: mapCanvas
        }).bindTooltip(`Route ${i+1}: ${route.length} stops`).addTo(layer);
    });
}

function renderTable(data) {
    const html = `
        <div style="display:flex;justify-content:space-between;align-items:center;">
            <h3>Comparison Results</h3>
            <button onclick="document.getElementById('comparison-panel').style.display='none'" class="btn btn-secondary" style="width:auto;padding:8px 16px;">CLOSE</button>
        </div>
        <table class="comparison-table">
            <thead><tr><th>Algorithm</th><th>Distance</th><th>Time (s)</th><th>Vehicles</th><th>Coverage</th></tr></thead>
            <tbody>
                ${data.results.filter(r=>r.success).map(r => `
                    <tr>
                        <td>${r.solver_name}</td>
                        <td>${r.total_distance.toFixed(1)}</td>
                        <td>${r.execution_time.toFixed(2)}</td>
                        <td>${r.vehicles_used}</td>
                        <td>${r.coverage}</td>
                    </tr>`).join('')}
            </tbody>
        </table>`;
    document.getElementById('comparison-table-container').innerHTML = html;
    document.getElementById('comparison-panel').style.display = 'block';
    data.results.filter(r=>r.success).forEach(r => plotRoutes(r.routes, currentInstance, r.solver));
}

// --- FIX LOGIC LOAD INSTANCES ---
async function loadInstances() {
    try {
        const res = await fetch('/api/instances');
        if (!res.ok) throw new Error("Failed to fetch instance list");
        
        const data = await res.json();
        console.log('Loaded instances:', data.length, data);
        
        const select = document.getElementById('instance-select');
        if (!select) {
            console.error('Select element not found!');
            return;
        }
        
        select.innerHTML = '<option value="">-- SELECT DATASET --</option>';
        
        if (!data || data.length === 0) {
            setStatus('No RC instances found', true);
            return;
        }
        
        const groups = {};
        
        data.forEach(i => {
            // Gom các instance RC theo nhóm: rc101, rc102, ... -> RC1; rc201, rc202, ... -> RC2
            let prefix = "Others";
            const match = i.name.match(/^RC(\d)/i); // Match RC + số đầu tiên (rc101 -> RC1, rc201 -> RC2)
            if (match) {
                prefix = 'RC' + match[1]; // rc101 -> RC1, rc201 -> RC2
            }
            
            if (!groups[prefix]) groups[prefix] = [];
            groups[prefix].push(i);
        });

        console.log('Groups:', groups);

        // Sắp xếp nhóm (RC1, RC2...) và render, bỏ qua nhóm "Others"
        Object.keys(groups).sort().forEach(key => {
            if (groups[key].length === 0 || key === "Others") return;
            const group = document.createElement('optgroup');
            group.label = `Solomon ${key}`; // Label: Solomon RC1, Solomon RC2...
            
            groups[key].sort((a,b) => a.name.localeCompare(b.name, undefined, {numeric: true})).forEach(i => {
                const opt = document.createElement('option');
                opt.value = JSON.stringify(i);
                opt.textContent = i.name.toUpperCase();
                group.appendChild(opt);
            });
            select.appendChild(group);
        });
        
        if (select.children.length === 1) {
            setStatus('No RC instances found in dropdown', true);
        } else {
            setStatus('Instances loaded successfully');
        }
        
    } catch (e) {
        console.error('Error loading instances:', e);
        setStatus('Error loading list: ' + e.message, true);
    }
}

function setStatus(msg, err=false) {
    const el = document.getElementById('status');
    el.textContent = msg;
    el.style.color = err ? '#d32f2f' : '#2e7d32';
}

document.addEventListener('DOMContentLoaded', () => {
    initMaps();
    loadInstances();

    const els = {
        select: document.getElementById('instance-select'),
        loadBtn: document.getElementById('load-instance-btn'),
        runBtn: document.getElementById('run-comparison-btn'),
        clearBtn: document.getElementById('clear-btn')
    };

    els.loadBtn.onclick = async () => {
        if (!els.select.value) return;
        setStatus('Downloading data...');
        els.loadBtn.disabled = true;

        try {
            const meta = JSON.parse(els.select.value);
            
            // 1. Fetch text file
            const fileRes = await fetch(`/data/${meta.path}`);
            if (!fileRes.ok) throw new Error(`HTTP Error ${fileRes.status}`);
            const text = await fileRes.text();

            // 2. Parse text
            currentInstance = parseInstance(text);
            
            // 3. Sync backend
            await fetch('/api/load_instance', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({instance_path: `data/${meta.path}`})
            });

            // 4. Update UI
            plotInstance(currentInstance);
            document.getElementById('instance-name').textContent = meta.name.toUpperCase();
            document.getElementById('total-customers').textContent = currentInstance.customers.length;
            document.getElementById('vehicle-capacity').textContent = currentInstance.capacity;
            document.getElementById('instance-info').style.display = 'block';
            
            els.runBtn.disabled = false;
            setStatus('Ready to solve');
        } catch (e) {
            console.error(e);
            setStatus(`Error: ${e.message}`, true);
        } finally {
            els.loadBtn.disabled = false;
        }
    };

    els.runBtn.onclick = async () => {
        const meta = JSON.parse(els.select.value);
        const solvers = SOLVERS.filter(s => document.getElementById(`solver-${s}`).checked);
        if (!solvers.length) return setStatus('Select at least one solver', true);

        els.runBtn.disabled = true;
        els.runBtn.textContent = 'Processing...';
        setStatus('Running algorithms...');

        try {
            const res = await fetch('/api/compare_solvers', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    instance_path: `data/${meta.path}`,
                    instance_name: meta.name,
                    num_vehicles: document.getElementById('num-vehicles').value,
                    solvers: solvers
                })
            });
            const data = await res.json();
            if (data.error) throw new Error(data.error);
            
            comparisonData = data;
            renderTable(data);
            setStatus('Completed');
        } catch (e) {
            setStatus(e.message, true);
        } finally {
            els.runBtn.disabled = false;
            els.runBtn.textContent = 'Run Comparison';
        }
    };

    els.clearBtn.onclick = () => {
        SOLVERS.forEach(s => layers[s].clearLayers());
        document.getElementById('comparison-panel').style.display = 'none';
        document.getElementById('instance-info').style.display = 'none';
        els.runBtn.disabled = true;
        setStatus('');
    };
});