// app.js - VRPTW Demo with Flask Backend Integration

let maps = {};
let markers = {};
let polylines = {};
let currentInstance = null;
   
const SOLVER_MAP_IDS = {
    'alns': 'map-alns',
    'dqn': 'map-dqn',
    'dqn_alns': 'map-dqn-alns',
    'ortools': 'map-ortools'
};

function initMaps() {
    Object.keys(SOLVER_MAP_IDS).forEach(solver => {
        const mapId = SOLVER_MAP_IDS[solver];
        maps[solver] = L.map(mapId, {
            zoomControl: false,        // disable default top-left control
            attributionControl: false
        }).setView([10.8231, 106.6297], 10);

        // Add zoom control at bottom-right
        L.control.zoom({ position: 'bottomright' }).addTo(maps[solver]);
        
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            maxZoom: 19,
            attribution: '© OpenStreetMap'
        }).addTo(maps[solver]);
        
        markers[solver] = [];
        polylines[solver] = [];
    });
}

function clearMap(solver) {
    if (!maps[solver]) return;
    markers[solver].forEach(m => maps[solver].removeLayer(m));
    polylines[solver].forEach(l => maps[solver].removeLayer(l));
    markers[solver] = [];
    polylines[solver] = [];
}

function clearAllMaps() {
    Object.keys(SOLVER_MAP_IDS).forEach(solver => clearMap(solver));
}

function plotInstanceOnMap(instance, solver) {
    if (!maps[solver]) return;
    clearMap(solver);
    
    const convertCoord = (x, y) => {
        const allX = [instance.depot.x, ...instance.customers.map(c => c.x)];
        const allY = [instance.depot.y, ...instance.customers.map(c => c.y)];
        const minX = Math.min(...allX);
        const maxX = Math.max(...allX);
        const minY = Math.min(...allY);
        const maxY = Math.max(...allY);
        const normalizedX = (x - minX) / (maxX - minX || 1);
        const normalizedY = (y - minY) / (maxY - minY || 1);
        const lat = 10.6 + normalizedY * 0.4;
        const lng = 106.4 + normalizedX * 0.6;
        return [lat, lng];
    };
    
    const depot = instance.depot;
    const customers = instance.customers;
    const [depotLat, depotLng] = convertCoord(depot.x, depot.y);
    const depotMarker = L.circleMarker([depotLat, depotLng], {
        radius: 12,
        color: '#fff',
        fillColor: '#e74c3c',
        fillOpacity: 1,
        weight: 3
    }).addTo(maps[solver]);
    depotMarker.bindPopup('<div style="text-align:center;font-weight:bold;color:#e74c3c;">🏢 DEPOT</div>');
    markers[solver].push(depotMarker);
    
    customers.forEach(c => {
        const [custLat, custLng] = convertCoord(c.x, c.y);
        const m = L.circleMarker([custLat, custLng], {
            radius: 8,
            color: '#fff',
            fillColor: '#3498db',
            fillOpacity: 0.9,
            weight: 2
        }).addTo(maps[solver]);
        m.bindPopup(`
            <div style="text-align:center;font-size:14px;">
                <div style="font-weight:bold;color:#2c3e50;margin-bottom:8px;">👤 Customer ${c.id}</div>
                <div style="color:#7f8c8d;">📦 Demand: <span style="color:#e74c3c;font-weight:bold;">${c.demand}</span></div>
                <div style="color:#7f8c8d;">⏰ Window: <span style="color:#27ae60;font-weight:bold;">[${c.ready_time}, ${c.due_date}]</span></div>
            </div>
        `);
        markers[solver].push(m);
    });
    
    if (markers[solver].length > 0) {
        const group = L.featureGroup(markers[solver]);
        maps[solver].fitBounds(group.getBounds().pad(0.1));
        if (maps[solver].getZoom() > 12) maps[solver].setZoom(12);
    }
}

function plotInstance(instance) {
    Object.keys(SOLVER_MAP_IDS).forEach(solver => plotInstanceOnMap(instance, solver));
}

function plotRoutesOnMap(routes, instance, solver) {
    if (!maps[solver] || !routes) return;
    polylines[solver].forEach(l => maps[solver].removeLayer(l));
    polylines[solver] = [];
    
    const convertCoord = (x, y) => {
        const allNodes = [instance.depot, ...instance.customers];
        const allX = allNodes.map(n => n.x);
        const allY = allNodes.map(n => n.y);
        const minX = Math.min(...allX);
        const maxX = Math.max(...allX);
        const minY = Math.min(...allY);
        const maxY = Math.max(...allY);
        const normalizedX = (x - minX) / (maxX - minX || 1);
        const normalizedY = (y - minY) / (maxY - minY || 1);
        const lat = 10.6 + normalizedY * 0.4;
        const lng = 106.4 + normalizedX * 0.6;
        return [lat, lng];
    };
    
    const colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22', '#34495e'];
    const allNodes = [instance.depot, ...instance.customers];
    
    routes.forEach((route, i) => {
        if (route.length < 2) return;
        const color = colors[i % colors.length];
        const latlngs = route.map(nodeId => {
            const node = allNodes.find(n => n.id === nodeId);
            const [lat, lng] = convertCoord(node.x, node.y);
            return [lat, lng];
        });
        
        const poly = L.polyline(latlngs, {
            color: color,
            weight: 5,
            opacity: 0.9,
            className: 'route-line'
        }).addTo(maps[solver]);
        
        const shadow = L.polyline(latlngs, {
            color: color,
            weight: 8,
            opacity: 0.3
        }).addTo(maps[solver]);
        
        poly.bindPopup(`
            <div style="text-align:center;font-size:14px;">
                <div style="font-weight:bold;color:${color};margin-bottom:8px;">🚛 Route ${i + 1}</div>
                <div style="color:#7f8c8d;">📍 Stops: <span style="color:#2c3e50;font-weight:bold;">${route.length}</span></div>
            </div>
        `);
        
        polylines[solver].push(shadow);
        polylines[solver].push(poly);
    });
}

function updateMetrics(results) {
    // Metrics are now shown in comparison table, not individual elements
    // This function kept for backward compatibility but does nothing
}

function resetMetrics() {
    // Metrics are now shown in comparison table, not individual elements
    // This function kept for backward compatibility but does nothing
}

function setStatus(message, isError = false) {
    const statusEl = document.getElementById('status');
    statusEl.textContent = message;
    statusEl.style.color = isError ? '#d93025' : '#34a853';
}

let comparisonData = null;

function displayComparisonResults(data) {
    comparisonData = data;
    const panel = document.getElementById('comparison-panel');
    if (panel) {
        panel.classList.remove('fullscreen'); // start inline so it won't cover maps
        panel.style.width = '';
        panel.style.height = '';
        panel.style.display = 'block';
    }
    
    const container = document.getElementById('comparison-table-container');
    const table = document.createElement('table');
    table.className = 'comparison-table';
    
    // Header
    const thead = document.createElement('thead');
    const headerRow = document.createElement('tr');
    ['Solver', 'Distance', 'Time (s)', 'Vehicles', 'Customers', 'Coverage (%)', 'Avg Dist'].forEach(h => {
        const th = document.createElement('th');
        th.textContent = h;
        headerRow.appendChild(th);
    });
    thead.appendChild(headerRow);
    table.appendChild(thead);
    
    // Body
    const tbody = document.createElement('tbody');
    const results = data.results.filter(r => r.success);
    
    if (results.length === 0) {
        const row = document.createElement('tr');
        const cell = document.createElement('td');
        cell.colSpan = 7;
        cell.textContent = 'No successful results';
        cell.style.textAlign = 'center';
        row.appendChild(cell);
        tbody.appendChild(row);
    } else {
        // Find best values
        const distances = results.map(r => r.total_distance).filter(d => d !== undefined);
        const times = results.map(r => r.execution_time).filter(t => t !== undefined);
        const minDist = Math.min(...distances);
        const minTime = Math.min(...times);
        
        results.forEach(result => {
            const row = document.createElement('tr');
            
            // Solver name
            const solverCell = document.createElement('td');
            solverCell.textContent = result.solver_name || result.solver;
            solverCell.style.fontWeight = '600';
            row.appendChild(solverCell);
            
            // Total Distance
            const distCell = document.createElement('td');
            distCell.textContent = result.total_distance?.toFixed(2) || '-';
            if (result.total_distance === minDist) distCell.className = 'best';
            row.appendChild(distCell);
            
            // Execution Time
            const timeCell = document.createElement('td');
            timeCell.textContent = result.execution_time?.toFixed(2) || '-';
            if (result.execution_time === minTime) timeCell.className = 'best';
            row.appendChild(timeCell);
            
            // Vehicles Used
            const vehiclesCell = document.createElement('td');
            vehiclesCell.textContent = result.vehicles_used || '-';
            row.appendChild(vehiclesCell);
            
            // Customers Served
            const customersCell = document.createElement('td');
            customersCell.textContent = result.customers_served || '-';
            row.appendChild(customersCell);
            
            // Coverage
            const coverageCell = document.createElement('td');
            coverageCell.textContent = result.coverage ? `${result.coverage.toFixed(1)}%` : '-';
            row.appendChild(coverageCell);
            
            // Avg Distance
            const avgCell = document.createElement('td');
            avgCell.textContent = result.avg_distance ? result.avg_distance.toFixed(2) : '-';
            row.appendChild(avgCell);
            
            tbody.appendChild(row);
        });
    }
    
    table.appendChild(tbody);
    container.innerHTML = '';
    container.appendChild(table);
    
    // Plot routes for each solver on its map
    results.forEach(result => {
        if (result.success && result.routes && SOLVER_MAP_IDS[result.solver] && currentInstance) {
            plotRoutesOnMap(result.routes, currentInstance, result.solver);
        }
    });
}

function exportComparison(format) {
    if (!comparisonData) {
        setStatus('No comparison data to export', true);
        return;
    }
    
    fetch('/api/export_comparison', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            format: format,
            comparison_data: comparisonData
        })
    })
    .then(response => {
        if (!response.ok) throw new Error('Export failed');
        return response.blob();
    })
    .then(blob => {
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `vrptw_comparison_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.${format}`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
        setStatus(`Exported ${format.toUpperCase()} successfully`);
    })
    .catch(error => {
        console.error('Export error:', error);
        setStatus('Export failed: ' + error.message, true);
    });
}

async function loadAvailableInstances() {
    try {
        const response = await fetch('/api/instances');
        if (response.ok) {
            const instances = await response.json();
            const select = document.getElementById('instance-select');
            select.innerHTML = '<option value="">Select instance...</option>';
            
            // Group instances by category
            const groups = {};
            instances.forEach(inst => {
                if (!groups[inst.group]) {
                    groups[inst.group] = [];
                }
                groups[inst.group].push(inst);
            });
            
            // Create optgroups for better organization
            Object.keys(groups).forEach(groupName => {
                const optgroup = document.createElement('optgroup');
                optgroup.label = groupName;
                
                groups[groupName].forEach(inst => {
                    const option = document.createElement('option');
                    option.value = JSON.stringify({
                        name: inst.name,
                        path: inst.path,
                        model_path: inst.model_path,
                        has_model: inst.has_model
                    });
                    
                    // Only show name (hide model indicator)
                    option.textContent = inst.name;
                    
                    optgroup.appendChild(option);
                });
                
                select.appendChild(optgroup);
            });
        }
    } catch (error) {
        console.error('Error loading instances:', error);
        // Fallback to hardcoded instances
        const select = document.getElementById('instance-select');
        const defaultInstances = ['c101', 'c102', 'c201', 'r101', 'r102', 'r201', 'rc101', 'rc102', 'rc201'];
        select.innerHTML = '<option value="">Select instance...</option>';
        defaultInstances.forEach(name => {
            const option = document.createElement('option');
            option.value = JSON.stringify({name: name, path: `Solomon/${name}.txt`, has_model: true});
            option.textContent = name.toUpperCase();
            select.appendChild(option);
        });
    }
}

document.addEventListener('DOMContentLoaded', function() {
    initMaps();
    
    const instanceSelect = document.getElementById('instance-select');
    const loadBtn = document.getElementById('load-instance-btn');
    const runBtn = document.getElementById('run-comparison-btn');
    const clearBtn = document.getElementById('clear-btn');
    const exportCsvBtn = document.getElementById('export-csv-btn');
    const exportJsonBtn = document.getElementById('export-json-btn');
    const expandBtn = document.getElementById('expand-comparison-btn');
    
    // Load available instances from server
    loadAvailableInstances();
    
    loadBtn.onclick = async function() {
        const selectedValue = instanceSelect.value;
        if (!selectedValue) {
            setStatus('Please select an instance', true);
            return;
        }
        
        const instanceData = JSON.parse(selectedValue);
        setStatus('Loading instance...');
        loadBtn.disabled = true;
        
        try {
            // Fetch instance file from server
            const resp = await fetch(`/data/${instanceData.path}`);
            if (!resp.ok) {
                throw new Error('Failed to load instance file');
            }
            
            const content = await resp.text();
            currentInstance = parseInstance(content);
            
            // Also get instance data from backend for consistency
            const backendResp = await fetch('/api/load_instance', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ instance_path: `data/${instanceData.path}` })
            });
            
            if (backendResp.ok) {
                const backendData = await backendResp.json();
                // Use backend data if available
                currentInstance = {
                    depot: backendData.depot,
                    customers: backendData.customers,
                    vehicleCapacity: backendData.capacity,
                    numVehicles: backendData.num_vehicles
                };
            }
            
            plotInstance(currentInstance);
            
            // Update instance info
            const instanceInfo = document.getElementById('instance-info');
            const instanceName = document.getElementById('instance-name');
            const totalCustomers = document.getElementById('total-customers');
            const vehicleCapacity = document.getElementById('vehicle-capacity');
            
            if (instanceInfo) instanceInfo.style.display = 'block';
            if (instanceName) instanceName.textContent = instanceData.name.toUpperCase();
            if (totalCustomers) totalCustomers.textContent = currentInstance.customers.length;
            if (vehicleCapacity) vehicleCapacity.textContent = currentInstance.vehicleCapacity || 'N/A';
            
            runBtn.disabled = false;
            resetMetrics();
            setStatus('Instance loaded successfully');
            
        } catch (error) {
            console.error('Error loading instance:', error);
            setStatus('Error loading instance: ' + error.message, true);
        } finally {
            loadBtn.disabled = false;
        }
    };
    
    runBtn.onclick = async function() {
        if (!currentInstance) {
            setStatus('No instance loaded', true);
            return;
        }
        
        const selectedValue = instanceSelect.value;
        if (!selectedValue) {
            setStatus('Please select an instance', true);
            return;
        }
        
        // Get selected solvers
        const selectedSolvers = [];
        ['solver-alns', 'solver-dqn', 'solver-dqn-alns', 'solver-ortools'].forEach(id => {
            const checkbox = document.getElementById(id);
            if (checkbox && checkbox.checked) {
                selectedSolvers.push(checkbox.value);
            }
        });
        
        if (selectedSolvers.length === 0) {
            setStatus('Please select at least one solver', true);
            return;
        }
        
        const instanceData = JSON.parse(selectedValue);
        const numVehicles = parseInt(document.getElementById('num-vehicles').value) || 5;
        
        setStatus(`Running comparison with ${selectedSolvers.length} solver(s)...`);
        runBtn.disabled = true;
        runBtn.textContent = 'Running...';
        
        try {
            const response = await fetch('/api/compare_solvers', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    instance_path: `data/${instanceData.path}`,
                    instance_name: instanceData.name,
                    num_vehicles: numVehicles,
                    solvers: selectedSolvers
                })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            
            const data = await response.json();
            
            if (data.error) {
                throw new Error(data.error);
            }
            
            displayComparisonResults(data);
            const successCount = data.results.filter(r => r.success).length;
            setStatus(`Comparison completed! ${successCount}/${data.results.length} solvers succeeded`);
            
        } catch (error) {
            console.error('Error running comparison:', error);
            setStatus('Error: ' + error.message, true);
        } finally {
            runBtn.disabled = false;
            runBtn.textContent = 'Run Comparison';
        }
    };
    
    exportCsvBtn.onclick = () => exportComparison('csv');
    if (expandBtn) {
        expandBtn.onclick = () => {
            const panel = document.getElementById('comparison-panel');
            if (panel) {
                const isFull = panel.classList.toggle('fullscreen');
                if (isFull) {
                    panel.style.width = '70vw';
                    panel.style.height = '70vh';
                } else {
                    panel.style.width = '';
                    panel.style.height = '';
                }
            }
        };
    }
    
    clearBtn.onclick = function() {
        clearAllMaps();
        const instanceInfo = document.getElementById('instance-info');
        const comparisonPanel = document.getElementById('comparison-panel');
        if (instanceInfo) instanceInfo.style.display = 'none';
        if (comparisonPanel) comparisonPanel.style.display = 'none';
        runBtn.disabled = true;
        resetMetrics();
        setStatus('');
        currentInstance = null;
        comparisonData = null;
    };
});