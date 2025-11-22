// app.js - VRPTW Demo with Flask Backend Integration

let map, markers = [], polylines = [], currentInstance = null;

function initMap() {
    map = L.map('map', {
        zoomControl: false
    }).setView([10.8231, 106.6297], 10);
    
    // Light tile layer
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        maxZoom: 19,
        attribution: '© OpenStreetMap'
    }).addTo(map);
    
    // Add zoom control to top-right
    L.control.zoom({
        position: 'topright'
    }).addTo(map);
}

function clearMap() {
    markers.forEach(m => map.removeLayer(m));
    polylines.forEach(l => map.removeLayer(l));
    markers = [];
    polylines = [];
}

function plotInstance(instance) {
    clearMap();
    
    // Convert coordinates to Vietnam area (scale and offset)
    const convertCoord = (x, y) => {
        // Find min/max from all points to normalize properly
        const allX = [instance.depot.x, ...instance.customers.map(c => c.x)];
        const allY = [instance.depot.y, ...instance.customers.map(c => c.y)];
        const minX = Math.min(...allX);
        const maxX = Math.max(...allX);
        const minY = Math.min(...allY);
        const maxY = Math.max(...allY);
        
        // Normalize to 0-1 range, then scale to Vietnam coordinates
        const normalizedX = (x - minX) / (maxX - minX || 1);
        const normalizedY = (y - minY) / (maxY - minY || 1);
        
        // Ho Chi Minh City area: lat 10.6-11.0, lng 106.4-107.0
        const lat = 10.6 + normalizedY * 0.4;  
        const lng = 106.4 + normalizedX * 0.6;
        return [lat, lng];
    };
    
    // Plot depot with custom styling
    const depot = instance.depot;
    const customers = instance.customers;
    const [depotLat, depotLng] = convertCoord(depot.x, depot.y);
    const depotMarker = L.circleMarker([depotLat, depotLng], {
        radius: 12,
        color: '#fff',
        fillColor: '#e74c3c',
        fillOpacity: 1,
        weight: 3
    }).addTo(map);
    depotMarker.bindPopup('<div style="text-align:center;font-weight:bold;color:#e74c3c;">🏢 DEPOT</div>');
    markers.push(depotMarker);
    
    // Plot customers with beautiful styling
    customers.forEach(c => {
        const [custLat, custLng] = convertCoord(c.x, c.y);
        const m = L.circleMarker([custLat, custLng], {
            radius: 8,
            color: '#fff',
            fillColor: '#3498db',
            fillOpacity: 0.9,
            weight: 2
        }).addTo(map);
        m.bindPopup(`
            <div style="text-align:center;font-size:14px;">
                <div style="font-weight:bold;color:#2c3e50;margin-bottom:8px;">👤 Customer ${c.id}</div>
                <div style="color:#7f8c8d;">📦 Demand: <span style="color:#e74c3c;font-weight:bold;">${c.demand}</span></div>
                <div style="color:#7f8c8d;">⏰ Window: <span style="color:#27ae60;font-weight:bold;">[${c.ready_time}, ${c.due_date}]</span></div>
            </div>
        `);
        markers.push(m);
    });
    
    if (markers.length > 0) {
        const group = L.featureGroup(markers);
        map.fitBounds(group.getBounds().pad(0.1));
        
        // Ensure minimum zoom for small instances
        if (map.getZoom() > 12) {
            map.setZoom(12);
        }
    }
}

function plotRoutes(routes, instance) {
    // Clear existing routes
    polylines.forEach(l => map.removeLayer(l));
    polylines = [];
    
    // Convert coordinates function (same as in plotInstance)
    const convertCoord = (x, y) => {
        // Get all coordinates for proper normalization
        const allNodes = [instance.depot, ...instance.customers];
        const allX = allNodes.map(n => n.x);
        const allY = allNodes.map(n => n.y);
        const minX = Math.min(...allX);
        const maxX = Math.max(...allX);
        const minY = Math.min(...allY);
        const maxY = Math.max(...allY);
        
        // Normalize and scale to Vietnam coordinates
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
        
        // Create animated route with glow effect
        const poly = L.polyline(latlngs, {
            color: color,
            weight: 5,
            opacity: 0.9,
            className: 'route-line'
        }).addTo(map);
        
        // Add shadow effect
        const shadow = L.polyline(latlngs, {
            color: color,
            weight: 8,
            opacity: 0.3
        }).addTo(map);
        
        poly.bindPopup(`
            <div style="text-align:center;font-size:14px;">
                <div style="font-weight:bold;color:${color};margin-bottom:8px;">🚛 Route ${i + 1}</div>
                <div style="color:#7f8c8d;">📍 Stops: <span style="color:#2c3e50;font-weight:bold;">${route.length}</span></div>
            </div>
        `);
        
        polylines.push(shadow);
        polylines.push(poly);
    });
}

function updateMetrics(results) {
    document.getElementById('total-distance').textContent = results.total_distance.toFixed(2);
    document.getElementById('total-time').textContent = results.execution_time.toFixed(2) + 's';
    document.getElementById('avg-distance').textContent = results.avg_distance.toFixed(2);
    document.getElementById('vehicles-used').textContent = results.vehicles_used;
    document.getElementById('customers-served').textContent = results.customers_served;
    document.getElementById('coverage').textContent = results.coverage;
}

function resetMetrics() {
    document.getElementById('total-distance').textContent = '-';
    document.getElementById('total-time').textContent = '-';
    document.getElementById('avg-distance').textContent = '-';
    document.getElementById('vehicles-used').textContent = '-';
    document.getElementById('customers-served').textContent = '-';
    document.getElementById('coverage').textContent = '-';
}

function setStatus(message, isError = false) {
    const statusEl = document.getElementById('status');
    statusEl.textContent = message;
    statusEl.style.color = isError ? '#d93025' : '#34a853';
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
                    
                    // Add model indicator
                    const modelIndicator = inst.has_model ? ' ✓ DQN' : ' (OR-Tools)';
                    option.textContent = `${inst.name}${modelIndicator}`;
                    
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
            option.textContent = name.toUpperCase() + ' ✓ DQN';
            select.appendChild(option);
        });
    }
}

document.addEventListener('DOMContentLoaded', function() {
    initMap();
    
    const instanceSelect = document.getElementById('instance-select');
    const loadBtn = document.getElementById('load-instance-btn');
    const runBtn = document.getElementById('run-inference-btn');
    const clearBtn = document.getElementById('clear-btn');
    
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
            currentInstance = parseSolomon(content);
            
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
            document.getElementById('instance-info').style.display = 'block';
            document.getElementById('instance-name').textContent = instanceData.name.toUpperCase();
            document.getElementById('total-customers').textContent = currentInstance.customers.length;
            document.getElementById('vehicle-capacity').textContent = currentInstance.vehicleCapacity || 'N/A';
            
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
        
        const instanceData = JSON.parse(selectedValue);
        const numVehicles = parseInt(document.getElementById('num-vehicles').value) || 5;
        
        setStatus(`Solving VRPTW using ${instanceData.has_model ? 'DQN Model (Learn-to-Solve)' : 'OR-Tools'}...`);
        runBtn.disabled = true;
        runBtn.textContent = 'Solving...';
        
        try {
            const response = await fetch('/api/solve_demo', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    instance_path: `data/${instanceData.path}`,
                    instance_name: instanceData.name,
                    num_vehicles: numVehicles
                })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            
            const results = await response.json();
            
            if (results.error) {
                throw new Error(results.error);
            }
            
            // Plot routes
            plotRoutes(results.routes, currentInstance);
            updateMetrics(results);
            
            // Display model information
            const modelInfo = results.using_learned_model ? 
                `✓ DQN Model: ${results.model_used}` : 
                `OR-Tools (No specific model available)`;
            setStatus(`Solution found! ${modelInfo}`);
            
        } catch (error) {
            console.error('Error solving:', error);
            setStatus('Error solving: ' + error.message, true);
        } finally {
            runBtn.disabled = false;
            runBtn.textContent = 'Run Inference';
        }
    };
    
    clearBtn.onclick = function() {
        clearMap();
        document.getElementById('instance-info').style.display = 'none';
        runBtn.disabled = true;
        resetMetrics();
        setStatus('');
        currentInstance = null;
    };
});