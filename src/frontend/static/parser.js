// parser.js
// Parse VRPTW instance file content (Solomon / Gehring-Homberger format) into JS objects

function parseInstance(content) {
    const lines = content.split(/\r?\n/).map(l => l.trim()).filter(l => l.length > 0);
    let idx = 0;
    // Skip to VEHICLE
    while (idx < lines.length && !lines[idx].startsWith('VEHICLE')) idx++;
    idx += 2; // skip header
    const [numVehicles, vehicleCapacity] = lines[idx++].split(/\s+/).map(Number);
    // Skip to CUSTOMER
    while (idx < lines.length && !lines[idx].startsWith('CUSTOMER')) idx++;
    idx += 2;
    const customers = [];
    for (; idx < lines.length; idx++) {
        const parts = lines[idx].split(/\s+/);
        if (parts.length < 7) continue;
        const [id, x, y, demand, ready, due, service] = parts.map(Number);
        customers.push({ id, x, y, demand, ready, due, service });
    }
    return {
        numVehicles,
        vehicleCapacity,
        depot: customers[0],
        customers: customers.slice(1)
    };
}


