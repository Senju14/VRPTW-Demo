function parseInstance(content) {
    if (!content) throw new Error("File content is empty");

    // Chuẩn hóa: Thay thế tab bằng space, xóa dòng trống
    const lines = content.replace(/\t/g, ' ').split(/\r?\n/)
                        .map(l => l.trim()).filter(l => l);

    // Tìm header
    const vIdx = lines.findIndex(l => l.toUpperCase().includes('VEHICLE'));
    const cIdx = lines.findIndex(l => l.toUpperCase().includes('CUSTOMER'));

    if (vIdx === -1 || cIdx === -1) throw new Error("Invalid file format: Missing sections");

    // Parse Vehicle Info (thường cách dòng VEHICLE 2 dòng)
    // Dùng regex để bắt số liệu bất kể khoảng cách
    const vLine = lines[vIdx + 2] || "";
    const vData = vLine.split(/\s+/).map(Number).filter(n => !isNaN(n));
    
    // Parse Customer Nodes
    const nodes = [];
    for (let i = cIdx + 2; i < lines.length; i++) {
        const parts = lines[i].split(/\s+/).map(Number);
        // Dòng hợp lệ phải có ít nhất 7 số (ID, X, Y, Demand, Ready, Due, Service)
        if (parts.length >= 7 && !isNaN(parts[0])) {
            nodes.push({
                id: parts[0], x: parts[1], y: parts[2],
                demand: parts[3], ready: parts[4], due: parts[5], service: parts[6]
            });
        }
    }

    if (nodes.length === 0) throw new Error("No customer data found");

    return {
        numVehicles: vData[0] || 25, // Fallback nếu lỗi
        capacity: vData[1] || 200,
        depot: nodes[0],
        customers: nodes.slice(1)
    };
}