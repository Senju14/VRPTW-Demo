# 🚛 VRPTW Demo - Vehicle Routing Problem with Time Windows

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0%2B-green)](https://flask.palletsprojects.com/)
[![OR-Tools](https://img.shields.io/badge/OR--Tools-9.7%2B-orange)](https://developers.google.com/optimization)

A comprehensive web-based demonstration application for solving the **Vehicle Routing Problem with Time Windows (VRPTW)** using advanced optimization algorithms and interactive visualization.

## 📋 Project Overview

This application provides an academic-standard interface to solve and visualize VRPTW instances using Google's OR-Tools optimization library. The system features:

- **Interactive Web Interface**: Modern HTML/CSS/JS frontend with Google Maps-like visualization
- **Multiple Dataset Support**: Solomon benchmark instances + Gehring-Homberger datasets
- **Learn-to-Solve Integration**: Instance-specific DQN model selection for enhanced performance
- **Real-time Visualization**: Interactive map showing optimized routes in Vietnam coordinates
- **Professional UI**: Clean white interface with dark buttons and responsive design

## 🏗️ Architecture & Folder Structure

```
VRPTW_Demo/
├── src/                          # Source code directory
│   ├── backend/                  # Flask API server
│   │   └── app.py               # Main Flask application with API endpoints
│   ├── frontend/                # Web frontend components
│   │   ├── index.html          # Main HTML page with Leaflet integration
│   │   └── static/             # Frontend assets
│   │       ├── style.css       # Professional CSS with glass morphism
│   │       ├── app.js          # JavaScript logic & map visualization
│   │       └── solomon-parser.js # Instance file parser
│   └── utils/                   # Utility modules
│       ├── data_loader.py      # Multi-format instance loader
│       ├── solver.py           # OR-Tools VRPTW solver with constraints
│       └── visualization.py    # Folium map generation (legacy)
├── data/                        # Benchmark datasets
│   ├── Solomon/                 # Solomon benchmark instances (25-100 customers)
│   │   ├── c101.txt - c109.txt # Clustered customers
│   │   ├── r101.txt - r112.txt # Random customers  
│   │   ├── c201.txt - c208.txt # Clustered with long time windows
│   │   ├── r201.txt - r211.txt # Random with long time windows
│   │   └── rc101.txt - rc208.txt# Mixed clustered/random
│   └── Gehring_Homberger/       # Extended Gehring-Homberger instances
│       ├── homberger_200_customer_instances/
│       ├── homberger_400_customer_instances/
│       ├── homberger_600_customer_instances/
│       ├── homberger_800_customer_instances/
│       └── homberger_1000_customer_instances/
├── models/                      # Pre-trained DQN models (Learn-to-Solve)
│   ├── dqn_c101.safetensor     # Solomon instance models
│   ├── dqn_200_C1_2_1.safetensor # Gehring 200-customer models
│   └── dqn_1000_C1_10_1.safetensor # Gehring 1000-customer models
├── .venv/                       # Virtual environment (auto-generated)
├── main.py                      # Application entry point
├── pyproject.toml              # Modern Python project configuration
└── requirements.txt            # Package dependencies
```

## 🚀 Installation & Setup

### Prerequisites
- **Python 3.10+** (Required for modern OR-Tools compatibility)
- **Git** (For cloning repository)

### Step-by-Step Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd VRPTW_Demo
   ```

2. **Create and activate virtual environment:**
   ```bash
   # Create .venv environment
   python -m venv .venv
   
   # Activate (Windows)
   .\.venv\Scripts\Activate.ps1
   
   # Activate (Linux/Mac)
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   # Method 1: Using pip
   pip install -r requirements.txt
   
   # Method 2: Using uv (faster)
   pip install uv
   uv pip install -r requirements.txt
   ```

4. **Verify installation:**
   ```bash
   python -c "import ortools; print('OR-Tools version:', ortools.__version__)"
   ```

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| **Flask** | ≥3.0.0 | Web framework & API server |
| **Flask-CORS** | ≥4.0.0 | Cross-origin resource sharing |
| **OR-Tools** | ≥9.7.0 | Optimization solver engine |
| **Folium** | ≥0.14.0 | Map visualization (legacy support) |
| **Pandas** | ≥2.0.0 | Data manipulation |

## 🎮 Usage Guide

### Starting the Application

```bash
# Ensure virtual environment is activated
.\.venv\Scripts\Activate.ps1  # Windows
# source .venv/bin/activate   # Linux/Mac

# Start the Flask server
python main.py
```

### Accessing the Interface

1. **Open web browser** and navigate to: `http://127.0.0.1:5000`
2. ⚠️ **Important**: Do not open `index.html` directly - it requires the Flask server for API calls and data serving

### Using the Demo

1. **Select Instance**: Choose from organized dropdown with instance groups:
   - **Solomon Benchmark**: Classic VRPTW instances (c101, r101, rc101...)
   - **Gehring-Homberger (200-1000 customers)**: Larger scale instances

2. **Model Indicators**:
   - **✓ DQN**: Instance has trained Learn-to-Solve model
   - **(OR-Tools)**: Uses classical optimization fallback

3. **Load Instance**: Click "Load Instance" to visualize customers on Vietnam map

4. **Configure Parameters**:
   - **Number of Vehicles**: Adjust fleet size (default: auto-calculated)
   - System auto-selects appropriate DQN model or OR-Tools solver

5. **Run Inference**: Click "Run Inference" to solve and visualize optimal routes

### Key Features

- **🗺️ Interactive Map**: Vietnam coordinates with Ho Chi Minh City area focus
- **🎯 Smart Model Selection**: Automatic DQN model matching for Learn-to-Solve approach
- **📊 Real-time Metrics**: Distance, time, coverage, and vehicle utilization
- **🎨 Professional UI**: Clean white background with dark button styling
- **⚡ Performance Optimized**: Dynamic constraints and solver parameters

## 🧠 Learn-to-Solve Architecture

The system implements an **instance-specific learning approach** where:

### Model Selection Logic
```python
# Solomon instances: c101 → dqn_c101.safetensor
# Gehring 200: C1_2_1 → dqn_200_C1_2_1.safetensor  
# Gehring 1000: C1_10_1 → dqn_1000_C1_10_1.safetensor
```

### Fallback Strategy
- If **DQN model exists**: Use trained neural network solution
- If **no model available**: Fallback to OR-Tools classical optimization
- **Hybrid approach**: Combine learned heuristics with constraint solving

## 🔧 Configuration & Customization

### Solver Parameters (in `src/utils/solver.py`)

```python
# Time constraints
max_time = max([tw[1] for tw in data['time_windows']]) + 1000

# Vehicle capacity with slack
capacity_slack = data['vehicle_capacities'][0] // 10

# Search parameters
search_parameters.time_limit.seconds = 60
search_parameters.solution_limit = 100
```

### Map Visualization (in `src/frontend/static/app.js`)

```javascript
// Vietnam coordinate conversion
const convertCoord = (x, y) => {
    // Normalize and scale to Ho Chi Minh City area
    const lat = 10.6 + normalizedY * 0.4;  // 10.6-11.0
    const lng = 106.4 + normalizedX * 0.6; // 106.4-107.0
    return [lat, lng];
};
```

## 🐛 Troubleshooting

### Common Issues

1. **"ModuleNotFoundError: No module named 'src'"**
   ```bash
   # Set Python path
   export PYTHONPATH=".":$PYTHONPATH  # Linux/Mac
   $env:PYTHONPATH=".";               # Windows PowerShell
   ```

2. **"customers is not defined" (JavaScript)**
   - Fixed in latest version - coordinate conversion uses `instance.customers`

3. **HTTP 400 errors on solve**
   - Ensure proper instance loading and solver constraints
   - Check terminal for detailed error messages

4. **No routes visualized**
   - Increase number of vehicles or time window multiplier
   - Check solver time limits and constraint relaxation

### Debug Mode

```bash
# Enable Flask debug mode (in main.py)
app.run(debug=True, host='127.0.0.1', port=5000)
```

## 📈 Performance Notes

- **Small instances (≤100 customers)**: Solve in seconds
- **Large instances (1000+ customers)**: May take 1-2 minutes
- **Memory usage**: Scales with instance size and number of vehicles
- **Browser compatibility**: Modern browsers with JavaScript ES6+ support

## 🤝 Contributing

This is an academic demonstration project. For contributions:

1. Follow Python PEP 8 style guidelines
2. Maintain separation between backend (Flask) and frontend (HTML/JS)
3. Document new solver parameters and constraints
4. Test with multiple instance types and sizes

## 📄 License & Citation

This project is for educational and research purposes. When using this code:

- Cite the OR-Tools library for optimization components
- Reference Solomon and Gehring-Homberger datasets appropriately
- Acknowledge the Learn-to-Solve methodology for DQN integration

---

**🎯 Ready to solve complex vehicle routing problems with modern web technology and advanced optimization!**