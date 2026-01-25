# VRPTW Hybrid Solver

Hybrid solver for Vehicle Routing Problem with Time Windows (VRPTW) combining Deep Q-Network (DQN) warm-start with Adaptive Large Neighborhood Search (ALNS).

## Method

- Stage 1: Attention-based DQN constructs initial solution
- Stage 2: ALNS refines the solution

Reference: He et al. (2021) arXiv:2103.05847

## Installation

```bash
python -m venv venv
.\venv\Scripts\activate
pip install uv
uv init .
uv pip install -r requirements.txt
uv lock
```

For GPU support:
```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Usage

```bash
python main.py
```

Open http://127.0.0.1:8000 in browser.

## Structure

```
src/
  core/           # Algorithm implementations
    vrptw_types.py  - Data classes
    alns_solver.py  - ALNS algorithm
    dqn_model.py    - DQN network
    hybrid_solver.py - DQN + ALNS
  api/            # FastAPI endpoints
    routes.py       - API routes
    schemas.py      - Request/Response models
  web/            # Google Material Design UI
    index.html, styles.css, script.js
data/Solomon/    # Benchmark instances
models/          # Pretrained weights
```

## License

MIT
