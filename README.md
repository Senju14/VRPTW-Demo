# VRPTW Route Planner

Hybrid solver for Vehicle Routing Problem with Time Windows combining DQN warm-start with ALNS refinement.

## Features

- **Solomon Benchmark**: Test against standard RC datasets
- **Real-world Operations**: Interactive map planning for Vietnam logistics
- **History**: Track and compare past optimization runs

## Quick Start

```bash
python -m venv venv
.\venv\Scripts\activate
uv pip install -r requirements.txt
python main.py
```

Open http://127.0.0.1:8000

## Structure

```
src/
  core/       # Algorithm (ALNS, DQN, Hybrid)
  api/        # FastAPI endpoints
  web/        # Frontend (Material Design)
```

## Reference

He et al. (2021) arXiv:2103.05847
