"""
Evaluation Script for DQN-ALNS on Solomon Benchmark.

Runs trained model on standard Solomon RC instances and compares
against ALNS baseline and Best Known Solutions (BKS).

Features:
- Statistical validation (multiple runs with different seeds)
- Gap to BKS calculation
- Comparison tables and visualizations
"""

import os
import time
import argparse
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd

import torch

from .dqn_alns import DQNConfig, DQNALNSSolver, DuelingDQN
from ..baseline.alns import ALNSSolver
from ...schemas import VRPTWInstance
from ...data_loader import load_solomon_instance


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


# Best Known Solutions for Solomon RC instances (SINTEF/Li-Lim)
# Format: (vehicles, distance)
BKS = {
    'RC101': (14, 1696.94),
    'RC102': (12, 1554.75),
    'RC103': (11, 1261.67),
    'RC104': (10, 1135.48),
    'RC105': (13, 1629.44),
    'RC106': (11, 1424.73),
    'RC107': (11, 1230.48),
    'RC108': (10, 1139.82),
    'RC201': (4, 1406.91),
    'RC202': (3, 1367.09),
    'RC203': (3, 1049.62),
    'RC204': (3, 798.41),
    'RC205': (4, 1297.19),
    'RC206': (3, 1146.32),
    'RC207': (3, 1061.14),
    'RC208': (3, 828.14),
}


class Evaluator:
    """Evaluates DQN-ALNS against baselines on Solomon benchmarks."""
    
    def __init__(
        self,
        model_path: str,
        config: Optional[DQNConfig] = None,
        data_dir: str = "data/solomon"
    ):
        self.config = config or DQNConfig()
        self.device = torch.device(self.config.device)
        self.data_dir = data_dir
        
        # Load trained model
        self.dqn = DuelingDQN(self.config).to(self.device)
        self._load_model(model_path)
        self.dqn.eval()
    
    def _load_model(self, path: str):
        """Load model weights."""
        try:
            from safetensors.torch import load_file
            state_dict = load_file(path)
        except (ImportError, FileNotFoundError):
            state_dict = torch.load(
                path.replace('.safetensors', '.pt'),
                map_location=self.device
            )
        self.dqn.load_state_dict(state_dict)
    
    def evaluate_instance(
        self,
        instance: VRPTWInstance,
        time_limit: float = 30.0,
        n_runs: int = 10
    ) -> Dict[str, any]:
        """
        Evaluate DQN-ALNS on a single instance with multiple runs.
        
        Returns:
            Dict with vehicles, cost, time stats (mean, std, best)
        """
        dqn_results = {'vehicles': [], 'cost': [], 'time': []}
        alns_results = {'vehicles': [], 'cost': [], 'time': []}
        
        for run in range(n_runs):
            # Seed for reproducibility
            np.random.seed(run)
            torch.manual_seed(run)
            
            # DQN-ALNS
            start = time.time()
            dqn_solver = DQNALNSSolver(
                instance=instance,
                time_limit=time_limit,
                config=self.config,
                model=self.dqn,
                training=False
            )
            dqn_sol = dqn_solver.solve()
            dqn_time = time.time() - start
            
            dqn_results['vehicles'].append(dqn_sol.num_vehicles)
            dqn_results['cost'].append(dqn_sol.cost)
            dqn_results['time'].append(dqn_time)
            
            # ALNS baseline
            start = time.time()
            alns_solver = ALNSSolver(instance, time_limit=time_limit)
            alns_sol = alns_solver.solve()
            alns_time = time.time() - start
            
            alns_results['vehicles'].append(alns_sol.num_vehicles)
            alns_results['cost'].append(alns_sol.cost)
            alns_results['time'].append(alns_time)
        
        return {
            'dqn': {
                'vehicles_mean': np.mean(dqn_results['vehicles']),
                'vehicles_std': np.std(dqn_results['vehicles']),
                'vehicles_best': min(dqn_results['vehicles']),
                'cost_mean': np.mean(dqn_results['cost']),
                'cost_std': np.std(dqn_results['cost']),
                'cost_best': min(dqn_results['cost']),
                'time_mean': np.mean(dqn_results['time']),
            },
            'alns': {
                'vehicles_mean': np.mean(alns_results['vehicles']),
                'vehicles_std': np.std(alns_results['vehicles']),
                'vehicles_best': min(alns_results['vehicles']),
                'cost_mean': np.mean(alns_results['cost']),
                'cost_std': np.std(alns_results['cost']),
                'cost_best': min(alns_results['cost']),
                'time_mean': np.mean(alns_results['time']),
            }
        }
    
    def evaluate_benchmark(
        self,
        instances: List[str] = None,
        time_limit: float = 30.0,
        n_runs: int = 10,
        output_path: str = None
    ) -> pd.DataFrame:
        """
        Evaluate on Solomon RC benchmark instances.
        
        Args:
            instances: List of instance names (default: all RC)
            time_limit: Time limit per run
            n_runs: Number of runs per instance
            output_path: Optional CSV output path
        
        Returns:
            DataFrame with results
        """
        if instances is None:
            instances = list(BKS.keys())
        
        results = []
        
        for inst_name in instances:
            logger.info(f"Evaluating {inst_name}...")
            
            # Load instance
            try:
                instance = load_solomon_instance(inst_name, self.data_dir)
            except Exception as e:
                logger.warning(f"Could not load {inst_name}: {e}")
                continue
            
            # Evaluate
            res = self.evaluate_instance(instance, time_limit, n_runs)
            
            # Get BKS
            bks_veh, bks_cost = BKS.get(inst_name, (None, None))
            
            # Compute gaps
            dqn_veh_gap = None
            dqn_cost_gap = None
            alns_veh_gap = None
            alns_cost_gap = None
            
            if bks_veh is not None:
                dqn_veh_gap = (res['dqn']['vehicles_best'] - bks_veh) / bks_veh
                alns_veh_gap = (res['alns']['vehicles_best'] - bks_veh) / bks_veh
            
            if bks_cost is not None:
                dqn_cost_gap = (res['dqn']['cost_best'] - bks_cost) / bks_cost
                alns_cost_gap = (res['alns']['cost_best'] - bks_cost) / bks_cost
            
            results.append({
                'Instance': inst_name,
                'BKS_Veh': bks_veh,
                'BKS_Cost': bks_cost,
                'DQN_Veh': f"{res['dqn']['vehicles_mean']:.1f}±{res['dqn']['vehicles_std']:.1f}",
                'DQN_Cost': f"{res['dqn']['cost_mean']:.1f}±{res['dqn']['cost_std']:.1f}",
                'DQN_Gap': f"{dqn_cost_gap:.1%}" if dqn_cost_gap else "N/A",
                'DQN_Time': f"{res['dqn']['time_mean']:.1f}s",
                'ALNS_Veh': f"{res['alns']['vehicles_mean']:.1f}±{res['alns']['vehicles_std']:.1f}",
                'ALNS_Cost': f"{res['alns']['cost_mean']:.1f}±{res['alns']['cost_std']:.1f}",
                'ALNS_Gap': f"{alns_cost_gap:.1%}" if alns_cost_gap else "N/A",
                'ALNS_Time': f"{res['alns']['time_mean']:.1f}s",
            })
            
            logger.info(
                f"  DQN: {res['dqn']['vehicles_best']} veh, {res['dqn']['cost_best']:.1f} cost | "
                f"ALNS: {res['alns']['vehicles_best']} veh, {res['alns']['cost_best']:.1f} cost"
            )
        
        df = pd.DataFrame(results)
        
        if output_path:
            df.to_csv(output_path, index=False)
            logger.info(f"Results saved to {output_path}")
        
        return df
    
    def print_summary(self, df: pd.DataFrame):
        """Print summary statistics."""
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        print(df.to_string(index=False))
        print("="*80)


def main():
    """CLI entry point for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate DQN-ALNS on Solomon")
    
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--instances', type=str, nargs='+', default=None,
                        help='Instance names (default: all RC)')
    parser.add_argument('--runs', type=int, default=10,
                        help='Number of runs per instance')
    parser.add_argument('--time-limit', type=float, default=30.0,
                        help='Time limit per run')
    parser.add_argument('--output', type=str, default='results/dqn_alns_results.csv',
                        help='Output CSV path')
    parser.add_argument('--data-dir', type=str, default='data/solomon',
                        help='Solomon data directory')
    
    args = parser.parse_args()
    
    # Create results directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Run evaluation
    evaluator = Evaluator(
        model_path=args.model,
        data_dir=args.data_dir
    )
    
    df = evaluator.evaluate_benchmark(
        instances=args.instances,
        time_limit=args.time_limit,
        n_runs=args.runs,
        output_path=args.output
    )
    
    evaluator.print_summary(df)


if __name__ == "__main__":
    main()
