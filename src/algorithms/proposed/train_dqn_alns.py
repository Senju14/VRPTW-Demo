"""
Training Script for DQN-ALNS on VRPTW.

Implements the full training loop with:
- Synthetic instance generation
- Parallel ALNS trajectory collection
- DQN training with Prioritized Experience Replay
- Logging and checkpointing
"""

import os
import time
import argparse
import logging
from datetime import datetime
from typing import Optional, Dict, List

import numpy as np
import torch

from .dqn_alns import DQNConfig, DQNALNSSolver, DuelingDQN, PrioritizedReplayBuffer
from .instance_generator import generate_training_batch, generate_scaling_test_instances
from ...schemas import VRPTWInstance


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DQNALNSTrainer:
    """
    Trainer for DQN-ALNS with synthetic Solomon instances.
    
    Features:
    - Batch training on diverse instances
    - Curriculum learning (easy → hard)
    - Periodic evaluation and checkpointing
    - TensorBoard logging
    """
    
    def __init__(
        self,
        config: Optional[DQNConfig] = None,
        output_dir: str = "models",
        log_dir: str = "runs"
    ):
        self.config = config or DQNConfig()
        self.device = torch.device(self.config.device)
        self.output_dir = output_dir
        self.log_dir = log_dir
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Shared DQN model
        self.dqn = DuelingDQN(self.config).to(self.device)
        self.target_dqn = DuelingDQN(self.config).to(self.device)
        self.target_dqn.load_state_dict(self.dqn.state_dict())
        
        # Shared replay buffer
        self.replay_buffer = PrioritizedReplayBuffer(self.config)
        
        # Optimizer with gradient clipping
        self.optimizer = torch.optim.Adam(
            self.dqn.parameters(), 
            lr=self.config.lr
        )
        
        # Training state
        self.epsilon = self.config.epsilon_start
        self.total_steps = 0
        self.best_avg_gap = float('inf')
        
        # TensorBoard writer (optional)
        self.writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=os.path.join(log_dir, 
                                        datetime.now().strftime('%Y%m%d_%H%M%S')))
        except ImportError:
            logger.warning("TensorBoard not available, skipping logging")
    
    def train(
        self,
        n_episodes: int = 1000,
        instances_per_episode: int = 8,
        n_customers: int = 100,
        alns_time_limit: float = 15.0,
        eval_freq: int = 50,
        checkpoint_freq: int = 100
    ) -> Dict[str, List[float]]:
        """
        Main training loop.
        
        Args:
            n_episodes: Total training episodes
            instances_per_episode: Instances solved per episode
            n_customers: Number of customers per instance
            alns_time_limit: Time limit for each ALNS run
            eval_freq: Evaluation frequency (episodes)
            checkpoint_freq: Checkpoint frequency (episodes)
        
        Returns:
            Training history dictionary
        """
        history = {
            'episode': [],
            'avg_vehicles': [],
            'avg_cost': [],
            'epsilon': [],
            'avg_reward': []
        }
        
        logger.info(f"Starting training on {self.device}")
        logger.info(f"Config: {self.config.__dict__}")
        start_time = time.time()
        
        for episode in range(1, n_episodes + 1):
            # Curriculum: increase difficulty over time
            tw_tightness = min(0.3 + 0.4 * (episode / n_episodes), 0.7)
            
            # Generate training batch
            instances = generate_training_batch(
                batch_size=instances_per_episode,
                n_customers=n_customers,
                tw_tightness_range=(tw_tightness - 0.1, tw_tightness + 0.1),
                seed_base=episode * instances_per_episode
            )
            
            # Run ALNS on each instance
            episode_stats = self._run_episode(instances, alns_time_limit)
            
            # Update epsilon
            self.epsilon = max(
                self.config.epsilon_end,
                self.epsilon * self.config.epsilon_decay
            )
            
            # Update beta for PER
            progress = episode / n_episodes
            self.replay_buffer.update_beta(progress)
            
            # Log stats
            history['episode'].append(episode)
            history['avg_vehicles'].append(episode_stats['avg_vehicles'])
            history['avg_cost'].append(episode_stats['avg_cost'])
            history['epsilon'].append(self.epsilon)
            history['avg_reward'].append(episode_stats['avg_reward'])
            
            if self.writer:
                self.writer.add_scalar('Train/Vehicles', episode_stats['avg_vehicles'], episode)
                self.writer.add_scalar('Train/Cost', episode_stats['avg_cost'], episode)
                self.writer.add_scalar('Train/Epsilon', self.epsilon, episode)
                self.writer.add_scalar('Train/Reward', episode_stats['avg_reward'], episode)
                self.writer.add_scalar('Train/BufferSize', len(self.replay_buffer), episode)
            
            if episode % 10 == 0:
                elapsed = time.time() - start_time
                logger.info(
                    f"Episode {episode}/{n_episodes} | "
                    f"Vehicles: {episode_stats['avg_vehicles']:.1f} | "
                    f"Cost: {episode_stats['avg_cost']:.1f} | "
                    f"Epsilon: {self.epsilon:.3f} | "
                    f"Time: {elapsed/60:.1f}min"
                )
            
            # Evaluation
            if episode % eval_freq == 0:
                eval_stats = self._evaluate()
                logger.info(f"  Eval | Gap to heuristic: {eval_stats['avg_gap']:.2%}")
                
                if self.writer:
                    self.writer.add_scalar('Eval/Gap', eval_stats['avg_gap'], episode)
                
                if eval_stats['avg_gap'] < self.best_avg_gap:
                    self.best_avg_gap = eval_stats['avg_gap']
                    self._save_checkpoint(f"best_model.safetensors")
                    logger.info(f"  New best model saved!")
            
            # Checkpoint
            if episode % checkpoint_freq == 0:
                self._save_checkpoint(f"checkpoint_ep{episode}.safetensors")
        
        # Final save
        self._save_checkpoint("final_model.safetensors")
        
        if self.writer:
            self.writer.close()
        
        total_time = time.time() - start_time
        logger.info(f"Training complete in {total_time/3600:.2f} hours")
        
        return history
    
    def _run_episode(
        self, 
        instances: List[VRPTWInstance],
        time_limit: float
    ) -> Dict[str, float]:
        """Run one training episode on a batch of instances."""
        all_vehicles = []
        all_costs = []
        all_rewards = []
        
        for instance in instances:
            solver = DQNALNSSolver(
                instance=instance,
                time_limit=time_limit,
                config=self.config,
                model=self.dqn,
                training=True
            )
            
            # Share training components
            solver.epsilon = self.epsilon
            solver.replay_buffer = self.replay_buffer
            solver.target_dqn = self.target_dqn
            solver.optimizer = self.optimizer
            solver.training_step = self.total_steps
            
            # Run ALNS
            best_sol = solver.solve()
            
            # Collect stats
            all_vehicles.append(best_sol.num_vehicles)
            all_costs.append(best_sol.cost)
            
            # Update total steps
            self.total_steps = solver.training_step
        
        # Sync target network periodically
        if self.total_steps % (self.config.target_update_freq * 10) == 0:
            self.target_dqn.load_state_dict(self.dqn.state_dict())
        
        return {
            'avg_vehicles': np.mean(all_vehicles),
            'avg_cost': np.mean(all_costs),
            'avg_reward': 0.0  # Could track from solver
        }
    
    def _evaluate(self, n_instances: int = 10) -> Dict[str, float]:
        """Evaluate current model vs greedy baseline."""
        instances = generate_training_batch(
            batch_size=n_instances,
            n_customers=50,  # Smaller for faster eval
            seed_base=99999
        )
        
        gaps = []
        
        for instance in instances:
            # DQN-ALNS (inference mode)
            dqn_solver = DQNALNSSolver(
                instance=instance,
                time_limit=5.0,
                config=self.config,
                model=self.dqn,
                training=False
            )
            dqn_sol = dqn_solver.solve()
            
            # Pure ALNS baseline
            from ..baseline.alns import ALNSSolver
            alns_solver = ALNSSolver(instance, time_limit=5.0)
            alns_sol = alns_solver.solve()
            
            # Gap = (dqn - alns) / alns
            if alns_sol.cost > 0:
                gap = (dqn_sol.cost - alns_sol.cost) / alns_sol.cost
                gaps.append(gap)
        
        return {'avg_gap': np.mean(gaps)}
    
    def _save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = os.path.join(self.output_dir, filename)
        try:
            from safetensors.torch import save_file
            save_file(self.dqn.state_dict(), path)
        except ImportError:
            torch.save(self.dqn.state_dict(), path.replace('.safetensors', '.pt'))


def main():
    """CLI entry point for training."""
    parser = argparse.ArgumentParser(description="Train DQN-ALNS for VRPTW")
    
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Number of training episodes')
    parser.add_argument('--customers', type=int, default=100,
                        help='Number of customers per instance')
    parser.add_argument('--batch', type=int, default=8,
                        help='Instances per episode')
    parser.add_argument('--time-limit', type=float, default=15.0,
                        help='ALNS time limit per instance')
    parser.add_argument('--output', type=str, default='models',
                        help='Output directory for models')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    config = DQNConfig()
    if args.device:
        config.device = args.device
    
    trainer = DQNALNSTrainer(config=config, output_dir=args.output)
    trainer.train(
        n_episodes=args.episodes,
        instances_per_episode=args.batch,
        n_customers=args.customers,
        alns_time_limit=args.time_limit
    )


if __name__ == "__main__":
    main()
