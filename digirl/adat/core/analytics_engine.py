"""
Analytics Engine - Centralized analytics for ADAT.

This module integrates FailureAnalyzer, DifficultyTracker, and TrajectoryWeighter
to provide a unified interface for adaptive training analytics.
"""

import os
import json
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import numpy as np

from .failure_analyzer import FailureAnalyzer
from .difficulty_tracker import DifficultyTracker
from .trajectory_weighter import TrajectoryWeighter


@dataclass
class AnalyticsSnapshot:
    """Snapshot of analytics at a point in time."""
    iteration: int
    total_trajectories: int
    success_rate: float
    avg_difficulty: float
    avg_learning_potential: float
    num_failure_clusters: int
    frontier_tasks: List[str]


class AnalyticsEngine:
    """
    Centralized analytics engine for ADAT.
    
    Integrates all analytics components and provides:
    - Unified trajectory processing
    - Summary statistics
    - Logging and visualization hooks
    - State persistence
    
    Attributes:
        failure_analyzer: FailureAnalyzer instance
        difficulty_tracker: DifficultyTracker instance
        trajectory_weighter: TrajectoryWeighter instance
    """
    
    def __init__(
        self,
        use_embeddings: bool = True,
        ema_alpha: float = 0.3,
        target_success_rate: float = 0.5,
        weight_alpha: float = 0.3,
        weight_beta: float = 0.5,
        weight_gamma: float = 0.2
    ):
        """
        Initialize the AnalyticsEngine.
        
        Args:
            use_embeddings: Use text embeddings for failure clustering
            ema_alpha: EMA smoothing for difficulty tracking
            target_success_rate: Target for Zone of Proximal Development
            weight_alpha: Task importance weight
            weight_beta: Near-miss weight
            weight_gamma: Progress weight
        """
        self.failure_analyzer = FailureAnalyzer(use_embeddings=use_embeddings)
        self.difficulty_tracker = DifficultyTracker(
            ema_alpha=ema_alpha,
            target_success_rate=target_success_rate
        )
        self.trajectory_weighter = TrajectoryWeighter(
            difficulty_tracker=self.difficulty_tracker,
            alpha=weight_alpha,
            beta=weight_beta,
            gamma=weight_gamma
        )
        
        self.iteration = 0
        self.snapshots: List[AnalyticsSnapshot] = []
        
        # Logging
        self._log_frequency = 10
        self._last_log = 0
    
    def process_trajectories(
        self,
        trajectories: List[List[dict]],
        update_clusters: bool = True
    ) -> Dict[str, Any]:
        """
        Process a batch of trajectories through all analytics components.
        
        Args:
            trajectories: Batch of trajectories from environment
            update_clusters: Whether to update failure clusters
            
        Returns:
            Dictionary with processing results and statistics
        """
        results = {
            "num_trajectories": len(trajectories),
            "successes": 0,
            "failures": 0,
            "avg_reward": 0.0
        }
        
        rewards = []
        
        for traj in trajectories:
            if not traj:
                continue
            
            task = traj[0].get("task", "unknown")
            reward = traj[-1].get("trajectory_reward", 0)
            success = reward > 0 and traj[-1].get("done", False)
            
            rewards.append(reward)
            
            # Update failure analyzer
            self.failure_analyzer.add_trajectory(traj, success, reward)
            
            # Update difficulty tracker
            self.difficulty_tracker.update(task, success)
            
            if success:
                results["successes"] += 1
            else:
                results["failures"] += 1
        
        results["avg_reward"] = np.mean(rewards) if rewards else 0.0
        results["success_rate"] = (
            results["successes"] / len(trajectories)
            if trajectories else 0.0
        )
        
        # Update clusters periodically
        if update_clusters and self.iteration % 10 == 0:
            self.failure_analyzer.cluster_failures()
        
        return results
    
    def get_sampling_weights(self, tasks: List[str]) -> np.ndarray:
        """Get curriculum-aware sampling weights for tasks."""
        return self.difficulty_tracker.get_sampling_weights(tasks)
    
    def get_trajectory_weights(
        self,
        trajectories: List[List[dict]],
        normalize: bool = True
    ) -> np.ndarray:
        """Get learning-potential weights for trajectories."""
        return self.trajectory_weighter.compute_weights(trajectories, normalize)
    
    def get_frontier_tasks(self, k: int = 10) -> List[str]:
        """Get tasks at the learning frontier."""
        return self.difficulty_tracker.get_frontier_tasks(k)
    
    def get_hardest_tasks(self, k: int = 10) -> List[str]:
        """Get hardest tasks by failure rate."""
        return [t[0] for t in self.failure_analyzer.get_hardest_tasks(k)]
    
    def step(self) -> None:
        """Increment iteration counter."""
        self.iteration += 1
        self.difficulty_tracker.step()
        
        # Take snapshot periodically
        if self.iteration % self._log_frequency == 0:
            self._take_snapshot()
    
    def _take_snapshot(self) -> None:
        """Take a snapshot of current analytics state."""
        fa_summary = self.failure_analyzer.get_summary()
        dt_summary = self.difficulty_tracker.get_summary()
        
        snapshot = AnalyticsSnapshot(
            iteration=self.iteration,
            total_trajectories=fa_summary.get("total_trajectories", 0),
            success_rate=fa_summary.get("success_rate", 0.0),
            avg_difficulty=dt_summary.get("avg_difficulty", 0.5),
            avg_learning_potential=dt_summary.get("avg_learning_potential", 0.5),
            num_failure_clusters=len(fa_summary.get("cluster_stats", {})),
            frontier_tasks=dt_summary.get("frontier_tasks", [])[:5]
        )
        
        self.snapshots.append(snapshot)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive analytics summary."""
        fa_summary = self.failure_analyzer.get_summary()
        dt_summary = self.difficulty_tracker.get_summary()
        
        return {
            "iteration": self.iteration,
            "failure_analysis": fa_summary,
            "difficulty_tracking": dt_summary,
            "num_snapshots": len(self.snapshots)
        }
    
    def get_wandb_metrics(self) -> Dict[str, float]:
        """Get metrics formatted for wandb logging."""
        summary = self.get_summary()
        fa = summary["failure_analysis"]
        dt = summary["difficulty_tracking"]
        
        metrics = {
            "adat/total_trajectories": fa.get("total_trajectories", 0),
            "adat/success_rate": fa.get("success_rate", 0.0),
            "adat/num_tasks": fa.get("num_tasks", 0),
            "adat/avg_difficulty": dt.get("avg_difficulty", 0.5),
            "adat/avg_learning_potential": dt.get("avg_learning_potential", 0.5),
        }
        
        return metrics
    
    def save(self, save_dir: str) -> None:
        """Save analytics state to directory."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save each component
        self.failure_analyzer.save(os.path.join(save_dir, "failure_analyzer.json"))
        self.difficulty_tracker.save(os.path.join(save_dir, "difficulty_tracker.json"))
        
        # Save engine state
        state = {
            "iteration": self.iteration,
            "snapshots": [
                {
                    "iteration": s.iteration,
                    "total_trajectories": s.total_trajectories,
                    "success_rate": s.success_rate,
                    "avg_difficulty": s.avg_difficulty,
                    "avg_learning_potential": s.avg_learning_potential,
                    "num_failure_clusters": s.num_failure_clusters,
                    "frontier_tasks": s.frontier_tasks
                }
                for s in self.snapshots
            ]
        }
        
        with open(os.path.join(save_dir, "analytics_engine.json"), 'w') as f:
            json.dump(state, f)
    
    def load(self, save_dir: str) -> None:
        """Load analytics state from directory."""
        fa_path = os.path.join(save_dir, "failure_analyzer.json")
        dt_path = os.path.join(save_dir, "difficulty_tracker.json")
        engine_path = os.path.join(save_dir, "analytics_engine.json")
        
        if os.path.exists(fa_path):
            self.failure_analyzer.load(fa_path)
        
        if os.path.exists(dt_path):
            self.difficulty_tracker.load(dt_path)
        
        if os.path.exists(engine_path):
            with open(engine_path, 'r') as f:
                state = json.load(f)
            
            self.iteration = state.get("iteration", 0)
            self.snapshots = [
                AnalyticsSnapshot(**s) for s in state.get("snapshots", [])
            ]
    
    def reset(self) -> None:
        """Reset all analytics state."""
        self.failure_analyzer = FailureAnalyzer(
            use_embeddings=self.failure_analyzer.use_embeddings
        )
        self.difficulty_tracker = DifficultyTracker(
            ema_alpha=self.difficulty_tracker.ema_alpha,
            target_success_rate=self.difficulty_tracker.target_success_rate
        )
        self.trajectory_weighter = TrajectoryWeighter(
            difficulty_tracker=self.difficulty_tracker,
            alpha=self.trajectory_weighter.alpha,
            beta=self.trajectory_weighter.beta,
            gamma=self.trajectory_weighter.gamma
        )
        self.iteration = 0
        self.snapshots = []
