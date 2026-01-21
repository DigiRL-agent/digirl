"""
Adaptive Sampler - Curriculum-aware task sampling.

This module provides adaptive task sampling based on difficulty tracking
and failure analysis to focus learning on the most informative tasks.
"""

import numpy as np
from typing import List, Optional, Dict, Any
from collections import defaultdict

from ..core.difficulty_tracker import DifficultyTracker
from ..core.failure_analyzer import FailureAnalyzer


class AdaptiveSampler:
    """
    Samples tasks based on learning potential and difficulty.
    
    Implements curriculum learning by focusing on tasks at the
    agent's "learning frontier" - tasks that are neither too easy
    nor too hard (Zone of Proximal Development).
    
    Sampling strategies:
    - "zpd": Focus on tasks with success rate ≈ target (default 0.5)
    - "hard_first": Prioritize difficult tasks
    - "easy_first": Start with easy tasks, progress to hard
    - "uniform": Random sampling (baseline)
    - "cluster_aware": Oversample from underperforming failure clusters
    """
    
    def __init__(
        self,
        all_tasks: List[str],
        difficulty_tracker: Optional[DifficultyTracker] = None,
        failure_analyzer: Optional[FailureAnalyzer] = None,
        strategy: str = "zpd",
        exploration_rate: float = 0.1,
        unknown_task_bonus: float = 2.0
    ):
        """
        Initialize the AdaptiveSampler.
        
        Args:
            all_tasks: List of all available tasks
            difficulty_tracker: DifficultyTracker for task statistics
            failure_analyzer: FailureAnalyzer for cluster information
            strategy: Sampling strategy ("zpd", "hard_first", etc.)
            exploration_rate: Probability of random exploration
            unknown_task_bonus: Bonus weight for unseen tasks
        """
        self.all_tasks = all_tasks
        self.difficulty_tracker = difficulty_tracker or DifficultyTracker()
        self.failure_analyzer = failure_analyzer
        self.strategy = strategy
        self.exploration_rate = exploration_rate
        self.unknown_task_bonus = unknown_task_bonus
        
        # Task indices for fast lookup
        self._task_to_idx = {t: i for i, t in enumerate(all_tasks)}
        
        # Sampling history
        self._sample_counts = defaultdict(int)
    
    def sample(self, n: int = 1) -> List[str]:
        """
        Sample n tasks according to the current strategy.
        
        Args:
            n: Number of tasks to sample
            
        Returns:
            List of sampled task names
        """
        # Exploration vs exploitation
        samples = []
        for _ in range(n):
            if np.random.random() < self.exploration_rate:
                # Random exploration
                task = np.random.choice(self.all_tasks)
            else:
                # Strategy-based sampling
                task = self._sample_one()
            samples.append(task)
            self._sample_counts[task] += 1
        
        return samples
    
    def _sample_one(self) -> str:
        """Sample a single task based on strategy."""
        weights = self._compute_weights()
        
        # Normalize
        weights = weights / weights.sum()
        
        # Sample
        idx = np.random.choice(len(self.all_tasks), p=weights)
        return self.all_tasks[idx]
    
    def _compute_weights(self) -> np.ndarray:
        """Compute sampling weights based on strategy."""
        if self.strategy == "uniform":
            return np.ones(len(self.all_tasks))
        
        elif self.strategy == "zpd":
            return self._zpd_weights()
        
        elif self.strategy == "hard_first":
            return self._difficulty_weights(prioritize_hard=True)
        
        elif self.strategy == "easy_first":
            return self._difficulty_weights(prioritize_hard=False)
        
        elif self.strategy == "cluster_aware":
            return self._cluster_aware_weights()
        
        else:
            return np.ones(len(self.all_tasks))
    
    def _zpd_weights(self) -> np.ndarray:
        """
        Zone of Proximal Development weights.
        
        Highest weight for tasks with ~50% success rate.
        """
        weights = np.zeros(len(self.all_tasks))
        
        for i, task in enumerate(self.all_tasks):
            lp = self.difficulty_tracker.get_learning_potential(task)
            attempts = self.difficulty_tracker._attempt_counts.get(task, 0)
            
            if attempts == 0:
                # Bonus for unseen tasks
                weights[i] = self.unknown_task_bonus
            else:
                weights[i] = lp
        
        return weights
    
    def _difficulty_weights(self, prioritize_hard: bool = True) -> np.ndarray:
        """Weights based on task difficulty."""
        weights = np.zeros(len(self.all_tasks))
        
        for i, task in enumerate(self.all_tasks):
            difficulty = self.difficulty_tracker.get_difficulty(task)
            attempts = self.difficulty_tracker._attempt_counts.get(task, 0)
            
            if attempts == 0:
                weights[i] = self.unknown_task_bonus
            elif prioritize_hard:
                weights[i] = difficulty + 0.1
            else:
                weights[i] = 1.0 - difficulty + 0.1
        
        return weights
    
    def _cluster_aware_weights(self) -> np.ndarray:
        """
        Oversample tasks from underperforming failure clusters.
        """
        weights = self._zpd_weights()  # Start with ZPD
        
        if self.failure_analyzer is None:
            return weights
        
        # Get cluster statistics
        cluster_stats = self.failure_analyzer.get_cluster_statistics()
        
        # Find largest failure clusters
        for cluster_id, stats in cluster_stats.items():
            top_tasks = stats.get("top_tasks", [])
            cluster_size = stats.get("size", 0)
            
            # Boost weight for tasks in large failure clusters
            for task_name, count in top_tasks:
                if task_name in self._task_to_idx:
                    idx = self._task_to_idx[task_name]
                    boost = 1.0 + (cluster_size / 100) * 0.5
                    weights[idx] *= boost
        
        return weights
    
    def update(self, task: str, success: bool) -> None:
        """
        Update statistics after a task attempt.
        
        Args:
            task: Task that was attempted
            success: Whether the attempt succeeded
        """
        self.difficulty_tracker.update(task, success)
    
    def set_strategy(self, strategy: str) -> None:
        """Change the sampling strategy."""
        valid = ["zpd", "hard_first", "easy_first", "uniform", "cluster_aware"]
        if strategy not in valid:
            raise ValueError(f"Strategy must be one of {valid}")
        self.strategy = strategy
    
    def get_task_statistics(self) -> Dict[str, Any]:
        """Get statistics about task sampling."""
        return {
            "strategy": self.strategy,
            "total_samples": sum(self._sample_counts.values()),
            "unique_tasks_sampled": len(self._sample_counts),
            "most_sampled": sorted(
                self._sample_counts.items(),
                key=lambda x: -x[1]
            )[:10]
        }
