"""
Trajectory Weighter - Computes learning-potential weights for trajectories.

This module assigns weights to trajectories based on their estimated
learning value, prioritizing "near-misses" and underrepresented tasks.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from .failure_analyzer import FailureAnalyzer
from .difficulty_tracker import DifficultyTracker


@dataclass
class WeightedTrajectory:
    """A trajectory with its computed weight."""
    trajectory: List[dict]
    weight: float
    task: str
    reward: float
    near_miss_score: float
    task_importance: float


class TrajectoryWeighter:
    """
    Computes learning-potential weights for trajectories.
    
    Weight computation considers:
    1. Task importance: Inverse of success rate (rare successes matter more)
    2. Near-miss bonus: Almost-successes provide valuable learning signal
    3. Progress score: How far the agent got before failing
    
    Attributes:
        alpha: Weight for task importance component
        beta: Weight for near-miss component
        gamma: Weight for progress component
    """
    
    def __init__(
        self,
        difficulty_tracker: Optional[DifficultyTracker] = None,
        alpha: float = 0.3,
        beta: float = 0.5,
        gamma: float = 0.2,
        min_weight: float = 0.1,
        max_weight: float = 5.0
    ):
        """
        Initialize the TrajectoryWeighter.
        
        Args:
            difficulty_tracker: DifficultyTracker for task statistics
            alpha: Weight for task importance (inverse success rate)
            beta: Weight for near-miss bonus
            gamma: Weight for progress bonus
            min_weight: Minimum trajectory weight
            max_weight: Maximum trajectory weight
        """
        self.difficulty_tracker = difficulty_tracker
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.min_weight = min_weight
        self.max_weight = max_weight
    
    def compute_weights(
        self,
        trajectories: List[List[dict]],
        normalize: bool = True
    ) -> np.ndarray:
        """
        Compute weights for a batch of trajectories.
        
        Args:
            trajectories: List of trajectories (each is a list of step dicts)
            normalize: Whether to normalize weights to sum to 1
            
        Returns:
            Array of weights, one per trajectory
        """
        if not trajectories:
            return np.array([])
        
        weights = []
        for traj in trajectories:
            weight = self._compute_single_weight(traj)
            weights.append(weight)
        
        weights = np.array(weights)
        
        # Clip to range
        weights = np.clip(weights, self.min_weight, self.max_weight)
        
        if normalize and len(weights) > 0:
            weights = weights / weights.sum()
        
        return weights
    
    def _compute_single_weight(self, trajectory: List[dict]) -> float:
        """Compute weight for a single trajectory."""
        if not trajectory:
            return self.min_weight
        
        # Extract trajectory info
        task = trajectory[0].get("task", "unknown")
        reward = trajectory[-1].get("trajectory_reward", 0)
        success = reward > 0 and trajectory[-1].get("done", False)
        
        # 1. Task importance (inverse of success rate)
        task_importance = self._compute_task_importance(task)
        
        # 2. Near-miss bonus (high for almost-successes)
        near_miss_score = self._compute_near_miss_score(trajectory, success, reward)
        
        # 3. Progress score (how far did the agent get)
        progress_score = self._compute_progress_score(trajectory)
        
        # Combine components
        weight = (
            self.alpha * task_importance +
            self.beta * near_miss_score +
            self.gamma * progress_score
        )
        
        return weight
    
    def _compute_task_importance(self, task: str) -> float:
        """
        Compute task importance based on inverse success rate.
        
        Rare successes are more valuable, so tasks with low success rates
        get higher importance weights.
        """
        if self.difficulty_tracker is None:
            return 1.0
        
        success_rate = self.difficulty_tracker.get_success_rate(task)
        
        # Inverse success rate, with smoothing
        # importance = 1 / (success_rate + 0.1)
        # Normalized to roughly [0.5, 5]
        importance = 1.0 / (success_rate + 0.2)
        
        return importance
    
    def _compute_near_miss_score(
        self,
        trajectory: List[dict],
        success: bool,
        reward: float
    ) -> float:
        """
        Compute near-miss score.
        
        Near-misses (almost succeeded) provide valuable learning signal
        because they show what the agent did right and where it went wrong.
        """
        if success:
            # Successful trajectories: moderate weight
            # Quick successes are less informative
            steps = len(trajectory)
            max_steps = 10
            return 1.0 - (steps / max_steps) * 0.5  # [0.5, 1.0]
        
        # Failed trajectories: weight by how close to success
        # Use progress as a proxy
        steps = len(trajectory)
        max_steps = 10
        progress = steps / max_steps
        
        # Check if there were any partial rewards
        step_rewards = [step.get("reward", 0) for step in trajectory]
        partial_reward = sum(step_rewards)
        
        # Near-miss score: combination of progress and partial rewards
        near_miss = 0.5 * progress + 0.5 * min(partial_reward, 1.0)
        
        # High progress failures are very valuable
        if progress > 0.7:
            near_miss *= 1.5
        
        return near_miss
    
    def _compute_progress_score(self, trajectory: List[dict]) -> float:
        """
        Compute progress score based on trajectory length and actions.
        
        Longer trajectories that made meaningful progress are more valuable.
        """
        if not trajectory:
            return 0.0
        
        steps = len(trajectory)
        max_steps = 10
        
        # Basic progress
        progress = steps / max_steps
        
        # Bonus for diverse actions (agent tried different things)
        actions = [step.get("action", "") for step in trajectory]
        unique_actions = len(set(str(a) for a in actions))
        diversity_bonus = min(unique_actions / max(steps, 1), 1.0) * 0.3
        
        return progress + diversity_bonus
    
    def get_weighted_trajectories(
        self,
        trajectories: List[List[dict]]
    ) -> List[WeightedTrajectory]:
        """
        Get trajectories with detailed weight information.
        
        Args:
            trajectories: List of trajectories
            
        Returns:
            List of WeightedTrajectory objects with breakdown
        """
        weighted = []
        
        for traj in trajectories:
            if not traj:
                continue
            
            task = traj[0].get("task", "unknown")
            reward = traj[-1].get("trajectory_reward", 0)
            success = reward > 0 and traj[-1].get("done", False)
            
            task_importance = self._compute_task_importance(task)
            near_miss_score = self._compute_near_miss_score(traj, success, reward)
            
            weight = self._compute_single_weight(traj)
            
            weighted.append(WeightedTrajectory(
                trajectory=traj,
                weight=weight,
                task=task,
                reward=reward,
                near_miss_score=near_miss_score,
                task_importance=task_importance
            ))
        
        return weighted
    
    def filter_by_weight(
        self,
        trajectories: List[List[dict]],
        percentile: float = 0.9
    ) -> List[List[dict]]:
        """
        Filter trajectories, keeping top percentile by weight.
        
        Args:
            trajectories: List of trajectories
            percentile: Keep top this fraction (e.g., 0.9 = top 10%)
            
        Returns:
            Filtered list of trajectories
        """
        if not trajectories:
            return []
        
        weights = self.compute_weights(trajectories, normalize=False)
        
        # Find cutoff
        cutoff = np.percentile(weights, (1 - percentile) * 100)
        
        # Keep trajectories above cutoff
        filtered = [
            traj for traj, weight in zip(trajectories, weights)
            if weight >= cutoff
        ]
        
        return filtered
    
    def sample_weighted(
        self,
        trajectories: List[List[dict]],
        n_samples: int
    ) -> List[List[dict]]:
        """
        Sample trajectories according to their weights.
        
        Args:
            trajectories: List of trajectories
            n_samples: Number of samples to draw
            
        Returns:
            Sampled trajectories (with replacement if needed)
        """
        if not trajectories:
            return []
        
        weights = self.compute_weights(trajectories, normalize=True)
        
        # Sample indices
        indices = np.random.choice(
            len(trajectories),
            size=min(n_samples, len(trajectories)),
            replace=n_samples > len(trajectories),
            p=weights
        )
        
        return [trajectories[i] for i in indices]
