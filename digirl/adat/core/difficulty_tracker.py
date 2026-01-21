"""
Difficulty Tracker - Tracks and estimates task difficulty over time.

This module maintains a dynamic difficulty estimate for each task
based on historical success rates and learning progress.
"""

import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import json
import os


@dataclass
class TaskDifficulty:
    """Difficulty estimate for a single task."""
    task: str
    success_rate: float
    attempts: int
    difficulty_score: float
    learning_potential: float
    last_updated: int = 0


class DifficultyTracker:
    """
    Tracks task difficulty and learning potential over time.
    
    Key concepts:
    - difficulty_score: How hard a task is (1 - success_rate)
    - learning_potential: How much learning signal a task provides
      (highest at success_rate ≈ 0.5, based on Zone of Proximal Development)
    
    Attributes:
        tasks: Dictionary of task difficulties
        iteration: Current training iteration
        ema_alpha: Exponential moving average smoothing factor
    """
    
    def __init__(
        self,
        ema_alpha: float = 0.3,
        min_attempts: int = 3,
        target_success_rate: float = 0.5
    ):
        """
        Initialize the DifficultyTracker.
        
        Args:
            ema_alpha: Smoothing factor for exponential moving average
            min_attempts: Minimum attempts before using actual statistics
            target_success_rate: Target success rate for maximum learning potential
        """
        self.ema_alpha = ema_alpha
        self.min_attempts = min_attempts
        self.target_success_rate = target_success_rate
        
        self.tasks: Dict[str, TaskDifficulty] = {}
        self.iteration = 0
        
        # Running statistics
        self._success_counts: Dict[str, int] = defaultdict(int)
        self._attempt_counts: Dict[str, int] = defaultdict(int)
        self._recent_successes: Dict[str, List[bool]] = defaultdict(list)
        self._window_size = 50
    
    def update(self, task: str, success: bool) -> None:
        """
        Update difficulty estimate after a task attempt.
        
        Args:
            task: Task name
            success: Whether the attempt was successful
        """
        self._success_counts[task] += int(success)
        self._attempt_counts[task] += 1
        
        # Update recent window
        self._recent_successes[task].append(success)
        if len(self._recent_successes[task]) > self._window_size:
            self._recent_successes[task].pop(0)
        
        # Compute new estimates
        attempts = self._attempt_counts[task]
        
        if attempts < self.min_attempts:
            # Not enough data, use prior
            success_rate = 0.5
        else:
            # Use recent window for responsiveness
            recent = self._recent_successes[task]
            if len(recent) >= self.min_attempts:
                success_rate = sum(recent) / len(recent)
            else:
                success_rate = self._success_counts[task] / attempts
        
        # Compute difficulty and learning potential
        difficulty_score = 1.0 - success_rate
        learning_potential = self._compute_learning_potential(success_rate)
        
        # Update or create task difficulty
        if task in self.tasks:
            # Exponential moving average for smoothing
            old = self.tasks[task]
            new_sr = self.ema_alpha * success_rate + (1 - self.ema_alpha) * old.success_rate
            new_diff = self.ema_alpha * difficulty_score + (1 - self.ema_alpha) * old.difficulty_score
            new_lp = self.ema_alpha * learning_potential + (1 - self.ema_alpha) * old.learning_potential
            
            self.tasks[task] = TaskDifficulty(
                task=task,
                success_rate=new_sr,
                attempts=attempts,
                difficulty_score=new_diff,
                learning_potential=new_lp,
                last_updated=self.iteration
            )
        else:
            self.tasks[task] = TaskDifficulty(
                task=task,
                success_rate=success_rate,
                attempts=attempts,
                difficulty_score=difficulty_score,
                learning_potential=learning_potential,
                last_updated=self.iteration
            )
    
    def _compute_learning_potential(self, success_rate: float) -> float:
        """
        Compute learning potential based on Zone of Proximal Development.
        
        Maximum learning occurs when success rate ≈ target (typically 0.5),
        using binary entropy as the measure.
        
        Args:
            success_rate: Current success rate for the task
            
        Returns:
            Learning potential score between 0 and 1
        """
        # Clip to avoid log(0)
        p = np.clip(success_rate, 0.01, 0.99)
        
        # Binary entropy: H(p) = -p*log(p) - (1-p)*log(1-p)
        entropy = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
        
        # Adjust for target success rate if not 0.5
        # Peak should be at target_success_rate
        if self.target_success_rate != 0.5:
            # Shift the peak
            distance_from_target = abs(success_rate - self.target_success_rate)
            penalty = distance_from_target ** 2
            entropy = entropy * (1 - penalty)
        
        return entropy
    
    def get_difficulty(self, task: str) -> float:
        """
        Get difficulty score for a task.
        
        Args:
            task: Task name
            
        Returns:
            Difficulty score between 0 (easy) and 1 (hard)
        """
        if task in self.tasks:
            return self.tasks[task].difficulty_score
        return 0.5  # Unknown task, assume medium difficulty
    
    def get_learning_potential(self, task: str) -> float:
        """
        Get learning potential for a task.
        
        Args:
            task: Task name
            
        Returns:
            Learning potential between 0 (low) and 1 (high)
        """
        if task in self.tasks:
            return self.tasks[task].learning_potential
        return 1.0  # Unknown task, high potential
    
    def get_success_rate(self, task: str) -> float:
        """Get current success rate estimate for a task."""
        if task in self.tasks:
            return self.tasks[task].success_rate
        return 0.5
    
    def get_sampling_weights(self, tasks: List[str]) -> np.ndarray:
        """
        Get sampling weights for a list of tasks.
        
        Higher weights for tasks with higher learning potential.
        
        Args:
            tasks: List of task names
            
        Returns:
            Normalized sampling weights
        """
        weights = np.array([self.get_learning_potential(t) for t in tasks])
        
        # Add small epsilon to avoid zero weights
        weights = weights + 0.01
        
        # Normalize
        return weights / weights.sum()
    
    def get_curriculum_order(self) -> List[str]:
        """
        Get tasks ordered by curriculum (easy to hard).
        
        Returns:
            List of task names sorted by difficulty
        """
        task_list = list(self.tasks.values())
        task_list.sort(key=lambda x: x.difficulty_score)
        return [t.task for t in task_list]
    
    def get_frontier_tasks(self, k: int = 10) -> List[str]:
        """
        Get tasks at the learning frontier (highest learning potential).
        
        Args:
            k: Number of tasks to return
            
        Returns:
            List of task names with highest learning potential
        """
        task_list = list(self.tasks.values())
        task_list.sort(key=lambda x: x.learning_potential, reverse=True)
        return [t.task for t in task_list[:k]]
    
    def step(self) -> None:
        """Increment the iteration counter."""
        self.iteration += 1
    
    def get_summary(self) -> Dict:
        """Get a summary of difficulty tracking."""
        if not self.tasks:
            return {"num_tasks": 0}
        
        difficulties = [t.difficulty_score for t in self.tasks.values()]
        potentials = [t.learning_potential for t in self.tasks.values()]
        
        return {
            "num_tasks": len(self.tasks),
            "avg_difficulty": np.mean(difficulties),
            "avg_learning_potential": np.mean(potentials),
            "frontier_tasks": self.get_frontier_tasks(5),
            "easiest_tasks": self.get_curriculum_order()[:5],
            "hardest_tasks": self.get_curriculum_order()[-5:][::-1]
        }
    
    def save(self, path: str) -> None:
        """Save tracker state to file."""
        state = {
            "iteration": self.iteration,
            "tasks": {
                name: {
                    "success_rate": t.success_rate,
                    "attempts": t.attempts,
                    "difficulty_score": t.difficulty_score,
                    "learning_potential": t.learning_potential
                }
                for name, t in self.tasks.items()
            },
            "success_counts": dict(self._success_counts),
            "attempt_counts": dict(self._attempt_counts)
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(state, f)
    
    def load(self, path: str) -> None:
        """Load tracker state from file."""
        with open(path, 'r') as f:
            state = json.load(f)
        
        self.iteration = state.get("iteration", 0)
        self._success_counts = defaultdict(int, state.get("success_counts", {}))
        self._attempt_counts = defaultdict(int, state.get("attempt_counts", {}))
        
        for name, data in state.get("tasks", {}).items():
            self.tasks[name] = TaskDifficulty(
                task=name,
                success_rate=data["success_rate"],
                attempts=data["attempts"],
                difficulty_score=data["difficulty_score"],
                learning_potential=data["learning_potential"]
            )
