"""
Curriculum Sampler - Progressive difficulty curriculum.

This module implements curriculum learning by gradually increasing
task difficulty as the agent improves.
"""

import numpy as np
from typing import List, Optional, Dict
from dataclasses import dataclass

from ..core.difficulty_tracker import DifficultyTracker


@dataclass
class CurriculumStage:
    """A stage in the curriculum."""
    name: str
    min_difficulty: float
    max_difficulty: float
    tasks: List[str]


class CurriculumSampler:
    """
    Progressive curriculum that advances through difficulty stages.
    
    The curriculum automatically advances when the agent achieves
    sufficient mastery at the current stage (based on success rate).
    
    Attributes:
        stages: List of curriculum stages
        current_stage: Index of current stage
        advancement_threshold: Success rate needed to advance
    """
    
    def __init__(
        self,
        all_tasks: List[str],
        difficulty_tracker: DifficultyTracker,
        n_stages: int = 5,
        advancement_threshold: float = 0.7,
        min_samples_per_stage: int = 50
    ):
        """
        Initialize the CurriculumSampler.
        
        Args:
            all_tasks: List of all available tasks
            difficulty_tracker: DifficultyTracker for task statistics
            n_stages: Number of curriculum stages
            advancement_threshold: Success rate needed to advance
            min_samples_per_stage: Minimum samples before advancement
        """
        self.all_tasks = all_tasks
        self.difficulty_tracker = difficulty_tracker
        self.n_stages = n_stages
        self.advancement_threshold = advancement_threshold
        self.min_samples_per_stage = min_samples_per_stage
        
        # Current state
        self.current_stage = 0
        self.samples_in_stage = 0
        self.successes_in_stage = 0
        
        # Build curriculum stages
        self.stages = self._build_stages()
    
    def _build_stages(self) -> List[CurriculumStage]:
        """Build curriculum stages based on initial difficulty estimates."""
        stages = []
        
        # Get difficulty for each task
        task_difficulties = []
        for task in self.all_tasks:
            diff = self.difficulty_tracker.get_difficulty(task)
            task_difficulties.append((task, diff))
        
        # Sort by difficulty
        task_difficulties.sort(key=lambda x: x[1])
        
        # Divide into stages
        tasks_per_stage = len(self.all_tasks) // self.n_stages
        
        for i in range(self.n_stages):
            start_idx = i * tasks_per_stage
            end_idx = start_idx + tasks_per_stage if i < self.n_stages - 1 else len(task_difficulties)
            
            stage_tasks = [t[0] for t in task_difficulties[start_idx:end_idx]]
            
            if stage_tasks:
                min_diff = task_difficulties[start_idx][1]
                max_diff = task_difficulties[min(end_idx - 1, len(task_difficulties) - 1)][1]
            else:
                min_diff, max_diff = i / self.n_stages, (i + 1) / self.n_stages
            
            stages.append(CurriculumStage(
                name=f"stage_{i}",
                min_difficulty=min_diff,
                max_difficulty=max_diff,
                tasks=stage_tasks
            ))
        
        return stages
    
    def sample(self, n: int = 1) -> List[str]:
        """
        Sample tasks from the current curriculum stage.
        
        Args:
            n: Number of tasks to sample
            
        Returns:
            List of sampled task names
        """
        if not self.stages or self.current_stage >= len(self.stages):
            # Fallback to random
            return list(np.random.choice(self.all_tasks, size=n, replace=True))
        
        current = self.stages[self.current_stage]
        
        if not current.tasks:
            # Empty stage, try next
            self._try_advance()
            return self.sample(n)
        
        # Sample from current stage
        samples = list(np.random.choice(current.tasks, size=n, replace=True))
        return samples
    
    def update(self, task: str, success: bool) -> None:
        """
        Update curriculum state after a task attempt.
        
        Args:
            task: Task that was attempted
            success: Whether the attempt succeeded
        """
        self.samples_in_stage += 1
        if success:
            self.successes_in_stage += 1
        
        # Update difficulty tracker
        self.difficulty_tracker.update(task, success)
        
        # Check for advancement
        self._try_advance()
    
    def _try_advance(self) -> None:
        """Try to advance to the next stage if criteria are met."""
        if self.current_stage >= len(self.stages) - 1:
            return  # Already at final stage
        
        if self.samples_in_stage < self.min_samples_per_stage:
            return  # Not enough samples
        
        # Check success rate
        success_rate = self.successes_in_stage / self.samples_in_stage
        
        if success_rate >= self.advancement_threshold:
            self._advance()
    
    def _advance(self) -> None:
        """Advance to the next curriculum stage."""
        self.current_stage += 1
        self.samples_in_stage = 0
        self.successes_in_stage = 0
        
        if self.current_stage < len(self.stages):
            print(f"[Curriculum] Advanced to stage {self.current_stage}: "
                  f"{self.stages[self.current_stage].name}")
    
    def rebuild_stages(self) -> None:
        """
        Rebuild curriculum stages based on updated difficulty estimates.
        
        Call this periodically to adapt the curriculum to new learning.
        """
        self.stages = self._build_stages()
        
        # Ensure current stage is valid
        self.current_stage = min(self.current_stage, len(self.stages) - 1)
    
    def get_current_stage(self) -> Optional[CurriculumStage]:
        """Get the current curriculum stage."""
        if 0 <= self.current_stage < len(self.stages):
            return self.stages[self.current_stage]
        return None
    
    def get_progress(self) -> Dict:
        """Get curriculum progress information."""
        current = self.get_current_stage()
        success_rate = (
            self.successes_in_stage / self.samples_in_stage
            if self.samples_in_stage > 0 else 0.0
        )
        
        return {
            "current_stage": self.current_stage,
            "total_stages": len(self.stages),
            "stage_name": current.name if current else "N/A",
            "samples_in_stage": self.samples_in_stage,
            "stage_success_rate": success_rate,
            "advancement_threshold": self.advancement_threshold,
            "progress_to_next": success_rate / self.advancement_threshold
        }
    
    def reset(self) -> None:
        """Reset curriculum to the beginning."""
        self.current_stage = 0
        self.samples_in_stage = 0
        self.successes_in_stage = 0
