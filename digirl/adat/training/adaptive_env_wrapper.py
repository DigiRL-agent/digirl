"""
Adaptive Environment Wrapper - Wraps BatchedAndroidEnv with adaptive sampling.

This module provides a drop-in replacement wrapper that adds
adaptive task sampling to the Android environment.
"""

from typing import List, Optional, Dict, Any
import numpy as np

from ..samplers.adaptive_sampler import AdaptiveSampler
from ..core.analytics_engine import AnalyticsEngine


class AdaptiveEnvWrapper:
    """
    Wrapper that adds adaptive sampling to BatchedAndroidEnv.
    
    This wrapper intercepts task sampling to use ADAT's
    curriculum-aware sampling instead of random/sequential.
    
    Attributes:
        env: The wrapped BatchedAndroidEnv
        sampler: AdaptiveSampler for task selection
        analytics: AnalyticsEngine for tracking
    """
    
    def __init__(
        self,
        env: Any,  # BatchedAndroidEnv
        sampler: AdaptiveSampler,
        analytics: Optional[AnalyticsEngine] = None
    ):
        """
        Initialize the wrapper.
        
        Args:
            env: BatchedAndroidEnv instance to wrap
            sampler: AdaptiveSampler for task selection
            analytics: Optional AnalyticsEngine for tracking
        """
        self._env = env
        self.sampler = sampler
        self.analytics = analytics
        
        # Store original tasks
        self._original_tasks = list(env.all_tasks) if hasattr(env, 'all_tasks') else []
        
        # Track current tasks for each emulator
        self._current_tasks: List[str] = []
    
    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to wrapped environment."""
        return getattr(self._env, name)
    
    def reset(self) -> List[Dict]:
        """
        Reset the environment with adaptive task sampling.
        
        Returns:
            List of initial observations
        """
        # Sample tasks for this reset
        bsize = self._env.bsize
        self._current_tasks = self.sampler.sample(bsize)
        
        # Override the environment's task assignment
        # This works by modifying the task list before reset
        original_all_tasks = self._env.all_tasks
        original_sample_mode = self._env.sample_mode
        
        try:
            # Temporarily set tasks to our sampled ones
            self._env.all_tasks = self._current_tasks
            self._env.sample_mode = "sequential"
            
            # Reset with our tasks
            observations = self._env.reset()
            
        finally:
            # Restore original settings
            self._env.all_tasks = original_all_tasks
            self._env.sample_mode = original_sample_mode
        
        return observations
    
    def step(self, actions: List[str]) -> List:
        """
        Step the environment and track results.
        
        Args:
            actions: List of actions for each emulator
            
        Returns:
            List of (observation, reward, done) tuples
        """
        results = self._env.step(actions)
        
        # Track completions for the sampler
        for i, result in enumerate(results):
            if result is None:
                continue
            
            obs_dict, reward, done = result
            
            if done and i < len(self._current_tasks):
                task = self._current_tasks[i]
                success = reward > 0
                
                # Update sampler
                self.sampler.update(task, success)
        
        return results
    
    def get_current_tasks(self) -> List[str]:
        """Get the currently assigned tasks."""
        return self._current_tasks
    
    def set_strategy(self, strategy: str) -> None:
        """Change the sampling strategy."""
        self.sampler.set_strategy(strategy)
    
    @property
    def all_tasks(self) -> List[str]:
        """Get all available tasks."""
        return self._original_tasks
    
    @property
    def bsize(self) -> int:
        """Get batch size."""
        return self._env.bsize


class AdaptiveTaskAssigner:
    """
    Simpler alternative for task assignment without full wrapping.
    
    Use this if you want adaptive sampling without modifying
    the environment flow.
    """
    
    def __init__(
        self,
        all_tasks: List[str],
        analytics: AnalyticsEngine,
        strategy: str = "zpd"
    ):
        """
        Initialize the task assigner.
        
        Args:
            all_tasks: List of all available tasks
            analytics: AnalyticsEngine instance
            strategy: Sampling strategy
        """
        self.all_tasks = all_tasks
        self.analytics = analytics
        
        self.sampler = AdaptiveSampler(
            all_tasks=all_tasks,
            difficulty_tracker=analytics.difficulty_tracker,
            failure_analyzer=analytics.failure_analyzer,
            strategy=strategy
        )
    
    def get_tasks(self, n: int) -> List[str]:
        """Get n tasks to assign."""
        return self.sampler.sample(n)
    
    def update(self, task: str, success: bool) -> None:
        """Update after task completion."""
        self.sampler.update(task, success)
    
    def get_sampling_weights(self) -> np.ndarray:
        """Get current sampling weights for all tasks."""
        return self.sampler._compute_weights()
