"""
Hyperparameter Scheduler - Adaptive hyperparameter tuning.

This module adjusts training hyperparameters based on learning dynamics
observed through analytics.
"""

import numpy as np
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field


@dataclass
class HyperparameterState:
    """Current hyperparameter values."""
    temperature: float = 1.0
    actor_epochs: int = 20
    lm_lr: float = 1e-4
    critic_lr: float = 1e-4
    exploration_rate: float = 0.1


@dataclass
class SchedulerConfig:
    """Configuration for hyperparameter scheduling."""
    # Temperature adjustment
    temp_min: float = 0.5
    temp_max: float = 2.0
    temp_increase_threshold: float = 0.3  # Success rate below this → increase temp
    temp_decrease_threshold: float = 0.7  # Success rate above this → decrease temp
    temp_adjustment: float = 0.1
    
    # Actor epochs adjustment
    epochs_min: int = 10
    epochs_max: int = 50
    epochs_plateau_threshold: int = 5  # Iterations without improvement
    epochs_adjustment: int = 5
    
    # Learning rate adjustment
    lr_min: float = 1e-6
    lr_max: float = 1e-3
    lr_warmup_iterations: int = 50


class HyperparameterScheduler:
    """
    Adapts training hyperparameters based on learning dynamics.
    
    Adaptation rules:
    - Temperature: Increase when stuck, decrease when succeeding
    - Actor epochs: Increase on plateau, decrease when improving fast
    - Learning rate: Warmup then decay based on progress
    
    Attributes:
        config: SchedulerConfig with adjustment parameters
        state: Current hyperparameter values
    """
    
    def __init__(
        self,
        initial_state: Optional[HyperparameterState] = None,
        config: Optional[SchedulerConfig] = None
    ):
        """
        Initialize the scheduler.
        
        Args:
            initial_state: Initial hyperparameter values
            config: Scheduler configuration
        """
        self.state = initial_state or HyperparameterState()
        self.config = config or SchedulerConfig()
        
        # History for plateau detection
        self._success_history: List[float] = []
        self._history_window = 20
        
        # Plateau detection
        self._no_improvement_count = 0
        self._best_success_rate = 0.0
    
    def step(
        self,
        success_rate: float,
        avg_reward: float = 0.0,
        iteration: int = 0
    ) -> HyperparameterState:
        """
        Update hyperparameters based on current metrics.
        
        Args:
            success_rate: Current success rate
            avg_reward: Average reward
            iteration: Current training iteration
            
        Returns:
            Updated HyperparameterState
        """
        # Update history
        self._success_history.append(success_rate)
        if len(self._success_history) > self._history_window:
            self._success_history.pop(0)
        
        # Check for improvement
        if success_rate > self._best_success_rate + 0.01:
            self._best_success_rate = success_rate
            self._no_improvement_count = 0
        else:
            self._no_improvement_count += 1
        
        # Adjust temperature
        self._adjust_temperature(success_rate)
        
        # Adjust actor epochs
        self._adjust_epochs()
        
        # Adjust learning rate
        self._adjust_learning_rate(iteration)
        
        return self.state
    
    def _adjust_temperature(self, success_rate: float) -> None:
        """Adjust temperature based on success rate."""
        cfg = self.config
        
        if success_rate < cfg.temp_increase_threshold:
            # Struggling → increase exploration
            self.state.temperature = min(
                cfg.temp_max,
                self.state.temperature + cfg.temp_adjustment
            )
        elif success_rate > cfg.temp_decrease_threshold:
            # Succeeding → decrease exploration
            self.state.temperature = max(
                cfg.temp_min,
                self.state.temperature - cfg.temp_adjustment
            )
    
    def _adjust_epochs(self) -> None:
        """Adjust actor epochs based on plateau detection."""
        cfg = self.config
        
        if self._no_improvement_count >= cfg.epochs_plateau_threshold:
            # Plateau detected → increase training intensity
            self.state.actor_epochs = min(
                cfg.epochs_max,
                self.state.actor_epochs + cfg.epochs_adjustment
            )
            # Reset counter after adjustment
            if self.state.actor_epochs >= cfg.epochs_max:
                self._no_improvement_count = 0
        
        # If improving rapidly, could decrease epochs
        if len(self._success_history) >= 5:
            recent_improvement = (
                self._success_history[-1] - self._success_history[-5]
            )
            if recent_improvement > 0.1:
                self.state.actor_epochs = max(
                    cfg.epochs_min,
                    self.state.actor_epochs - cfg.epochs_adjustment // 2
                )
    
    def _adjust_learning_rate(self, iteration: int) -> None:
        """Adjust learning rate with warmup and decay."""
        cfg = self.config
        
        # Warmup
        if iteration < cfg.lr_warmup_iterations:
            warmup_factor = iteration / cfg.lr_warmup_iterations
            self.state.lm_lr = cfg.lr_min + (1e-4 - cfg.lr_min) * warmup_factor
            self.state.critic_lr = self.state.lm_lr
        
        # Decay on plateau
        elif self._no_improvement_count > cfg.epochs_plateau_threshold * 2:
            self.state.lm_lr = max(
                cfg.lr_min,
                self.state.lm_lr * 0.9
            )
            self.state.critic_lr = self.state.lm_lr
    
    def get_config_dict(self) -> Dict[str, Any]:
        """Get current hyperparameters as a dictionary."""
        return {
            "temperature": self.state.temperature,
            "actor_epochs": self.state.actor_epochs,
            "lm_lr": self.state.lm_lr,
            "critic_lr": self.state.critic_lr,
            "exploration_rate": self.state.exploration_rate
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get scheduler summary."""
        return {
            "current_state": self.get_config_dict(),
            "best_success_rate": self._best_success_rate,
            "no_improvement_count": self._no_improvement_count,
            "history_length": len(self._success_history)
        }
    
    def reset(self) -> None:
        """Reset scheduler state."""
        self.state = HyperparameterState()
        self._success_history = []
        self._no_improvement_count = 0
        self._best_success_rate = 0.0
