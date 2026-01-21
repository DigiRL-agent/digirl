# Analytics-Driven Adaptive Training (ADAT) Module
# This module provides failure-informed adaptive learning for DigiRL

# Core components (always available)
from .core.failure_analyzer import FailureAnalyzer
from .core.difficulty_tracker import DifficultyTracker
from .core.trajectory_weighter import TrajectoryWeighter
from .core.analytics_engine import AnalyticsEngine

# Samplers
from .samplers.adaptive_sampler import AdaptiveSampler
from .samplers.curriculum_sampler import CurriculumSampler

# Schedulers
from .schedulers.hyperparameter_scheduler import HyperparameterScheduler

# Training integration (requires full DigiRL environment)
try:
    from .training.adaptive_train_loop import adaptive_train_loop
    from .training.adaptive_env_wrapper import AdaptiveEnvWrapper
    _TRAINING_AVAILABLE = True
except ImportError:
    # Full DigiRL environment not available
    adaptive_train_loop = None
    AdaptiveEnvWrapper = None
    _TRAINING_AVAILABLE = False

__version__ = "0.1.0"
__all__ = [
    "FailureAnalyzer",
    "DifficultyTracker", 
    "TrajectoryWeighter",
    "AnalyticsEngine",
    "AdaptiveSampler",
    "CurriculumSampler",
    "HyperparameterScheduler",
    "adaptive_train_loop",
    "AdaptiveEnvWrapper",
]
