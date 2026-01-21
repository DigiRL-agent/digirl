"""
Tests for ADAT Samplers.
"""

import pytest
import numpy as np

from digirl.adat.core.difficulty_tracker import DifficultyTracker
from digirl.adat.core.failure_analyzer import FailureAnalyzer
from digirl.adat.samplers.adaptive_sampler import AdaptiveSampler
from digirl.adat.samplers.curriculum_sampler import CurriculumSampler


class TestAdaptiveSampler:
    """Tests for AdaptiveSampler."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.tasks = ["Easy task", "Medium task", "Hard task", "New task"]
        self.tracker = DifficultyTracker()
        self.sampler = AdaptiveSampler(
            all_tasks=self.tasks,
            difficulty_tracker=self.tracker,
            strategy="zpd"
        )
    
    def test_sample_uniform(self):
        """Test uniform sampling."""
        self.sampler.set_strategy("uniform")
        
        samples = self.sampler.sample(n=100)
        
        assert len(samples) == 100
        # Check that all tasks appear
        unique = set(samples)
        assert len(unique) >= 1
    
    def test_sample_zpd(self):
        """Test Zone of Proximal Development sampling."""
        # Setup different success rates
        for _ in range(10):
            self.tracker.update("Easy task", success=True)
        for _ in range(5):
            self.tracker.update("Medium task", success=True)
        for _ in range(5):
            self.tracker.update("Medium task", success=False)
        for _ in range(10):
            self.tracker.update("Hard task", success=False)
        
        self.sampler.set_strategy("zpd")
        samples = self.sampler.sample(n=100)
        
        # Medium task should be sampled more (at learning frontier)
        counts = {t: samples.count(t) for t in self.tasks}
        
        # New task gets bonus, Medium gets ZPD bonus
        assert counts["Medium task"] > counts["Easy task"]
    
    def test_update_sampler(self):
        """Test sampler update."""
        self.sampler.update("Easy task", success=True)
        
        assert self.tracker._attempt_counts["Easy task"] == 1
        assert self.tracker._success_counts["Easy task"] == 1
    
    def test_exploration_rate(self):
        """Test exploration causes random sampling."""
        self.sampler.exploration_rate = 1.0  # Always explore
        
        samples = self.sampler.sample(n=100)
        
        # Should sample uniformly
        assert len(samples) == 100


class TestCurriculumSampler:
    """Tests for CurriculumSampler."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.tasks = [f"Task_{i}" for i in range(20)]
        self.tracker = DifficultyTracker()
        
        # Setup difficulties
        for i, task in enumerate(self.tasks):
            for _ in range(5):
                # Lower index = easier (more successes)
                success = np.random.random() > (i / 20)
                self.tracker.update(task, success)
        
        self.curriculum = CurriculumSampler(
            all_tasks=self.tasks,
            difficulty_tracker=self.tracker,
            n_stages=4,
            advancement_threshold=0.7,
            min_samples_per_stage=10
        )
    
    def test_initial_stage(self):
        """Test curriculum starts at stage 0."""
        assert self.curriculum.current_stage == 0
    
    def test_sample_from_stage(self):
        """Test sampling from current stage."""
        samples = self.curriculum.sample(n=10)
        
        assert len(samples) == 10
        # All samples should be from stage 0 tasks
        stage_tasks = set(self.curriculum.stages[0].tasks)
        for s in samples:
            assert s in stage_tasks
    
    def test_advancement(self):
        """Test curriculum advancement on success."""
        initial_stage = self.curriculum.current_stage
        
        # Simulate many successes
        for _ in range(15):
            task = self.curriculum.sample(1)[0]
            self.curriculum.update(task, success=True)
        
        # Should have advanced
        assert self.curriculum.current_stage > initial_stage
    
    def test_get_progress(self):
        """Test progress reporting."""
        progress = self.curriculum.get_progress()
        
        assert "current_stage" in progress
        assert "total_stages" in progress
        assert "stage_success_rate" in progress


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
