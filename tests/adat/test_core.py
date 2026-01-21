"""
Tests for ADAT Core Components.

Run with: pytest tests/adat/ -v
"""

import pytest
import numpy as np
import os
import tempfile
import json

from digirl.adat.core.failure_analyzer import FailureAnalyzer
from digirl.adat.core.difficulty_tracker import DifficultyTracker
from digirl.adat.core.trajectory_weighter import TrajectoryWeighter
from digirl.adat.core.analytics_engine import AnalyticsEngine


class TestFailureAnalyzer:
    """Tests for FailureAnalyzer."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.analyzer = FailureAnalyzer(use_embeddings=True)
    
    def test_add_trajectory(self):
        """Test adding trajectories."""
        trajectory = [
            {"task": "Open settings", "action": "tap", "observation": "home screen"},
            {"task": "Open settings", "action": "tap", "observation": "settings menu"}
        ]
        
        self.analyzer.add_trajectory(trajectory, success=True, reward=1.0)
        
        assert len(self.analyzer.trajectories) == 1
        assert self.analyzer.task_stats["Open settings"]["total"] == 1
        assert self.analyzer.task_stats["Open settings"]["success"] == 1
    
    def test_add_failed_trajectory(self):
        """Test adding failed trajectories."""
        trajectory = [
            {"task": "Set alarm", "action": "tap", "observation": "clock app"},
        ]
        
        self.analyzer.add_trajectory(trajectory, success=False, reward=0.0)
        
        assert self.analyzer.task_stats["Set alarm"]["total"] == 1
        assert self.analyzer.task_stats["Set alarm"]["success"] == 0
    
    def test_get_task_success_rate(self):
        """Test success rate calculation."""
        trajectories = [
            ([{"task": "Test task", "action": "a1"}], True),
            ([{"task": "Test task", "action": "a2"}], True),
            ([{"task": "Test task", "action": "a3"}], False),
        ]
        
        for traj, success in trajectories:
            self.analyzer.add_trajectory(traj, success)
        
        rate = self.analyzer.get_task_success_rate("Test task")
        assert rate == pytest.approx(2/3, abs=0.01)
    
    def test_get_hardest_tasks(self):
        """Test getting hardest tasks."""
        # Add tasks with different success rates
        for _ in range(5):
            self.analyzer.add_trajectory([{"task": "Easy task"}], success=True)
        for _ in range(5):
            self.analyzer.add_trajectory([{"task": "Hard task"}], success=False)
        for _ in range(3):
            self.analyzer.add_trajectory([{"task": "Medium task"}], success=True)
        for _ in range(2):
            self.analyzer.add_trajectory([{"task": "Medium task"}], success=False)
        
        hardest = self.analyzer.get_hardest_tasks(k=3)
        
        assert hardest[0][0] == "Hard task"
        assert hardest[0][1] == 0.0  # 0% success rate
    
    def test_cluster_failures(self):
        """Test failure clustering."""
        # Add enough failures for clustering
        for i in range(20):
            traj = [{"task": f"Task {i % 3}", "action": f"action_{i}", "observation": f"obs_{i}"}]
            self.analyzer.add_trajectory(traj, success=False)
        
        clusters = self.analyzer.cluster_failures(n_clusters=3)
        
        assert len(clusters) >= 1  # At least one cluster
    
    def test_save_load(self):
        """Test saving and loading state."""
        # Add some data
        self.analyzer.add_trajectory([{"task": "Test"}], success=True)
        self.analyzer.add_trajectory([{"task": "Test"}], success=False)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "analyzer.json")
            self.analyzer.save(path)
            
            # Load into new analyzer
            new_analyzer = FailureAnalyzer()
            new_analyzer.load(path)
            
            assert new_analyzer.task_stats["Test"]["total"] == 2


class TestDifficultyTracker:
    """Tests for DifficultyTracker."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.tracker = DifficultyTracker(ema_alpha=0.3, min_attempts=3)
    
    def test_update_task(self):
        """Test updating task statistics."""
        self.tracker.update("Task A", success=True)
        self.tracker.update("Task A", success=True)
        self.tracker.update("Task A", success=False)
        
        assert "Task A" in self.tracker.tasks
        # With 2/3 success, difficulty should be ~0.33
        difficulty = self.tracker.get_difficulty("Task A")
        assert 0.2 < difficulty < 0.5
    
    def test_learning_potential(self):
        """Test learning potential computation."""
        # 50% success rate should have highest learning potential
        for _ in range(5):
            self.tracker.update("Balanced", success=True)
        for _ in range(5):
            self.tracker.update("Balanced", success=False)
        
        # Very easy task
        for _ in range(10):
            self.tracker.update("Easy", success=True)
        
        balanced_lp = self.tracker.get_learning_potential("Balanced")
        easy_lp = self.tracker.get_learning_potential("Easy")
        
        # Balanced should have higher learning potential
        assert balanced_lp > easy_lp
    
    def test_sampling_weights(self):
        """Test sampling weight computation."""
        tasks = ["Task A", "Task B", "Task C"]
        
        # Add varied success rates
        for _ in range(5):
            self.tracker.update("Task A", success=True)
        for _ in range(5):
            self.tracker.update("Task B", success=True)
        for _ in range(5):
            self.tracker.update("Task B", success=False)
        
        weights = self.tracker.get_sampling_weights(tasks)
        
        assert len(weights) == 3
        assert np.sum(weights) == pytest.approx(1.0)
    
    def test_frontier_tasks(self):
        """Test getting frontier tasks."""
        for _ in range(5):
            self.tracker.update("Frontier", success=True)
        for _ in range(5):
            self.tracker.update("Frontier", success=False)
        
        for _ in range(10):
            self.tracker.update("Mastered", success=True)
        
        frontier = self.tracker.get_frontier_tasks(k=1)
        
        assert frontier[0] == "Frontier"


class TestTrajectoryWeighter:
    """Tests for TrajectoryWeighter."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.tracker = DifficultyTracker()
        self.weighter = TrajectoryWeighter(difficulty_tracker=self.tracker)
    
    def test_compute_weights(self):
        """Test weight computation."""
        trajectories = [
            [{"task": "A", "action": "x"}, {"done": True, "trajectory_reward": 1.0}],
            [{"task": "B", "action": "y"}, {"done": False, "trajectory_reward": 0.0}],
        ]
        
        weights = self.weighter.compute_weights(trajectories, normalize=True)
        
        assert len(weights) == 2
        assert np.sum(weights) == pytest.approx(1.0)
    
    def test_near_miss_scoring(self):
        """Test that near-misses get higher weights than hopeless failures."""
        # Long trajectory (almost succeeded)
        near_miss = [
            {"task": "Task", "action": f"action_{i}"}
            for i in range(8)
        ] + [{"done": False, "trajectory_reward": 0.0}]
        
        # Short trajectory (gave up early)
        hopeless = [
            {"task": "Task", "action": "action_0"},
            {"done": False, "trajectory_reward": 0.0}
        ]
        
        weights = self.weighter.compute_weights([near_miss, hopeless], normalize=False)
        
        # Near-miss should have higher weight
        assert weights[0] > weights[1]
    
    def test_filter_by_weight(self):
        """Test filtering trajectories by weight."""
        trajectories = [
            [{"task": "A"}, {"trajectory_reward": 1.0, "done": True}],
            [{"task": "B"}, {"trajectory_reward": 0.0, "done": False}],
            [{"task": "C"}, {"trajectory_reward": 0.5, "done": True}],
        ]
        
        filtered = self.weighter.filter_by_weight(trajectories, percentile=0.5)
        
        # Should keep roughly top half
        assert len(filtered) >= 1


class TestAnalyticsEngine:
    """Tests for AnalyticsEngine."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.engine = AnalyticsEngine()
    
    def test_process_trajectories(self):
        """Test processing trajectory batches."""
        trajectories = [
            [{"task": "Task A"}, {"trajectory_reward": 1.0, "done": True}],
            [{"task": "Task B"}, {"trajectory_reward": 0.0, "done": False}],
        ]
        
        results = self.engine.process_trajectories(trajectories)
        
        assert results["num_trajectories"] == 2
        assert results["successes"] == 1
        assert results["failures"] == 1
        assert results["success_rate"] == 0.5
    
    def test_get_wandb_metrics(self):
        """Test wandb metrics generation."""
        trajectories = [
            [{"task": "Task A"}, {"trajectory_reward": 1.0, "done": True}],
        ]
        self.engine.process_trajectories(trajectories)
        
        metrics = self.engine.get_wandb_metrics()
        
        assert "adat/total_trajectories" in metrics
        assert "adat/success_rate" in metrics
    
    def test_save_load(self):
        """Test saving and loading engine state."""
        trajectories = [
            [{"task": "Task A"}, {"trajectory_reward": 1.0, "done": True}],
        ]
        self.engine.process_trajectories(trajectories)
        self.engine.step()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            self.engine.save(tmpdir)
            
            new_engine = AnalyticsEngine()
            new_engine.load(tmpdir)
            
            assert new_engine.iteration == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
