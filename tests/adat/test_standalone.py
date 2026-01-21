"""
Standalone Tests for ADAT Core Components.

These tests can run without the full DigiRL environment installed.
Run with: python tests/adat/test_standalone.py
"""

import sys
import os

# Add the digirl package to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import tempfile
import json


def test_failure_analyzer():
    """Test FailureAnalyzer component."""
    print("Testing FailureAnalyzer...")
    
    from digirl.adat.core.failure_analyzer import FailureAnalyzer
    
    analyzer = FailureAnalyzer(use_embeddings=True)
    
    # Test adding trajectories
    traj1 = [
        {"task": "Open settings", "action": "tap", "observation": "home"},
        {"task": "Open settings", "action": "swipe", "observation": "settings"}
    ]
    analyzer.add_trajectory(traj1, success=True, reward=1.0)
    assert len(analyzer.trajectories) == 1, "Failed to add trajectory"
    
    # Add more for testing
    for i in range(20):
        traj = [{"task": f"Task {i % 5}", "action": f"action_{i}"}]
        success = i % 3 == 0  # ~33% success rate
        analyzer.add_trajectory(traj, success=success, reward=1.0 if success else 0.0)
    
    # Test success rate
    rate = analyzer.get_task_success_rate("Task 0")
    assert 0 <= rate <= 1, "Invalid success rate"
    
    # Test clustering
    clusters = analyzer.cluster_failures(n_clusters=3)
    assert len(clusters) >= 1, "Clustering failed"
    
    # Test hardest tasks
    hardest = analyzer.get_hardest_tasks(k=3)
    assert len(hardest) >= 1, "Failed to get hardest tasks"
    
    # Test save/load
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "analyzer.json")
        analyzer.save(path)
        assert os.path.exists(path), "Save failed"
        
        new_analyzer = FailureAnalyzer()
        new_analyzer.load(path)
        assert len(new_analyzer.task_stats) > 0, "Load failed"
    
    print("✓ FailureAnalyzer: All tests passed!")
    return True


def test_difficulty_tracker():
    """Test DifficultyTracker component."""
    print("Testing DifficultyTracker...")
    
    from digirl.adat.core.difficulty_tracker import DifficultyTracker
    
    tracker = DifficultyTracker(ema_alpha=0.3, min_attempts=3)
    
    # Update with successes/failures
    for _ in range(10):
        tracker.update("Easy Task", success=True)
    
    for _ in range(10):
        tracker.update("Hard Task", success=False)
    
    for _ in range(5):
        tracker.update("Medium Task", success=True)
    for _ in range(5):
        tracker.update("Medium Task", success=False)
    
    # Test difficulty
    easy_diff = tracker.get_difficulty("Easy Task")
    hard_diff = tracker.get_difficulty("Hard Task")
    assert easy_diff < hard_diff, "Difficulty ordering wrong"
    
    # Test learning potential (medium should be highest)
    easy_lp = tracker.get_learning_potential("Easy Task")
    medium_lp = tracker.get_learning_potential("Medium Task")
    hard_lp = tracker.get_learning_potential("Hard Task")
    assert medium_lp > easy_lp, "Learning potential should be highest at 50%"
    assert medium_lp > hard_lp, "Learning potential should be highest at 50%"
    
    # Test sampling weights
    weights = tracker.get_sampling_weights(["Easy Task", "Medium Task", "Hard Task"])
    assert len(weights) == 3, "Wrong number of weights"
    assert np.isclose(weights.sum(), 1.0), "Weights should sum to 1"
    
    # Test frontier tasks
    frontier = tracker.get_frontier_tasks(k=2)
    assert "Medium Task" in frontier[:2], "Medium task should be at frontier"
    
    print("✓ DifficultyTracker: All tests passed!")
    return True


def test_trajectory_weighter():
    """Test TrajectoryWeighter component."""
    print("Testing TrajectoryWeighter...")
    
    from digirl.adat.core.difficulty_tracker import DifficultyTracker
    from digirl.adat.core.trajectory_weighter import TrajectoryWeighter
    
    tracker = DifficultyTracker()
    weighter = TrajectoryWeighter(difficulty_tracker=tracker)
    
    # Create test trajectories
    trajectories = [
        # Success
        [{"task": "Task A", "action": "a1"}, {"done": True, "trajectory_reward": 1.0}],
        # Short failure (hopeless)
        [{"task": "Task B", "action": "a1"}, {"done": False, "trajectory_reward": 0.0}],
        # Long failure (near-miss)
        [{"task": "Task C", "action": f"a{i}"} for i in range(8)] + [{"done": False, "trajectory_reward": 0.0}],
    ]
    
    weights = weighter.compute_weights(trajectories, normalize=True)
    assert len(weights) == 3, "Wrong number of weights"
    assert np.isclose(weights.sum(), 1.0), "Weights should sum to 1"
    
    # Near-miss (long failure) should have higher weight than short failure
    assert weights[2] > weights[1], "Near-miss should have higher weight"
    
    # Test filtering
    filtered = weighter.filter_by_weight(trajectories, percentile=0.5)
    assert len(filtered) >= 1, "Filtering should keep some trajectories"
    
    print("✓ TrajectoryWeighter: All tests passed!")
    return True


def test_analytics_engine():
    """Test AnalyticsEngine component."""
    print("Testing AnalyticsEngine...")
    
    from digirl.adat.core.analytics_engine import AnalyticsEngine
    
    engine = AnalyticsEngine()
    
    # Process trajectories
    trajectories = [
        [{"task": "Task A"}, {"trajectory_reward": 1.0, "done": True}],
        [{"task": "Task B"}, {"trajectory_reward": 0.0, "done": False}],
        [{"task": "Task A"}, {"trajectory_reward": 1.0, "done": True}],
    ]
    
    results = engine.process_trajectories(trajectories)
    assert results["num_trajectories"] == 3, "Wrong trajectory count"
    assert results["successes"] == 2, "Wrong success count"
    assert results["failures"] == 1, "Wrong failure count"
    
    # Test wandb metrics
    metrics = engine.get_wandb_metrics()
    assert "adat/total_trajectories" in metrics, "Missing wandb metric"
    assert "adat/success_rate" in metrics, "Missing wandb metric"
    
    # Test step
    engine.step()
    assert engine.iteration == 1, "Iteration not incremented"
    
    # Test save/load
    with tempfile.TemporaryDirectory() as tmpdir:
        engine.save(tmpdir)
        
        new_engine = AnalyticsEngine()
        new_engine.load(tmpdir)
        assert new_engine.iteration == 1, "State not restored"
    
    print("✓ AnalyticsEngine: All tests passed!")
    return True


def test_adaptive_sampler():
    """Test AdaptiveSampler component."""
    print("Testing AdaptiveSampler...")
    
    from digirl.adat.core.difficulty_tracker import DifficultyTracker
    from digirl.adat.samplers.adaptive_sampler import AdaptiveSampler
    
    tasks = ["Easy", "Medium", "Hard", "New"]
    tracker = DifficultyTracker()
    sampler = AdaptiveSampler(
        all_tasks=tasks,
        difficulty_tracker=tracker,
        strategy="zpd"
    )
    
    # Setup difficulties
    for _ in range(10):
        tracker.update("Easy", success=True)
        tracker.update("Hard", success=False)
    for _ in range(5):
        tracker.update("Medium", success=True)
        tracker.update("Medium", success=False)
    
    # Test sampling
    samples = sampler.sample(n=100)
    assert len(samples) == 100, "Wrong sample count"
    
    # Check distribution favors frontier (Medium or New)
    counts = {t: samples.count(t) for t in tasks}
    assert counts["Medium"] > 0, "Medium should be sampled"
    assert counts["New"] > 0, "New tasks should be sampled"
    
    # Test strategy change
    sampler.set_strategy("uniform")
    samples = sampler.sample(n=100)
    assert len(samples) == 100, "Sampling failed after strategy change"
    
    print("✓ AdaptiveSampler: All tests passed!")
    return True


def test_curriculum_sampler():
    """Test CurriculumSampler component."""
    print("Testing CurriculumSampler...")
    
    from digirl.adat.core.difficulty_tracker import DifficultyTracker
    from digirl.adat.samplers.curriculum_sampler import CurriculumSampler
    
    tasks = [f"Task_{i}" for i in range(20)]
    tracker = DifficultyTracker()
    
    # Setup varying difficulties
    for i, task in enumerate(tasks):
        for _ in range(10):
            success = np.random.random() > (i / 20)
            tracker.update(task, success)
    
    curriculum = CurriculumSampler(
        all_tasks=tasks,
        difficulty_tracker=tracker,
        n_stages=4,
        advancement_threshold=0.7,
        min_samples_per_stage=5
    )
    
    # Test initial stage
    assert curriculum.current_stage == 0, "Should start at stage 0"
    
    # Test sampling
    samples = curriculum.sample(n=10)
    assert len(samples) == 10, "Wrong sample count"
    
    # Simulate advancement
    initial_stage = curriculum.current_stage
    for _ in range(10):
        task = curriculum.sample(1)[0]
        curriculum.update(task, success=True)
    
    # Should have advanced after many successes
    assert curriculum.current_stage >= initial_stage, "Should advance on success"
    
    # Test progress
    progress = curriculum.get_progress()
    assert "current_stage" in progress, "Missing progress info"
    assert "stage_success_rate" in progress, "Missing progress info"
    
    print("✓ CurriculumSampler: All tests passed!")
    return True


def test_hyperparameter_scheduler():
    """Test HyperparameterScheduler component."""
    print("Testing HyperparameterScheduler...")
    
    from digirl.adat.schedulers.hyperparameter_scheduler import HyperparameterScheduler
    
    scheduler = HyperparameterScheduler()
    
    # Initial state
    config = scheduler.get_config_dict()
    assert "temperature" in config, "Missing temperature"
    assert "actor_epochs" in config, "Missing actor_epochs"
    
    # Simulate low success rate -> should increase temperature
    initial_temp = scheduler.state.temperature
    for i in range(10):
        scheduler.step(success_rate=0.1, iteration=i)
    
    assert scheduler.state.temperature >= initial_temp, "Temperature should increase on low success"
    
    # Simulate high success rate -> should decrease temperature
    scheduler.reset()
    initial_temp = scheduler.state.temperature
    for i in range(10):
        scheduler.step(success_rate=0.9, iteration=i)
    
    assert scheduler.state.temperature <= initial_temp, "Temperature should decrease on high success"
    
    print("✓ HyperparameterScheduler: All tests passed!")
    return True


def run_all_tests():
    """Run all standalone tests."""
    print("=" * 60)
    print("ADAT Standalone Tests")
    print("=" * 60)
    print()
    
    tests = [
        test_failure_analyzer,
        test_difficulty_tracker,
        test_trajectory_weighter,
        test_analytics_engine,
        test_adaptive_sampler,
        test_curriculum_sampler,
        test_hyperparameter_scheduler,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ {test.__name__}: FAILED - {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        print()
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
