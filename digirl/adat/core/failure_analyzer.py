"""
Failure Analyzer - Clusters failed trajectories to identify failure patterns.

This module analyzes trajectory failures using embedding-based clustering
to identify common failure modes and inform adaptive training.
"""

import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
import json
import os


@dataclass
class TrajectoryRecord:
    """Record of a trajectory with metadata."""
    trajectory: List[dict]
    task: str
    success: bool
    reward: float
    features: Optional[np.ndarray] = None
    cluster_id: int = -1


class FailureAnalyzer:
    """
    Analyzes failed trajectories to identify failure patterns and clusters.
    
    This is a core component of ADAT that enables:
    - Clustering of failures by similarity
    - Identification of hard tasks
    - Tracking of failure patterns over time
    
    Attributes:
        trajectories: List of all recorded trajectories
        task_stats: Per-task success/failure statistics
        use_embeddings: Whether to use text embeddings for clustering
    """
    
    def __init__(
        self,
        use_embeddings: bool = True,
        embedding_dim: int = 384,
        max_trajectories: int = 10000,
        window_size: int = 500
    ):
        """
        Initialize the FailureAnalyzer.
        
        Args:
            use_embeddings: Whether to compute embeddings for clustering
            embedding_dim: Dimension of text embeddings
            max_trajectories: Maximum trajectories to keep in memory
            window_size: Sliding window for recent statistics
        """
        self.use_embeddings = use_embeddings
        self.embedding_dim = embedding_dim
        self.max_trajectories = max_trajectories
        self.window_size = window_size
        
        self.trajectories: List[TrajectoryRecord] = []
        self.task_stats: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"success": 0, "total": 0, "recent_success": 0, "recent_total": 0}
        )
        
        # Simple TF-IDF-like encoder (no external dependencies)
        self._vocabulary: Dict[str, int] = {}
        self._idf: Optional[np.ndarray] = None
        
        # Cluster information
        self.clusters: Dict[int, List[int]] = defaultdict(list)
        self.cluster_labels: Dict[int, str] = {}
        
    def add_trajectory(
        self, 
        trajectory: List[dict], 
        success: bool,
        reward: float = 0.0
    ) -> None:
        """
        Add a trajectory for analysis.
        
        Args:
            trajectory: List of step dictionaries with observation, action, etc.
            success: Whether the trajectory completed the task successfully
            reward: Total reward obtained
        """
        if len(trajectory) == 0:
            return
            
        task = trajectory[0].get("task", "unknown")
        
        # Update task statistics
        self.task_stats[task]["total"] += 1
        self.task_stats[task]["recent_total"] += 1
        if success:
            self.task_stats[task]["success"] += 1
            self.task_stats[task]["recent_success"] += 1
        
        # Create record
        record = TrajectoryRecord(
            trajectory=trajectory,
            task=task,
            success=success,
            reward=reward
        )
        
        # Compute features if using embeddings
        if self.use_embeddings:
            record.features = self._compute_features(trajectory)
        
        self.trajectories.append(record)
        
        # Maintain max size
        if len(self.trajectories) > self.max_trajectories:
            removed = self.trajectories.pop(0)
            # Update recent stats by decrementing old values
            old_task = removed.task
            self.task_stats[old_task]["recent_total"] = max(
                0, self.task_stats[old_task]["recent_total"] - 1
            )
            if removed.success:
                self.task_stats[old_task]["recent_success"] = max(
                    0, self.task_stats[old_task]["recent_success"] - 1
                )
    
    def _compute_features(self, trajectory: List[dict]) -> np.ndarray:
        """
        Compute feature representation of a trajectory using bag-of-words.
        
        This is a lightweight alternative to neural embeddings that works
        without requiring additional model dependencies.
        """
        # Extract text from trajectory
        texts = []
        
        # Task description
        if trajectory and "task" in trajectory[0]:
            texts.append(trajectory[0]["task"])
        
        # Actions taken
        for step in trajectory:
            if "action" in step:
                texts.append(str(step["action"]))
            if "observation" in step:
                obs = step["observation"]
                if isinstance(obs, str):
                    texts.append(obs[:200])  # Truncate long observations
        
        # Combine and tokenize
        combined_text = " ".join(texts).lower()
        tokens = combined_text.split()
        
        # Update vocabulary
        for token in tokens:
            if token not in self._vocabulary:
                self._vocabulary[token] = len(self._vocabulary)
        
        # Create bag-of-words vector (sparse to dense)
        feature_size = min(len(self._vocabulary), self.embedding_dim)
        features = np.zeros(self.embedding_dim)
        
        for token in tokens:
            if token in self._vocabulary:
                idx = self._vocabulary[token] % self.embedding_dim
                features[idx] += 1
        
        # Normalize
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
            
        return features
    
    def cluster_failures(self, n_clusters: int = 5) -> Dict[int, List[TrajectoryRecord]]:
        """
        Cluster failed trajectories by similarity.
        
        Uses K-Means clustering on trajectory features to identify
        common failure patterns.
        
        Args:
            n_clusters: Number of clusters to create
            
        Returns:
            Dictionary mapping cluster ID to list of trajectories
        """
        # Get failed trajectories with features
        failed = [t for t in self.trajectories if not t.success and t.features is not None]
        
        if len(failed) < n_clusters:
            # Not enough data for clustering
            return {0: failed}
        
        # Stack features
        features = np.stack([t.features for t in failed])
        
        # Simple K-Means implementation
        labels = self._kmeans(features, n_clusters)
        
        # Group by cluster
        clusters: Dict[int, List[TrajectoryRecord]] = defaultdict(list)
        for i, (record, label) in enumerate(zip(failed, labels)):
            record.cluster_id = int(label)
            clusters[int(label)].append(record)
        
        # Label clusters by most common task
        self.cluster_labels = {}
        for cluster_id, records in clusters.items():
            task_counts = defaultdict(int)
            for r in records:
                task_counts[r.task] += 1
            most_common = max(task_counts.items(), key=lambda x: x[1])
            self.cluster_labels[cluster_id] = f"cluster_{cluster_id}_{most_common[0][:20]}"
        
        self.clusters = {k: [self.trajectories.index(r) for r in v] for k, v in clusters.items()}
        
        return clusters
    
    def _kmeans(self, X: np.ndarray, k: int, max_iters: int = 100) -> np.ndarray:
        """Simple K-Means implementation."""
        n_samples = X.shape[0]
        
        # Initialize centroids randomly
        np.random.seed(42)
        indices = np.random.choice(n_samples, k, replace=False)
        centroids = X[indices].copy()
        
        labels = np.zeros(n_samples, dtype=int)
        
        for _ in range(max_iters):
            # Assign to nearest centroid
            distances = np.zeros((n_samples, k))
            for i in range(k):
                distances[:, i] = np.linalg.norm(X - centroids[i], axis=1)
            new_labels = np.argmin(distances, axis=1)
            
            # Check convergence
            if np.all(labels == new_labels):
                break
            labels = new_labels
            
            # Update centroids
            for i in range(k):
                mask = labels == i
                if np.sum(mask) > 0:
                    centroids[i] = X[mask].mean(axis=0)
        
        return labels
    
    def get_task_success_rate(self, task: str, use_recent: bool = True) -> float:
        """
        Get the success rate for a specific task.
        
        Args:
            task: Task name
            use_recent: Use recent window statistics
            
        Returns:
            Success rate between 0 and 1
        """
        stats = self.task_stats.get(task)
        if stats is None:
            return 0.5  # Unknown task, assume 50%
        
        if use_recent:
            total = stats["recent_total"]
            success = stats["recent_success"]
        else:
            total = stats["total"]
            success = stats["success"]
        
        if total == 0:
            return 0.5
        return success / total
    
    def get_hardest_tasks(self, k: int = 10) -> List[Tuple[str, float]]:
        """
        Get tasks with the lowest success rates.
        
        Args:
            k: Number of tasks to return
            
        Returns:
            List of (task_name, success_rate) tuples, sorted by difficulty
        """
        task_rates = []
        for task, stats in self.task_stats.items():
            if stats["total"] >= 3:  # Minimum samples
                rate = stats["success"] / stats["total"]
                task_rates.append((task, rate))
        
        task_rates.sort(key=lambda x: x[1])
        return task_rates[:k]
    
    def get_easiest_tasks(self, k: int = 10) -> List[Tuple[str, float]]:
        """Get tasks with the highest success rates."""
        task_rates = []
        for task, stats in self.task_stats.items():
            if stats["total"] >= 3:
                rate = stats["success"] / stats["total"]
                task_rates.append((task, rate))
        
        task_rates.sort(key=lambda x: x[1], reverse=True)
        return task_rates[:k]
    
    def get_cluster_statistics(self) -> Dict[int, Dict[str, Any]]:
        """Get statistics for each failure cluster."""
        stats = {}
        for cluster_id, indices in self.clusters.items():
            records = [self.trajectories[i] for i in indices if i < len(self.trajectories)]
            if not records:
                continue
                
            tasks = [r.task for r in records]
            task_counts = defaultdict(int)
            for t in tasks:
                task_counts[t] += 1
            
            stats[cluster_id] = {
                "size": len(records),
                "label": self.cluster_labels.get(cluster_id, f"cluster_{cluster_id}"),
                "avg_reward": np.mean([r.reward for r in records]),
                "top_tasks": sorted(task_counts.items(), key=lambda x: -x[1])[:3]
            }
        
        return stats
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the failure analysis."""
        total = len(self.trajectories)
        if total == 0:
            return {"total": 0, "success_rate": 0}
        
        successes = sum(1 for t in self.trajectories if t.success)
        
        return {
            "total_trajectories": total,
            "success_rate": successes / total,
            "num_tasks": len(self.task_stats),
            "hardest_tasks": self.get_hardest_tasks(5),
            "easiest_tasks": self.get_easiest_tasks(5),
            "cluster_stats": self.get_cluster_statistics()
        }
    
    def save(self, path: str) -> None:
        """Save analyzer state to file."""
        state = {
            "task_stats": dict(self.task_stats),
            "cluster_labels": self.cluster_labels,
            "vocabulary_size": len(self._vocabulary)
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(state, f)
    
    def load(self, path: str) -> None:
        """Load analyzer state from file."""
        with open(path, 'r') as f:
            state = json.load(f)
        
        self.task_stats = defaultdict(
            lambda: {"success": 0, "total": 0, "recent_success": 0, "recent_total": 0}
        )
        self.task_stats.update(state.get("task_stats", {}))
        self.cluster_labels = state.get("cluster_labels", {})
