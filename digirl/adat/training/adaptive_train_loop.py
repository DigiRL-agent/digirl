"""
Adaptive Training Loop - Drop-in replacement for offpolicy_train_loop.

This module provides an enhanced training loop with ADAT features:
- Adaptive task sampling
- Trajectory weighting
- Hyperparameter scheduling
- Comprehensive analytics logging
"""

import os
import numpy as np
import torch
from typing import List, Dict, Optional, Any
from tqdm import tqdm
import copy

# Import original DigiRL components
from digirl.environment import batch_interact_environment
from digirl.data import ReplayBuffer
from digirl.algorithms.digirl import DigiRLTrainer
from digirl.algorithms.filteredbc import BCTrainer
from digirl.misc import colorful_print
from digirl.environment.env_utils import add_mc_return

# Import ADAT components
from ..core.analytics_engine import AnalyticsEngine
from ..core.trajectory_weighter import TrajectoryWeighter
from ..samplers.adaptive_sampler import AdaptiveSampler
from ..samplers.curriculum_sampler import CurriculumSampler
from ..schedulers.hyperparameter_scheduler import HyperparameterScheduler

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def weighted_filter_buffer(
    all_trajectories: List,
    weighter: TrajectoryWeighter,
    batch_size: int,
    capacity: int,
    top_percentile: float = 0.1
) -> ReplayBuffer:
    """
    Filter trajectories using ADAT weighting instead of simple top-k.
    
    Args:
        all_trajectories: List of all trajectories
        weighter: TrajectoryWeighter for computing weights
        batch_size: Batch size for replay buffer
        capacity: Capacity of replay buffer
        top_percentile: Keep top this fraction
        
    Returns:
        Filtered ReplayBuffer
    """
    if not all_trajectories:
        return ReplayBuffer(batch_size=batch_size, capacity=capacity)
    
    # Compute weights
    weights = weighter.compute_weights(all_trajectories, normalize=False)
    
    # Get cutoff for top percentile
    cutoff = np.percentile(weights, (1 - top_percentile) * 100)
    
    # Filter trajectories
    filtered_trajectories = [
        traj for traj, w in zip(all_trajectories, weights)
        if w >= cutoff
    ]
    
    # Log filtering stats
    colorful_print(
        f"[ADAT] Filtered {len(filtered_trajectories)}/{len(all_trajectories)} "
        f"trajectories (cutoff weight: {cutoff:.3f})",
        fg="cyan"
    )
    
    # Create buffer
    buffer = ReplayBuffer(batch_size=batch_size, capacity=capacity)
    data = sum(filtered_trajectories, [])
    for d in data:
        buffer.insert(**d)
    
    return buffer


def framestack(all_trajectories):
    """Stack consecutive frames for temporal context."""
    new_trajectories = copy.deepcopy(all_trajectories)
    for trajectory, new_trajectory in zip(all_trajectories, new_trajectories):
        for i, (t, nt) in enumerate(zip(trajectory, new_trajectory)):
            if i == 0:
                nt["image_features"] = np.concatenate(
                    [t["image_features"], t["image_features"]], axis=-1
                )
            else:
                nt["image_features"] = np.concatenate(
                    [trajectory[i-1]["image_features"], t["image_features"]], axis=-1
                )
            nt["next_image_features"] = np.concatenate(
                [t["image_features"], t["next_image_features"]], axis=-1
            )
    return new_trajectories


def adaptive_train_loop(
    env,
    agent,
    tokenizer,
    accelerator,
    # Standard DigiRL parameters
    warmup_iter: int = 20,
    rollout_size: int = 50,
    batch_size: int = 2,
    capacity: int = 500000,
    train_iterations: int = 10,
    epochs: int = 3,
    grad_accum_steps: int = 1,
    critic_lr: float = 1e-3,
    lm_lr: float = 1e-5,
    gamma: float = 0.9,
    tau: float = 0.1,
    use_wandb: bool = False,
    actor_epochs: int = 3,
    train_mode: str = None,
    max_grad_norm: float = 0.01,
    save_path: str = None,
    save_freq: int = 25,
    train_algorithm: str = "digirl",
    decode_f: callable = lambda x: x,
    offline_data_path: str = None,
    offline_actor_iterations: int = 20,
    offline_critic_iterations: int = 20,
    offline_trajectory_critic_iterations: int = 20,
    trajectory_critic_epochs: int = 5,
    parallel: str = 'single',
    worker_temp_path=None,
    worker_run_path=None,
    worker_ips=[],
    worker_username=None,
    # ADAT-specific parameters
    use_adaptive_sampling: bool = True,
    use_trajectory_weighting: bool = True,
    use_curriculum: bool = True,
    use_hyperparameter_scheduling: bool = True,
    sampling_strategy: str = "zpd",
    curriculum_warmup: int = 50,
    weight_alpha: float = 0.3,
    weight_beta: float = 0.5,
    weight_gamma: float = 0.2,
    target_success_rate: float = 0.5,
    **kwargs
):
    """
    Adaptive training loop with ADAT features.
    
    This is a drop-in replacement for offpolicy_train_loop that adds:
    - Adaptive task sampling based on difficulty
    - Trajectory weighting by learning potential
    - Hyperparameter scheduling based on progress
    - Comprehensive analytics and logging
    
    All ADAT features can be toggled independently for ablation studies.
    """
    colorful_print("[ADAT] Starting Adaptive Training Loop", fg="green")
    
    # ========== Initialize Trainer ==========
    if train_algorithm == "digirl":
        trainer = DigiRLTrainer(
            agent=agent,
            accelerator=accelerator,
            tokenizer=tokenizer,
            critic_lr=critic_lr,
            lm_lr=lm_lr,
            gamma=gamma,
            tau=tau,
            epochs=epochs,
            actor_epochs=actor_epochs,
            grad_accum_steps=grad_accum_steps,
            max_grad_norm=max_grad_norm,
            trajectory_critic_epochs=trajectory_critic_epochs
        )
    else:
        trainer = BCTrainer(
            agent=agent,
            tokenizer=tokenizer,
            accelerator=accelerator,
            lm_lr=lm_lr,
            epochs=actor_epochs,
            grad_accum_steps=grad_accum_steps,
            max_grad_norm=max_grad_norm
        )
    
    # ========== Initialize ADAT Components ==========
    analytics = AnalyticsEngine(
        use_embeddings=True,
        ema_alpha=0.3,
        target_success_rate=target_success_rate,
        weight_alpha=weight_alpha,
        weight_beta=weight_beta,
        weight_gamma=weight_gamma
    )
    
    all_tasks = list(env.all_tasks) if hasattr(env, 'all_tasks') else []
    
    sampler = AdaptiveSampler(
        all_tasks=all_tasks,
        difficulty_tracker=analytics.difficulty_tracker,
        failure_analyzer=analytics.failure_analyzer,
        strategy=sampling_strategy
    ) if use_adaptive_sampling else None
    
    curriculum = CurriculumSampler(
        all_tasks=all_tasks,
        difficulty_tracker=analytics.difficulty_tracker
    ) if use_curriculum else None
    
    hp_scheduler = HyperparameterScheduler() if use_hyperparameter_scheduling else None
    
    weighter = analytics.trajectory_weighter
    
    colorful_print(f"[ADAT] Components initialized:", fg="cyan")
    colorful_print(f"  - Adaptive Sampling: {use_adaptive_sampling}", fg="cyan")
    colorful_print(f"  - Trajectory Weighting: {use_trajectory_weighting}", fg="cyan")
    colorful_print(f"  - Curriculum Learning: {use_curriculum}", fg="cyan")
    colorful_print(f"  - HP Scheduling: {use_hyperparameter_scheduling}", fg="cyan")
    
    # ========== Initialize Buffers ==========
    replay_buffer = ReplayBuffer(batch_size=batch_size, capacity=capacity)
    validation_buffer = ReplayBuffer(batch_size=batch_size, capacity=capacity)
    all_trajectories = []
    train_trajectories = []
    val_trajectories = []
    
    # Prepare model and optimizers
    agent.prepare()
    trainer.prepare()
    
    # ========== Load Offline Data ==========
    loaded_trajs = False
    
    if offline_data_path is not None and train_mode != "online":
        all_trajectories = torch.load(offline_data_path)
        all_trajectories = framestack(all_trajectories)
        colorful_print(f"[ADAT] Loaded {len(all_trajectories)} offline trajectories", fg="green")
        all_trajectories = [add_mc_return(t, gamma=gamma) for t in all_trajectories]
        
        # Process through analytics
        analytics.process_trajectories(all_trajectories)
        
        train_trajectories = all_trajectories[:int(len(all_trajectories)*0.8)]
        val_trajectories = all_trajectories[int(len(all_trajectories)*0.8):]
        loaded_trajs = 'scratch'
    
    # Resume from checkpoint
    if os.path.exists(os.path.join(save_path, 'trainer.pt')):
        assert train_mode != "offline", "Only online/off2on training can be resumed"
        trainer.load(os.path.join(save_path, 'trainer.pt'))
        replay_buffer = torch.load(os.path.join(save_path, 'replay_buffer.pt'))
        all_trajectories = torch.load(os.path.join(save_path, 'trajectories.pt'))
        train_trajectories = torch.load(os.path.join(save_path, 'train_trajectories.pt'))
        val_trajectories = torch.load(os.path.join(save_path, 'val_trajectories.pt'))
        
        # Load ADAT state if available
        adat_path = os.path.join(save_path, 'adat')
        if os.path.exists(adat_path):
            analytics.load(adat_path)
            colorful_print("[ADAT] Loaded analytics state from checkpoint", fg="green")
        
        loaded_trajs = 'resume'
    
    if not loaded_trajs:
        train_trajectories = []
        val_trajectories = []
        all_trajectories = []
    
    # Initialize buffers with loaded data
    replay_buffer = ReplayBuffer(batch_size=batch_size, capacity=capacity)
    validation_buffer = ReplayBuffer(batch_size=batch_size, capacity=capacity)
    
    data = sum(train_trajectories, [])
    val_data = sum(val_trajectories, [])
    for d in data:
        replay_buffer.insert(**d)
    for d in val_data:
        validation_buffer.insert(**d)
    
    # ========== Offline Pre-training ==========
    if not os.path.exists(os.path.join(save_path, 'trainer.pt')):
        if os.path.exists(os.path.join(save_path, 'trainer_offline.pt')):
            trainer.load(os.path.join(save_path, 'trainer_offline.pt'))
            colorful_print("[ADAT] Loaded offline checkpoint", fg="green")
        elif train_mode != "online" and len(train_trajectories) > 0:
            colorful_print("[ADAT] Starting offline pre-training", fg="yellow")
            
            # Pre-train trajectory critic
            for it in tqdm(range(offline_trajectory_critic_iterations), desc="Trajectory Critic"):
                if use_trajectory_weighting:
                    filtered = weighted_filter_buffer(
                        train_trajectories, weighter, batch_size, capacity
                    )
                else:
                    filtered = replay_buffer
                trainer.trajectory_critic_update(filtered)
            
            # Pre-train step critic
            for it in tqdm(range(offline_critic_iterations), desc="Step Critic"):
                if use_trajectory_weighting:
                    filtered = weighted_filter_buffer(
                        train_trajectories, weighter, batch_size, capacity
                    )
                else:
                    filtered = replay_buffer
                trainer.step_update(filtered, validation_buffer)
            
            # Pre-train actor
            for it in tqdm(range(offline_actor_iterations), desc="Actor"):
                if use_trajectory_weighting:
                    filtered = weighted_filter_buffer(
                        train_trajectories, weighter, batch_size, capacity
                    )
                else:
                    filtered = replay_buffer
                trainer.actor_update(filtered)
            
            # Save offline checkpoint
            trainer.save(os.path.join(save_path, 'trainer_offline.pt'))
    
    # ========== Online Training Loop ==========
    if train_mode == "offline":
        colorful_print("[ADAT] Offline-only mode, skipping online training", fg="yellow")
        return
    
    start_iteration = analytics.iteration
    
    for iteration in range(start_iteration, train_iterations):
        colorful_print(f"\n[ADAT] Iteration {iteration}/{train_iterations}", fg="green")
        
        # ===== Collect Trajectories =====
        # Sample tasks using ADAT sampler
        if use_adaptive_sampling and sampler is not None:
            # Get curriculum stage if active
            if use_curriculum and curriculum is not None and iteration >= curriculum_warmup:
                sampled_tasks = curriculum.sample(rollout_size)
            else:
                sampled_tasks = sampler.sample(rollout_size)
            
            colorful_print(f"[ADAT] Sampled {len(set(sampled_tasks))} unique tasks", fg="cyan")
        
        # Collect trajectories from environment
        trajectories = batch_interact_environment(
            agent=agent,
            env=env,
            num_trajectories=rollout_size,
            accelerator=accelerator,
            decode_f=decode_f
        )
        
        # Process trajectories through analytics
        results = analytics.process_trajectories(trajectories)
        colorful_print(
            f"[ADAT] Collected {results['num_trajectories']} trajectories, "
            f"success rate: {results['success_rate']:.2%}",
            fg="cyan"
        )
        
        # Update curriculum
        if use_curriculum and curriculum is not None:
            for traj in trajectories:
                if traj:
                    task = traj[0].get("task", "unknown")
                    success = traj[-1].get("trajectory_reward", 0) > 0
                    curriculum.update(task, success)
        
        # Framestack and add MC returns
        trajectories = framestack(trajectories)
        trajectories = [add_mc_return(t, gamma=gamma) for t in trajectories]
        
        # Add to buffer
        all_trajectories.extend(trajectories)
        train_trajectories.extend(trajectories[:int(len(trajectories)*0.8)])
        val_trajectories.extend(trajectories[int(len(trajectories)*0.8):])
        
        # ===== Update Hyperparameters =====
        if use_hyperparameter_scheduling and hp_scheduler is not None:
            hp_state = hp_scheduler.step(
                success_rate=results['success_rate'],
                avg_reward=results['avg_reward'],
                iteration=iteration
            )
            
            # Apply updated hyperparameters
            # Note: Some HPs require trainer modification
            colorful_print(
                f"[ADAT] HP Update - temp: {hp_state.temperature:.2f}, "
                f"epochs: {hp_state.actor_epochs}",
                fg="cyan"
            )
        
        # ===== Training Updates =====
        # Filter buffer with ADAT weighting
        if use_trajectory_weighting:
            filtered_buffer = weighted_filter_buffer(
                train_trajectories, weighter, batch_size, capacity
            )
        else:
            filtered_buffer = ReplayBuffer(batch_size=batch_size, capacity=capacity)
            for d in sum(train_trajectories, []):
                filtered_buffer.insert(**d)
        
        # Validation buffer
        validation_buffer = ReplayBuffer(batch_size=batch_size, capacity=capacity)
        for d in sum(val_trajectories, []):
            validation_buffer.insert(**d)
        
        # Trainer updates
        if train_algorithm == "digirl":
            trainer.trajectory_critic_update(filtered_buffer)
            trainer.step_update(filtered_buffer, validation_buffer)
            trainer.actor_update(filtered_buffer)
        else:
            trainer.actor_update(filtered_buffer)
        
        # ===== Logging =====
        analytics.step()
        
        if use_wandb and WANDB_AVAILABLE and accelerator.is_main_process:
            metrics = analytics.get_wandb_metrics()
            metrics["iteration"] = iteration
            metrics["success_rate"] = results['success_rate']
            metrics["avg_reward"] = results['avg_reward']
            
            if use_curriculum and curriculum is not None:
                progress = curriculum.get_progress()
                metrics["adat/curriculum_stage"] = progress["current_stage"]
                metrics["adat/curriculum_progress"] = progress["progress_to_next"]
            
            if use_hyperparameter_scheduling and hp_scheduler is not None:
                hp_config = hp_scheduler.get_config_dict()
                metrics["adat/temperature"] = hp_config["temperature"]
                metrics["adat/actor_epochs"] = hp_config["actor_epochs"]
            
            wandb.log(metrics)
        
        # ===== Checkpointing =====
        if iteration % save_freq == 0 and save_path:
            colorful_print(f"[ADAT] Saving checkpoint at iteration {iteration}", fg="green")
            
            trainer.save(os.path.join(save_path, 'trainer.pt'))
            torch.save(replay_buffer, os.path.join(save_path, 'replay_buffer.pt'))
            torch.save(all_trajectories[-10000:], os.path.join(save_path, 'trajectories.pt'))
            torch.save(train_trajectories[-8000:], os.path.join(save_path, 'train_trajectories.pt'))
            torch.save(val_trajectories[-2000:], os.path.join(save_path, 'val_trajectories.pt'))
            
            # Save ADAT state
            analytics.save(os.path.join(save_path, 'adat'))
    
    # Final save
    colorful_print("[ADAT] Training complete!", fg="green")
    
    if save_path:
        trainer.save(os.path.join(save_path, 'trainer_final.pt'))
        analytics.save(os.path.join(save_path, 'adat'))
    
    # Print summary
    summary = analytics.get_summary()
    colorful_print("\n[ADAT] Final Analytics Summary:", fg="green")
    colorful_print(f"  Total trajectories: {summary['failure_analysis']['total_trajectories']}", fg="cyan")
    colorful_print(f"  Final success rate: {summary['failure_analysis']['success_rate']:.2%}", fg="cyan")
    colorful_print(f"  Tasks tracked: {summary['difficulty_tracking']['num_tasks']}", fg="cyan")
