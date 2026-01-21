"""
ADAT Entry Point - Run DigiRL with Analytics-Driven Adaptive Training.

Usage:
    python run_adaptive.py --config-path config/adat --config-name default
    
This script is a drop-in replacement for run.py that enables ADAT features.
"""

import os
import sys
import hydra
from omegaconf import DictConfig, OmegaConf
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from accelerate import Accelerator


def setup_environment(cfg: DictConfig):
    """Set up environment variables from config."""
    if cfg.get("hf_token"):
        os.environ["HF_TOKEN"] = cfg.hf_token
    if cfg.get("wandb_token"):
        os.environ["WANDB_API_KEY"] = cfg.wandb_token
    if cfg.get("gemini_token"):
        os.environ["GEMINI_API_KEY"] = cfg.gemini_token


def load_tasks(cfg: DictConfig):
    """Load task set from file."""
    task_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "digirl", "environment", "android", "assets", "task_set",
        f"{cfg.task_set}_{cfg.task_split}.txt"
    )
    
    if os.path.exists(task_path):
        with open(task_path, 'r') as f:
            tasks = [line.strip() for line in f if line.strip()]
        return tasks
    else:
        print(f"Warning: Task file not found at {task_path}")
        return []


@hydra.main(config_path="config/adat", config_name="default", version_base="1.2")
def main(cfg: DictConfig):
    """Main entry point for ADAT training."""
    print("=" * 60)
    print("Analytics-Driven Adaptive Training (ADAT) for DigiRL")
    print("=" * 60)
    
    # Print configuration
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))
    
    # Setup
    setup_environment(cfg)
    
    # Initialize accelerator
    accelerator = Accelerator()
    
    # Load tasks
    all_tasks = load_tasks(cfg)
    print(f"\nLoaded {len(all_tasks)} tasks from {cfg.task_set}/{cfg.task_split}")
    
    # Initialize wandb if enabled
    if cfg.use_wandb and accelerator.is_main_process:
        import wandb
        wandb.init(
            project=cfg.get("wandb_project", "digirl-adat"),
            name=cfg.get("run_name", "adat-run"),
            config=OmegaConf.to_container(cfg, resolve=True)
        )
    
    # Import components
    from digirl.models import AutoUIAgent
    from digirl.environment import BatchedAndroidEnv
    from digirl.environment.android import EndResultEvaluator
    from digirl.adat import adaptive_train_loop, AnalyticsEngine
    
    # Initialize tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.get("lm_path", "AnotherSamWilson/auto-ui-base"),
        trust_remote_code=True
    )
    
    # Initialize agent
    print("\nInitializing agent...")
    agent = AutoUIAgent(
        device=f"cuda:{cfg.get('main_device', 0)}",
        accelerator=accelerator,
        policy_lm=cfg.get("lm_path", "AnotherSamWilson/auto-ui-base"),
        critic_lm=cfg.get("critic_lm_path", "roberta-base"),
        cache_dir=cfg.get("cache_dir", None),
        use_lora=cfg.get("use_lora", False),
        lora_path=cfg.get("lora_path", None)
    )
    
    # Initialize environment (only if doing online training)
    env = None
    if cfg.train_mode != "offline":
        print("\nInitializing environment...")
        
        # Create evaluators
        evaluators = [
            EndResultEvaluator(task_set=cfg.task_set)
            for _ in range(cfg.bsize)
        ]
        
        env = BatchedAndroidEnv(
            avd_name=cfg.avd_name,
            cache_avd_names=[f"{cfg.avd_name}_{i}" for i in range(cfg.bsize)],
            udids=[f"emulator-{5554 + i*2}" for i in range(cfg.bsize)],
            appium_base_port=cfg.get("appium_base_port", 4723),
            android_avd_home=cfg.android_avd_home,
            emulator_path=cfg.emulator_path,
            adb_path=cfg.adb_path,
            run_headless=cfg.get("run_headless", True),
            max_steps=cfg.max_steps,
            evaluators=evaluators,
            all_tasks=all_tasks,
            task_split=cfg.task_split,
            temp_path=cfg.get("temp_path", "/tmp/digirl_images"),
            save_images=cfg.get("save_images", False),
            record=cfg.get("record", False)
        )
    
    # Create save directory
    save_path = cfg.save_path
    os.makedirs(save_path, exist_ok=True)
    print(f"\nSave path: {save_path}")
    
    # Run ADAT training loop
    print("\nStarting ADAT training...")
    adaptive_train_loop(
        env=env,
        agent=agent,
        tokenizer=tokenizer,
        accelerator=accelerator,
        # Standard DigiRL params
        warmup_iter=cfg.get("warmup_iter", 20),
        rollout_size=cfg.get("rollout_size", 16),
        batch_size=cfg.get("batch_size", 2),
        capacity=cfg.get("capacity", 500000),
        train_iterations=cfg.get("train_iterations", 600),
        epochs=cfg.get("epochs", 3),
        grad_accum_steps=cfg.get("grad_accum_steps", 4),
        critic_lr=cfg.get("critic_lr", 1e-4),
        lm_lr=cfg.get("lm_lr", 1e-4),
        gamma=cfg.get("gamma", 0.5),
        tau=cfg.get("tau", 0.1),
        use_wandb=cfg.get("use_wandb", False),
        actor_epochs=cfg.get("actor_epochs", 20),
        train_mode=cfg.train_mode,
        max_grad_norm=cfg.get("max_grad_norm", 0.01),
        save_path=save_path,
        save_freq=cfg.get("save_freq", 25),
        train_algorithm=cfg.get("train_algorithm", "digirl"),
        offline_data_path=cfg.get("offline_data_path", None),
        offline_actor_iterations=cfg.get("offline_actor_iterations", 20),
        offline_critic_iterations=cfg.get("offline_critic_iterations", 20),
        offline_trajectory_critic_iterations=cfg.get("offline_trajectory_critic_iterations", 20),
        trajectory_critic_epochs=cfg.get("trajectory_critic_epochs", 5),
        # ADAT params
        use_adaptive_sampling=cfg.get("use_adaptive_sampling", True),
        use_trajectory_weighting=cfg.get("use_trajectory_weighting", True),
        use_curriculum=cfg.get("use_curriculum", True),
        use_hyperparameter_scheduling=cfg.get("use_hyperparameter_scheduling", True),
        sampling_strategy=cfg.get("sampling_strategy", "zpd"),
        curriculum_warmup=cfg.get("curriculum_warmup", 50),
        weight_alpha=cfg.get("weight_alpha", 0.3),
        weight_beta=cfg.get("weight_beta", 0.5),
        weight_gamma=cfg.get("weight_gamma", 0.2),
        target_success_rate=cfg.get("target_success_rate", 0.5)
    )
    
    # Cleanup
    if env is not None:
        env.close()
    
    if cfg.use_wandb and accelerator.is_main_process:
        wandb.finish()
    
    print("\n" + "=" * 60)
    print("ADAT Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
