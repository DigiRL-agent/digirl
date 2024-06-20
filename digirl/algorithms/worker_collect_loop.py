from digirl.environment import batch_interact_environment
from digirl.data import ReplayBuffer
from digirl.algorithms.digirl import DigiRLTrainer
from digirl.algorithms.filteredbc import BCTrainer
from digirl.misc import colorful_print
import os
import torch

def worker_collect_loop(env,\
                agent,\
                tokenizer,\
                accelerator,\
                warmup_iter: int = 20,
                rollout_size: int = 50,\
                batch_size: int = 2,
                capacity: int = 500000,
                train_iterations: int = 1,\
                epochs:int = 3, \
                grad_accum_steps: int = 1,\
                do_sample: bool = False,\
                temperature: float = 2.0,\
                critic_lr: float= 1e-3,\
                lm_lr: float = 1e-5,\
                gamma: float = 0.9,
                tau: float = 0.1,
                use_wandb: bool = False,
                env_load_path: str = '',
                actor_epochs: int = 3,
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
                **kwargs):
    if train_algorithm == "digirl":
        trainer = DigiRLTrainer(agent=agent,\
                            accelerator=accelerator,\
                                tokenizer=tokenizer,\
                                critic_lr = critic_lr,\
                                lm_lr = lm_lr,\
                                gamma = gamma,\
                                tau = tau,\
                                epochs = epochs,\
                                actor_epochs = actor_epochs,
                                grad_accum_steps=grad_accum_steps,
                                max_grad_norm=max_grad_norm,
                                trajectory_critic_epochs = trajectory_critic_epochs)
    elif train_algorithm == "filteredbc":
        trainer = BCTrainer(agent=agent,\
                                tokenizer=tokenizer,\
                                accelerator=accelerator,
                                lm_lr = lm_lr,\
                                epochs = actor_epochs,\
                                grad_accum_steps=grad_accum_steps,
                                max_grad_norm=max_grad_norm)
    replay_buffer= ReplayBuffer(batch_size= batch_size, capacity=capacity)
    all_trajectories = []
    #prepare the model and optimizers
    agent.prepare()
    trainer.prepare()

    colorful_print(">>> Loading Current Trainer from Host", fg='blue')
    trainer.load(os.path.join(save_path, 'trainer_current.pt'))

    colorful_print(">>> Worker Collecting Online Data", fg='blue')
    
    for i in range(train_iterations):
        trajectories = batch_interact_environment(agent = agent,\
                                            env = env,\
                                            num_trajectories= rollout_size,\
                                            accelerator = accelerator,\
                                            use_tqdm=False,
                                            decode_f = decode_f,
                                            gamma = gamma,
                                            iter=i)

        torch.save(trajectories, os.path.join(save_path, 'trajectories.pt'))

            