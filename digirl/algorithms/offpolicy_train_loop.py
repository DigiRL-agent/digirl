from digirl.environment import batch_interact_environment
from digirl.data import ReplayBuffer
import numpy as np
from tqdm import tqdm
from digirl.algorithms.digirl import DigiRLTrainer
from digirl.algorithms.filteredbc import BCTrainer
from digirl.misc import colorful_print
import wandb
import os
import torch
import time
import copy
from digirl.environment.env_utils import add_mc_return
from digirl.algorithms.parallel_utils import remote_collect_trajectories

def label_trajectories(trajectories, agent):
    print("Labeling Trajectories")
    baselines = []
    for i in range(0, len(trajectories), 16):
        observations = [t[0]["observation"] for t in trajectories[i:i+16]]
        with torch.no_grad():
            v = agent.trajectory_critic(observations)
            v = torch.nn.Softmax(dim = -1)(v)[:,1]
            baselines.append(v.flatten())
    baselines = torch.cat(baselines, dim = -1)
    print("Done Labeling Trajectories")
    return torch.clamp(baselines.cpu(), 1e-4, 1-1e-4)

def framestack(all_trajectories):
    new_trajectories = copy.deepcopy(all_trajectories)
    for trajectory, new_trajectory in zip(all_trajectories, new_trajectories):
        for i,(t, nt) in enumerate(zip(trajectory, new_trajectory)):
            if i  == 0:
                nt["image_features"] = np.concatenate([t["image_features"], t["image_features"]], axis = -1)
            else:
                nt["image_features"] = np.concatenate([trajectory[i-1]["image_features"], t["image_features"]], axis = -1)
            nt["next_image_features"] = np.concatenate([t["image_features"], t["next_image_features"]], axis = -1)
    return new_trajectories

def filterbc_buffer(all_trajectories, batch_size, capacity, agent):
    trajectory_rewards = np.array([t[0]["trajectory_reward"] if len(t) > 0 else 0 for t in all_trajectories]).flatten()
    cutoff = np.quantile(trajectory_rewards, 1 - 0.1)
    top10 = np.argsort(trajectory_rewards)[-10:]
    print("Top 10 Trajectories: ")
    for i in top10:
        print(all_trajectories[i][0]["observation"])
        print(trajectory_rewards[i])
    print("Cutoff: ", cutoff)
    filtered_trajectories = []
    for t, b in zip(all_trajectories, trajectory_rewards):
        if b >= cutoff:
            filtered_trajectories.append(t)
    data = sum(filtered_trajectories, [])
    filtered_buffer= ReplayBuffer(batch_size= batch_size, capacity=capacity)
    for d in data:
        filtered_buffer.insert(**d)
    return filtered_buffer


def filter_buffer(all_trajectories, batch_size, capacity, agent):
    baselines = label_trajectories(all_trajectories, agent).numpy().flatten()
    trajectory_rewards = np.array([t[0]["trajectory_reward"] if len(t) > 0 else 0 for t in all_trajectories]).flatten()
    baselines = trajectory_rewards - baselines
    cutoff = np.quantile(baselines, 1 - 0.1)
    top10 = np.argsort(baselines)[-10:]
    print("Top 10 Trajectories: ")
    for i in top10:
        print(all_trajectories[i][0]["observation"])
        print(baselines[i])
    print("Cutoff: ", cutoff)
    filtered_trajectories = []
    for t, b in zip(all_trajectories, baselines):
        if b >= cutoff:
            filtered_trajectories.append(t)
    data = sum(filtered_trajectories, [])
    filtered_buffer= ReplayBuffer(batch_size= batch_size, capacity=capacity)
    for d in data:
        filtered_buffer.insert(**d)
    return filtered_buffer

def offpolicy_train_loop(env,\
                agent,\
                tokenizer,\
                accelerator,\
                warmup_iter: int = 20,
                rollout_size: int = 50,\
                batch_size: int = 2,
                capacity: int = 500000,
                train_iterations: int = 10,\
                epochs:int = 3, \
                grad_accum_steps: int = 1,\
                critic_lr: float= 1e-3,\
                lm_lr: float = 1e-5,\
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
    
    # prepare the model
    agent.prepare()
    # prepare the optimizers
    trainer.prepare()

    loaded_trajs = False
    
    # off-to-on
    # no offline ckpt, no online ckpt -> offline training
    # offline ckpt, no online ckpt -> online training
    # offline ckpt, online ckpt -> resume online training
    
    # offline
    # no resume supported
    
    # online
    # no online ckpt -> online training
    # online ckpt -> resume online training
    
    # omit this for online training
    if offline_data_path is not None and train_mode != "online":
        all_trajectories = torch.load(offline_data_path)
        all_trajectories = framestack(all_trajectories)
        print(f"The number of offline trajectories is {len(all_trajectories)}")
        all_trajectories = [add_mc_return(t, gamma=gamma) for t in all_trajectories]
        train_trajectories = all_trajectories[:int(len(all_trajectories)*0.8)]
        val_trajectories = all_trajectories[int(len(all_trajectories)*0.8):]
        loaded_trajs = 'scratch'
        
    # resume training from the saved checkpoint
    if os.path.exists(os.path.join(save_path, 'trainer.pt')):
        assert train_mode != "offline", "Only online/off2on training can be resumed"
        trainer.load(os.path.join(save_path, 'trainer.pt'))
        replay_buffer = torch.load(os.path.join(save_path, 'replay_buffer.pt'))
        all_trajectories = torch.load(os.path.join(save_path, 'trajectories.pt'))
        train_trajectories = torch.load(os.path.join(save_path, 'train_trajectories.pt'))
        val_trajectories = torch.load(os.path.join(save_path, 'val_trajectories.pt'))
        print(f"The number of online trajectories is {len(all_trajectories)}")
        if use_wandb and accelerator.is_main_process:
            print("Loading from checkpoint")
        loaded_trajs = 'resume'
            
    if not loaded_trajs:
        train_trajectories = []
        val_trajectories = []
        all_trajectories = []

    replay_buffer = ReplayBuffer(batch_size= batch_size, capacity=capacity)
    validation_buffer = ReplayBuffer(batch_size= batch_size, capacity=capacity)

    data = sum(train_trajectories, [])
    val_data = sum(val_trajectories, [])
    for d in data:
        replay_buffer.insert(**d)
    for d in val_data:
        validation_buffer.insert(**d)
    # offline training
    if not os.path.exists(os.path.join(save_path, 'trainer.pt')):
        #if nothing in the trainer only the offline trainer is saved
        if os.path.exists(os.path.join(save_path, 'trainer_offline.pt')):
            trainer.load(os.path.join(save_path, 'trainer_offline.pt'))
            print("Loading from offline trainer")
        else:
            if offline_data_path is not None and train_mode != "online":
                print(">>>Training Offline")
                info = {}
                # offline training will never use the trajectory-level critic filter, so please use filterbc_buffer
                filtered_buffer = filterbc_buffer(train_trajectories, batch_size, capacity, agent)
                filtered_validation_buffer = filterbc_buffer(val_trajectories, batch_size, capacity, agent)
                
                if train_algorithm == "filteredbc":
                    # filtered BC training phase
                    for i in tqdm(range(offline_actor_iterations), disable=not accelerator.is_main_process):
                        info.update(trainer.update(filtered_buffer))
                        if use_wandb and accelerator.is_main_process:
                            wandb.log(info)
                elif train_algorithm == "digirl":
                    # digirl training phase
                    for i in tqdm(range(offline_trajectory_critic_iterations), disable=not accelerator.is_main_process):
                        info.update(trainer.update_trajectory_critic(train_trajectories, val_trajectories))
                        if use_wandb and accelerator.is_main_process:
                            wandb.log(info)
                    for i in tqdm(range(offline_critic_iterations), disable=not accelerator.is_main_process):
                        info.update(trainer.update_critic(replay_buffer, validation_buffer))
                        if use_wandb and accelerator.is_main_process:
                            wandb.log(info)

                    print(">>>Training Policy")
                    for i in tqdm(range(offline_actor_iterations), disable=not accelerator.is_main_process):
                        info.update(trainer.update_policy(filtered_buffer, filtered_validation_buffer))
                        if use_wandb and accelerator.is_main_process:
                            wandb.log(info)
                if accelerator.is_main_process:
                    trainer.save(os.path.join(save_path, 'trainer_offline.pt'))

    if accelerator.is_main_process:
        print(">>>start iterations")
    if loaded_trajs == "resume":
        resume_iter = len(all_trajectories) // rollout_size
    else:
        resume_iter = 0
    
    progress_bar = tqdm(total=train_iterations, initial=resume_iter)
    
    for i in range(resume_iter, train_iterations):
        assert train_mode != "offline", "Only online/off2on need to iteractively train; offline should directly go to eval loop after training"
        if parallel == 'single':
            trajectories = batch_interact_environment(agent = agent,\
                                                env = env,\
                                                num_trajectories= rollout_size,\
                                                accelerator = accelerator,\
                                                use_tqdm=False,
                                                decode_f = decode_f,
                                                gamma = gamma,
                                                iter=i)
        elif parallel == 'host':
            if i == 0:
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
            trajectories = remote_collect_trajectories(save_path=save_path, 
                                                       worker_temp_path=worker_temp_path, 
                                                       worker_run_path=worker_run_path,
                                                       worker_ips=worker_ips, 
                                                       worker_username=worker_username,
                                                       trainer=trainer)
        
        trajectories = framestack(trajectories)
        if accelerator.is_main_process:
            info = {"iteration": i,\
                    "rollout.mean": np.mean([d[0]["trajectory_reward"] if len(d) > 0 else 0 for d in trajectories]),\
                    "rollout.max": np.max([d[0]["trajectory_reward"] if len(d) > 0 else 0 for d in trajectories]),\
                    "rollout.min": np.min([d[0]["trajectory_reward"] if len(d) > 0 else 0 for d in trajectories]),\
                    "walltime": time.time()}
            all_trajectories += trajectories
            colorful_print(f">>> length of all_trajectories: {len(trajectories)}", fg='green')
            new_train_trajectories = trajectories[:int(len(trajectories)*0.8)]
            new_val_trajectories = trajectories[int(len(trajectories)*0.8):]
            train_trajectories += new_train_trajectories
            val_trajectories += new_val_trajectories
            data = sum(new_train_trajectories, [])
            val_data = sum(new_val_trajectories, [])
            for d in data:
                replay_buffer.insert(**d)
            for d in val_data:
                validation_buffer.insert(**d)
        
            info.update({"rollout.reward.mean": np.mean([d["reward"] for d in data]),\
                    "rollout.reward.max": np.max([d["reward"] for d in data]),\
                    "rollout.reward.min": np.min([d["reward"] for d in data])})
            print(">>> Saving Replay Buffer")
            torch.save(replay_buffer, os.path.join(save_path, 'replay_buffer.pt'))
            torch.save(all_trajectories, os.path.join(save_path, 'trajectories.pt'))
            torch.save(train_trajectories, os.path.join(save_path, 'train_trajectories.pt'))
            torch.save(val_trajectories, os.path.join(save_path, 'val_trajectories.pt'))
            print(">>> Saved Replay Buffer")
            time.sleep(15)
        else:
            info = {}
        accelerator.wait_for_everyone()
        
        train_trajectories = torch.load(os.path.join(save_path, 'train_trajectories.pt'))
        val_trajectories = torch.load(os.path.join(save_path, 'val_trajectories.pt'))
        all_trajectories = torch.load(os.path.join(save_path, 'trajectories.pt'))
        replay_buffer = torch.load(os.path.join(save_path, 'replay_buffer.pt'))

        assert train_algorithm in ['digirl', 'filteredbc'], "Only digirl and filteredbc are supported"
        if train_algorithm == "filteredbc":
            filtered_buffer = filterbc_buffer(train_trajectories, batch_size, capacity, agent)
            filtered_validation_buffer = filterbc_buffer(val_trajectories, batch_size, capacity, agent)
        elif train_algorithm == 'digirl':
            filtered_buffer = filter_buffer(train_trajectories, batch_size, capacity, agent)
            filtered_validation_buffer = filter_buffer(val_trajectories, batch_size, capacity, agent)
        
        print("Training")
        if 'filtered' in train_algorithm:
            info.update(trainer.update(filtered_buffer, no_update_actor = (i < warmup_iter)))
            del filtered_buffer
        else:
            info.update(trainer.update_trajectory_critic(train_trajectories, val_trajectories))
            info.update(trainer.update(replay_buffer, validation_buffer, filtered_buffer, filtered_validation_buffer, no_update_actor = (i < warmup_iter)))
    
        if use_wandb and accelerator.is_main_process:
            wandb.log(info)
        if (i+1) % save_freq == 0 and save_path is not None and accelerator.is_main_process:
            print("Saving")
            trainer.save(os.path.join(save_path, 'trainer.pt'))
            torch.save(replay_buffer, os.path.join(save_path, 'replay_buffer.pt'))
            
        if accelerator.is_main_process:
            progress_bar.update(1)
        
