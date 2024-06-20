import transformers
from tqdm import tqdm
from digirl.environment import BatchedAndroidEnv
from digirl.models import AutoUIAgent, CogAgent
from digirl.algorithms import offpolicy_train_loop, eval_loop, worker_collect_loop
from digirl.misc import colorful_print
from digirl.environment.android import EndResultEvaluator
from digirl.environment.android import autoui_translate_action, cogagent_translate_action
import torch.nn as nn
import numpy as np 
import wandb
from omegaconf import DictConfig, OmegaConf
import os
import hydra
from accelerate import Accelerator
from datetime import timedelta
from accelerate import DistributedDataParallelKwargs, InitProcessGroupKwargs
transformers.logging.set_verbosity_error()

import torch.distributed as dist
import datetime

def load_task_file(assets_path, task_set, task_split):
    all_tasks = []
    with open(os.path.join(assets_path, task_set + "_" + task_split + ".txt")) as fb: 
        for line in fb:
            all_tasks.append(line.replace("\n", ""))
    return all_tasks

@hydra.main(version_base=None, config_path=None, config_name=None)
def main(config: "DictConfig"):
    colorful_print(OmegaConf.to_yaml(config), fg='red')
    try:
        from huggingface_hub import login
        login(token=config.huggingface_token)
    except:
        print(">>> Huggingface token not found.")

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(InitProcessGroupKwargs(timeout=timedelta(minutes=40)), kwargs_handlers=[ddp_kwargs], project_dir = config.save_path)
    device = accelerator.device
    env = None
    if accelerator.is_main_process:
        # load environment
        all_tasks = load_task_file(config.assets_path, config.task_set, config.task_split)
        bsize = config.bsize
        base_port = 5554
        evaluators = [EndResultEvaluator(config.gemini_key, config.task_set)] * bsize
        assert len(evaluators) == bsize
        if config.agent_name == "autoui":
            translate_action = autoui_translate_action
            use_feature_extractor = True
        elif config.agent_name == "cogagent":
            translate_action = cogagent_translate_action
            use_feature_extractor = False
    decode_f = lambda x:x
    if config.task_mode != "evaluate":
        assert config.agent_name == "autoui", "Only AutoUI agent is supported for training"
        colorful_print(">>> Agent: AutoUI", fg='blue')
        colorful_print(">>> Training algorithm: "+config.train_algorithm, fg='blue')
        colorful_print(">>> Training mode: "+config.train_mode, fg='blue')
    else:
        colorful_print(">>> Agent: "+config.agent_name, fg='blue')
        colorful_print(">>> Evauation mode", fg='blue')
    
    if config.agent_name == "autoui":
        agent = AutoUIAgent(device=device, accelerator=accelerator, 
                            temperature=config.temperature, do_sample=config.do_sample, 
                            policy_lm=config.policy_lm, critic_lm=config.critic_lm,
                            cache_dir=config.cache_dir, max_new_tokens=config.max_new_tokens)
        tokenizer = agent.tokenizer
    elif config.agent_name == "cogagent":
        agent = CogAgent(url=config.cogagent_url)
        tokenizer = None
    else:
        raise NotImplementedError("Only AutoUI agent is supported for now")

    if config.use_wandb and accelerator.is_main_process:
        wandb.login(key=config.wandb_key)
        wandb.init(project=config.project_name, entity=config.entity_name, name=config.run_name, config=dict(config))

    # this bunch of code should handle these functions:
    # |-- autoui
    #   |-- online train (eval in the end)
    #   |-- offline train (eval in the end)
    #   |-- off2on train (eval in the end)
    #   |-- eval-only
    # |-- cogagent (eval only)
    # |-- set-of-marks (eval only)
    # |-- appagent (eval only)

    def construct_env(sample_mode):
        env = BatchedAndroidEnv(avd_name="test_Android", 
            cache_avd_names=[f"test{i}" for i in range(1,1+bsize)], 
            android_avd_home=config.android_avd_home,
            emulator_path=config.emulator_path, 
            adb_path=config.adb_path, 
            udids = [f"emulator-{base_port+2*i}" for i in range(bsize)],
            max_steps=config.max_steps-1, # will have 1 dangling step after stop signal is triggered
            appium_base_port = base_port+1098,
            run_headless=True, 
            use_feature_extractor=use_feature_extractor, 
            device=accelerator.device,
            translate_action=translate_action,
            evaluators=evaluators,
            temp_path = os.path.join(config.save_path, "images"),
            save_images=True,
            all_tasks=all_tasks,
            task_split=config.task_split,
            sample_mode=sample_mode,
            record=config.record,
        )
        return env

    # autoui will be trained first then evaluated
    if config.parallel in ["single", "host"]:
        if config.agent_name == "cogagent" or config.task_mode == "evaluate":
            if accelerator.is_main_process:
                env = construct_env(sample_mode=config.eval_sample_mode)
            eval_loop(env = env,
                        tokenizer=tokenizer,
                        agent = agent,
                        accelerator = accelerator,
                        decode_f=decode_f,
                        **config)
        elif config.agent_name == "autoui":
            if accelerator.is_main_process:
                env = construct_env(sample_mode="random")
            offpolicy_train_loop(env = env,
                    tokenizer=tokenizer,
                    agent = agent,
                    accelerator = accelerator,
                    decode_f=decode_f,
                    **config)
                
            # always do eval after training (unless this is only a worker machine to collect trajectories)
            if accelerator.is_main_process:
                env = construct_env(sample_mode=config.eval_sample_mode)
            eval_loop(env = env,
                        tokenizer=tokenizer,
                        agent = agent,
                        accelerator = accelerator,
                        decode_f=decode_f,
                        **config)

    elif config.parallel == "worker":
        if accelerator.is_main_process:
            env = construct_env(sample_mode="random")
        worker_collect_loop(env = env,
                            agent = agent,
                            tokenizer=tokenizer,
                            accelerator = accelerator,
                            decode_f=decode_f,
                            **config)

if __name__ == "__main__":
    main()
