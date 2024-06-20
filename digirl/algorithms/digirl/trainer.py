import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from digirl.data import DummyDataset
import random

def dict_mean(dict_list):
    mean_dict = {}
    if len(dict_list) > 0:
        for key in dict_list[0].keys():
            if "min" in key:
                mean_dict[key] = min(d[key] for d in dict_list)
            elif "max" in key:
                mean_dict[key] = max(d[key] for d in dict_list)
            else:
                mean_dict[key] = sum(d[key] for d in dict_list) / len(dict_list)
    return mean_dict

class DigiRLTrainer():
    def __init__(self, agent,\
                 accelerator,\
                    tokenizer,\
                    critic_lr: float = 1e-3,\
                    lm_lr: float = 1e-5,\
                    grad_accum_steps: int = 8,\
                    gamma: float = 0.9,
                    tau: float = 0.1,
                    epochs: int = 3,
                    max_grad_norm: float=0.01,
                    actor_epochs: int = 3,
                    trajectory_critic_epochs: int = 3,):
        """
        beta: coefficient for the bc loss
        """
        super().__init__()
        self.agent = agent
        self.tokenizer = tokenizer
        self.lm_optimizer = torch.optim.Adam(agent.model.parameters(), lr = lm_lr)
        self.critic_optimizer = torch.optim.Adam(agent.critic.parameters(), lr = critic_lr)
        self.trajectory_critic_optimizer = torch.optim.Adam(agent.trajectory_critic.parameters(), lr = critic_lr)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.grad_accum_steps = grad_accum_steps
        self.actor_epochs = actor_epochs
        self.gamma = gamma
        self.epochs = epochs
        self.trajectory_critic_epochs = trajectory_critic_epochs
        self.step = 0
        self.tau = tau
        self.max_grad_norm = max_grad_norm
        self.accelerator = accelerator
        self.softmax = torch.nn.Softmax(dim = -1)

    def prepare(self):
        self.lm_optimizer = self.accelerator.prepare(self.lm_optimizer)
        self.critic_optimizer = self.accelerator.prepare(self.critic_optimizer)
        self.trajectory_critic_optimizer = self.accelerator.prepare(self.trajectory_critic_optimizer)
    
    def trajectory_critic_loss(self, observation, mc_return, validation = False, **kwargs):
        with torch.autograd.set_detect_anomaly(True):
            mc_return = torch.Tensor(mc_return).to(self.accelerator.unwrap_model(self.agent.model).device, dtype = self.accelerator.unwrap_model(self.agent.model).dtype).flatten()
            v = self.agent.trajectory_critic(observation, detach_model=False)
            regression_target = (mc_return > 0).long()
            v_loss = self.criterion(v, regression_target)
            v_acc = (v.argmax(dim = 1) == regression_target).float().mean()
            if not validation:
                self.accelerator.backward(v_loss)
            v_loss = v_loss.detach().cpu()
            v_acc = v_acc.detach().cpu()
            mc_return = mc_return.detach().cpu()
            v = self.softmax(v)[:, 1]
        info = {"trajectory.v1.loss": v_loss,\
                "trajectory.v1.acc": v_acc,\
                "trajectory.v1.mean": torch.mean(v),\
                "trajectory.v1.min": torch.min(v),\
                "trajectory.v1.max": torch.max(v),\
                "trajectory.v1.std": torch.std(v),\
                "mc_return.mean": torch.mean(mc_return),
                "mc_return.max": torch.max(mc_return),
                "mc_return.min": torch.min(mc_return),
                "mc_return.std": torch.std(mc_return),
                }
        if validation:
            validation_info = {}
            for k,v in info.items():
                validation_info["validation."+k] = v
            return validation_info
        return info

    def critic_loss(self, observation, image_features, action, reward, next_observation, next_image_features,done, mc_return,
                    validation = False, **kwargs):
        reward = torch.Tensor(reward).to(self.accelerator.unwrap_model(self.agent.model).device, dtype = self.accelerator.unwrap_model(self.agent.model).dtype).flatten()
        done = torch.Tensor(done).to(self.accelerator.unwrap_model(self.agent.model).device, dtype = self.accelerator.unwrap_model(self.agent.model).dtype).flatten()
        mc_return = torch.Tensor(mc_return).to(self.accelerator.unwrap_model(self.agent.model).device, dtype = self.accelerator.unwrap_model(self.agent.model).dtype).flatten()
        v1, v2 = self.agent.critic(observation, image_features, action, detach_model=False)
        nv1, nv2 = self.agent.critic(next_observation, next_image_features, action, detach_model=False)

        v1 = v1.reshape(-1, 2)
        v2 = v2.reshape(-1, 2)
        nv1 = nv1.reshape(-1, 2)
        nv2 = nv2.reshape(-1, 2)
        regression_target = (mc_return > 0).long()
        v1_loss = self.criterion(v1, regression_target)
        v1_acc = (v1.argmax(dim = 1) == regression_target).float().mean()
        v2_loss = self.criterion(v2, regression_target)
        v2_acc = (v2.argmax(dim = 1) == regression_target).float().mean()
        nv1_loss = self.criterion(nv1, regression_target)
        nv2_loss = self.criterion(nv2, regression_target)
        if not validation:
            self.accelerator.backward(v1_loss + v2_loss + nv1_loss + nv2_loss)
        v1_loss, v2_loss = v1_loss.detach().cpu(), v2_loss.detach().cpu()
        v1_acc, v2_acc = v1_acc.detach().cpu(), v2_acc.detach().cpu()

        #calculate the probability for logging purpose
        v1 = self.softmax(v1)[:, 1]
        v2 = self.softmax(v2)[:, 1]
        info = {"v1.loss": v1_loss,\
                "v2.loss": v2_loss,\
                "v1.acc": v1_acc,\
                "v2.acc": v2_acc,\
                "v1.mean": torch.mean(v1),\
                "v1.min": torch.min(v1),\
                "v1.max": torch.max(v1),\
                "v1.std": torch.std(v1),
                "v2.mean": torch.mean(v2),
                "v2.max": torch.max(v2),
                "v2.min": torch.min(v2),
                "v2.std": torch.std(v2),
                }
        if validation:
            validation_info = {}
            for k,v in info.items():
                validation_info["validation."+k] = v
            return validation_info
        return info

    def actor_loss(self, observation, action, image_features, next_observation, next_image_features, mc_return, pi_action, advantage, reward,
                   validation=False,**kwargs):
        mc_return = torch.Tensor(mc_return).to(self.accelerator.unwrap_model(self.agent.model).device, dtype = self.accelerator.unwrap_model(self.agent.model).dtype).flatten()
        reward = torch.Tensor(reward).to(self.accelerator.unwrap_model(self.agent.model).device, dtype = self.accelerator.unwrap_model(self.agent.model).dtype).flatten()
        with torch.no_grad():
            v1, v2 = self.agent.critic(observation, image_features, action, detach_model=False)
            nv1, nv2 = self.agent.critic(next_observation, next_image_features, action, detach_model=False)
        v1 = self.softmax(v1)[:, 1]
        v2 = self.softmax(v2)[:, 1]
        nv1 = self.softmax(nv1)[:, 1]
        nv2 = self.softmax(nv2)[:, 1]
        v = torch.minimum(v1, v2).flatten()
        nv = torch.minimum(nv1, nv2).flatten()
        #TODO: set +1 so that the advantage is always positive
        advantage = nv - v - 0.05 + reward + mc_return
        advantage = torch.clamp(advantage, 0, 1)
        advantage = (advantage > 0).to(dtype = self.accelerator.unwrap_model(self.agent.model).dtype)
        image_features = image_features.to(self.agent.device)
        log_prob = self.agent.get_log_prob(observation, image_features, action).sum(dim = 1).flatten()
        advantage = torch.Tensor(advantage).to(self.accelerator.unwrap_model(self.agent.model).device, dtype = self.accelerator.unwrap_model(self.agent.model).dtype)
        advantages = advantage.flatten()
        values = torch.zeros_like(advantages)
        residual_advantage = torch.zeros_like(advantages)
        pg_loss = -torch.mean(log_prob.flatten()*advantages)
        value_loss = torch.zeros_like(pg_loss)
        if not validation:
            self.accelerator.backward(pg_loss+value_loss)
        advantages = advantages.detach().cpu()
        info =  {"pg.loss": pg_loss.detach().cpu().item(),
                "values.loss": value_loss.detach().cpu().item(),
                "values.mean": values.mean(),
                "values.max": torch.max(values),
                "values.min": torch.min(values),
                "values.std": torch.std(values),
                "advantages.mean": advantages.mean(),
                "advantages.max": torch.max(advantages),
                "advantages.min": torch.min(advantages),
                "advantages.std": torch.std(advantages),
                "residual_advantages.mean": residual_advantage.mean(),
                "residual_advantages.max": torch.max(residual_advantage),
                "residual_advantages.min": torch.min(residual_advantage),
                "residual_advantages.std": torch.std(residual_advantage),}
        if validation:
            validation_info = {}
            for k,v in info.items():
                validation_info["validation."+k] = v
            return validation_info
        return info

    def update_trajectory_critic(self, trajectories, validation_trajectories = None):
        info = {}
        info_list = []
        batch_size = 8
        with torch.autograd.set_detect_anomaly(True):
            for _ in tqdm(range(self.trajectory_critic_epochs), disable= not self.accelerator.is_main_process):
                data = [{"observation": traj[0]["observation"], "mc_return": traj[-1]["mc_return"]} for traj in trajectories]
                data = [random.sample(data, 1)[0] for _ in range(self.grad_accum_steps*batch_size)]
                dataloader = DataLoader(DummyDataset(data), batch_size=batch_size)
                dataloader = self.accelerator.prepare(dataloader)
                self.trajectory_critic_optimizer.zero_grad()
                for batch in tqdm(dataloader, disable=True):
                    info_list.append(self.trajectory_critic_loss(**batch))
                self.accelerator.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.trajectory_critic_optimizer.step()
        info.update(dict_mean(info_list))
        if validation_trajectories is not None:
            info_list = []
            data = [{"observation": traj[0]["observation"], "mc_return": traj[-1]["mc_return"]} for traj in validation_trajectories]
            data = [random.sample(data, 1)[0] for _ in range(self.grad_accum_steps*batch_size)]
            dataloader = DataLoader(DummyDataset(data), batch_size=batch_size)
            dataloader = self.accelerator.prepare(dataloader)
            with torch.no_grad():
                for batch in tqdm(dataloader, disable=True):
                    info_list.append(self.trajectory_critic_loss(validation=True, **batch))
            info.update(dict_mean(info_list))
        return info

    def update_critic(self, replay_buffer, validation_buffer = None):
        self.step += 1
        info = {}
        info_list = []
        with torch.autograd.set_detect_anomaly(True):
            for _ in tqdm(range(self.epochs), disable= not self.accelerator.is_main_process):
                data = [replay_buffer.sample(1) for _ in range(self.grad_accum_steps*replay_buffer.batch_size)]
                for d in data:
                    for k,v in d.items():
                        d[k] = v[0]
                dataloader = DataLoader(DummyDataset(data), batch_size=replay_buffer.batch_size)
                dataloader = self.accelerator.prepare(dataloader)
                self.critic_optimizer.zero_grad()
                for batch in tqdm(dataloader, disable=True):
                    info_list.append(self.critic_loss(**batch))
                self.accelerator.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
        info.update(dict_mean(info_list))
        if validation_buffer is not None:
            info_list = []
            data = [validation_buffer.sample(1) for _ in range(self.grad_accum_steps*replay_buffer.batch_size)]
            for d in data:
                for k,v in d.items():
                    d[k] = v[0]
            dataloader = DataLoader(DummyDataset(data), batch_size=replay_buffer.batch_size)
            dataloader = self.accelerator.prepare(dataloader)
            with torch.no_grad():
                for batch in tqdm(dataloader, disable=True):
                    info_list.append(self.critic_loss(validation=True, **batch))
            info.update(dict_mean(info_list))
        return info
        
        
    def update_policy(self, replay_buffer, validation_buffer = None, no_update_actor=False):
        self.step += 1
        info = {}
        info_list = []
        action_bsize = 2 if 'mistral' in self.agent.policy_lm else replay_buffer.batch_size
        #update actor
        if not no_update_actor:
            print(">>>updating actor")
            #batchsize for the actor set to 1 for mistral due to memory concern
            # action_bsize = 2 if 'mistral' in self.agent.policy_lm else replay_buffer.batch_size
            #action_bsize = replay_buffer.batch_size
            for _ in tqdm(range(self.actor_epochs), disable= not self.accelerator.is_main_process):
                data = [replay_buffer.sample(1) for _ in range(self.grad_accum_steps*replay_buffer.batch_size)]
                grad_index = 0
                for d in data:
                    for k,v in d.items():
                        d[k] = v[0]
                dataloader = DataLoader(DummyDataset(data), batch_size=action_bsize, shuffle=False)
                all_pi_actions = []
                all_advantages = []
                # import IPython; IPython.embed()
                dataloader = self.accelerator.prepare(dataloader)
                self.lm_optimizer.zero_grad()
                for batch in dataloader:
                    pi_action = None
                    advantages = None
                    info_list.append(self.actor_loss(**batch, pi_action=pi_action, advantage=advantages))
                self.accelerator.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.lm_optimizer.step()
        info.update(dict_mean(info_list))
        if validation_buffer is not None:
            info_list = []
            data = [validation_buffer.sample(1) for _ in range(self.grad_accum_steps*replay_buffer.batch_size)]
            for d in data:
                for k,v in d.items():
                    d[k] = v[0]
            dataloader = DataLoader(DummyDataset(data), batch_size=action_bsize, shuffle=False)
            dataloader = self.accelerator.prepare(dataloader)
            with torch.no_grad():
                for batch in tqdm(dataloader, disable=True):
                    info_list.append(self.actor_loss(validation=True, pi_action=None, advantage=None, **batch))
            info.update(dict_mean(info_list))
            return info
        return info

    def update(self, replay_buffer, validation_buffer = None, filtered_buffer = None, filtered_validation_buffer = None,no_update_actor=False):
        if filtered_validation_buffer is None:
            filtered_validation_buffer = validation_buffer
        if filtered_buffer is None:
            filtered_buffer = replay_buffer
        info = {}
        info.update(self.update_critic(replay_buffer, validation_buffer))
        info.update(self.update_policy(filtered_buffer, filtered_validation_buffer,no_update_actor=no_update_actor))
        return info

    def save(self, path):
        self.accelerator.save_state(path, safe_serialization=False)

    def load(self, path):
        self.accelerator.load_state(path)
        
        