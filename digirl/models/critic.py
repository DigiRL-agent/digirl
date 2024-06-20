import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn

class VLMDoubleCritic(torch.nn.Module):
    def __init__(self, device, accelerator, critic_lm, cache_dir, in_dim, out_dim):
        """
        VLM critic using image features
        """
        super(VLMDoubleCritic, self).__init__()
        self.device = device
        self.accelerator = accelerator
        self.base_lm = AutoModel.from_pretrained(critic_lm, cache_dir=cache_dir).to(device)
        self.base_tokenizer = AutoTokenizer.from_pretrained(critic_lm, cache_dir=cache_dir)
        self.base_tokenizer.truncation_side = 'left'
        image_feature_dim = 1408*2
        out_dim = 2

        # for v
        self.critic1 = nn.Sequential(nn.Linear(in_dim+image_feature_dim, in_dim),\
                                    nn.ReLU(),\
                                    nn.Linear(in_dim, in_dim),\
                                    nn.ReLU(),\
                                    nn.Linear(in_dim, out_dim)).to(device)
        self.critic2 = nn.Sequential(nn.Linear(in_dim+image_feature_dim, in_dim),\
                                    nn.ReLU(),\
                                    nn.Linear(in_dim, in_dim),\
                                    nn.ReLU(),\
                                    nn.Linear(in_dim, out_dim)).to(device)

    def forward(self, observation, image_features, action, detach_model=False):
        detach_model = True
        obs_ids = self.base_tokenizer(observation, padding = True, return_tensors='pt', max_length=512, truncation = True).to(self.device)
        if detach_model:
            with torch.no_grad():
                lm_states = self.base_lm(**obs_ids).pooler_output
        else:
            lm_states = self.base_lm(**obs_ids).pooler_output
        v_states = torch.cat([lm_states, image_features], dim = 1)
        return self.critic1(v_states), self.critic2(v_states)


class TrajectoryCritic(torch.nn.Module):
    def __init__(self, device, accelerator, critic_lm, cache_dir, in_dim, out_dim):
        """
        VLM critic using image features
        """
        super(TrajectoryCritic, self).__init__()
        self.device = device
        self.accelerator = accelerator
        self.base_lm = AutoModel.from_pretrained(critic_lm, cache_dir=cache_dir).to(device)
        self.base_tokenizer = AutoTokenizer.from_pretrained(critic_lm, cache_dir=cache_dir)
        self.base_tokenizer.truncation_side = 'left'
        out_dim = 2
        self.critic = nn.Linear(in_dim, out_dim).to(device)

    def forward(self, observation, detach_model=False):
        detach_model = False
        obs_ids = self.base_tokenizer(observation, padding = True, return_tensors='pt', max_length=512, truncation = True).to(self.device)
        if detach_model:
            with torch.no_grad():
                lm_states = self.base_lm(**obs_ids).pooler_output
        else:
            lm_states = self.base_lm(**obs_ids).pooler_output
        return self.critic(lm_states)
