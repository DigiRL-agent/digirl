import torch
from transformers import AutoTokenizer
from digirl.models.critic import VLMDoubleCritic, TrajectoryCritic
from .model import T5ForMultimodalGeneration
import signal

class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message
    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)
    def __exit__(self, type, value, traceback):
        signal.alarm(0)

class AutoUIAgent(torch.nn.Module):
    def __init__(self, device, accelerator, policy_lm = "gpt2", critic_lm = "roberta-base", 
                cache_dir = '~/.cache', dropout = 0.5, TEMPLATE = None, use_lora=False,
                do_sample = True, temperature = 1.0, max_new_tokens = 32, use_bfloat16 = False, eos_str = None):
        super(AutoUIAgent, self).__init__()
        if use_bfloat16:
            self.model = T5ForMultimodalGeneration.from_pretrained(policy_lm, cache_dir=cache_dir,
                                                              torch_dtype = torch.bfloat16).to(device)
        else:
            self.model = T5ForMultimodalGeneration.from_pretrained(policy_lm, cache_dir=cache_dir).to(device)
        if use_lora:
            from peft import LoraConfig, TaskType, get_peft_model
            lora_config = LoraConfig(
                r=16,
                target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
                task_type=TaskType.CAUSAL_LM,
                lora_alpha=32,
                lora_dropout=0.05
            )
            self.model = get_peft_model(self.model, lora_config)
            print("Using LoRA")
            self.model.print_trainable_parameters()
        self.template = TEMPLATE
        self.policy_lm = policy_lm
        self.critic = VLMDoubleCritic(device, accelerator, critic_lm = critic_lm, cache_dir = cache_dir, in_dim = 768, out_dim = 1)  
        self.trajectory_critic = TrajectoryCritic(device, accelerator, critic_lm = critic_lm, cache_dir = cache_dir, in_dim = 768, out_dim = 1)
        self.target_critic = None
        self.tokenizer = AutoTokenizer.from_pretrained(policy_lm, trust_remote_code=True, cache_dir=cache_dir)
        self.tokenizer.truncation_side = 'left'
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.device = device
        self.dropout = torch.nn.Dropout(p=dropout)
        self.softmax = torch.nn.Softmax(dim= -1)
        self.do_sample = do_sample
        self.temperature = temperature
        self.accelerator = accelerator
        self.max_new_tokens = max_new_tokens
        self.eos_str = eos_str
    
    def prepare(self):
        self.model = self.accelerator.prepare(self.model)
        self.critic = self.accelerator.prepare(self.critic)
        self.trajectory_critic = self.accelerator.prepare(self.trajectory_critic)

    def get_action(self, observation, image_features):
        image_features = image_features[..., -1408:]
        # if self.template is not None:
        #     observation = [self.template.replace("{obs}", obs) for obs in observation]
        for _ in range(3):
            try:
                with timeout(seconds=60):
                    with torch.no_grad():
                        obs_ids = self.tokenizer(observation, return_tensors='pt', padding=True, max_length=512, truncation = True).to(self.device)
                        image_features = image_features.to(self.device)
                        outputs = self.accelerator.unwrap_model(self.model).generate(**obs_ids, image_ids = image_features,
                                                    max_new_tokens=self.max_new_tokens, do_sample=self.do_sample, temperature = self.temperature,
                                                    pad_token_id = self.tokenizer.eos_token_id).cpu()
                    break
            except TimeoutError:
                print("Timeout while accessing actions")
                continue
        raw_action = self.tokenizer.batch_decode(outputs, skip_special_tokens  = True)
        for _ in range(3):
            raw_action = [a[1:] if a.startswith('\n') else a for a in raw_action]
        # return raw_action
        if self.eos_str is not None:
            # print(f"using eos str {eos_str}")
            # print([raw_a.split(self.eos_str)[0] + self.eos_str for raw_a in raw_action])
            return [raw_a.split(self.eos_str)[0] for raw_a in raw_action]
        else:
            return raw_action

    def get_log_prob(self, observation, image_features, action):
        image_features = image_features[...,-1408:]
        if self.template is not None:
            observation = [self.template.replace("{obs}", obs) for obs in observation]
        obs_ids = self.tokenizer(observation, return_tensors='pt', padding=True, max_length=512, truncation = True).to(self.device)
        action_ids = self.tokenizer(action, return_tensors='pt', padding=True, max_length=512, truncation = True).to(self.device)
        outputs = self.model(input_ids = obs_ids["input_ids"],
                            image_ids = image_features,
                            attention_mask = obs_ids["attention_mask"],
                            labels = action_ids["input_ids"])
        
        # # action_embeds = self.model.get_input_embeddings()(action_ids["input_ids"]).detach()
        # # obs_embeds = self.model.get_input_embeddings()(obs_ids["input_ids"]).detach()
        # input_ids = torch.cat([obs_ids["input_ids"], action_ids["input_ids"]], dim = 1)
        # # input_embeds = torch.cat([obs_embeds, action_embeds], dim = 1)
        # attention_mask = torch.cat([obs_ids["attention_mask"], action_ids["attention_mask"]],\
        #                         dim = 1)
        # outputs = self.model(input_ids=input_ids, attention_mask = attention_mask)
        # values = None
        # if isinstance(outputs, Tuple):
        #     values, outputs = outputs
        ## TODO: need to check if token shifting is done correctly
        prediction_probs = self.softmax(outputs.logits)
        selected_prediction_probs = torch.take_along_dim(prediction_probs,\
                                                 action_ids["input_ids"].unsqueeze(2), dim=2).squeeze(2)
        selected_prediction_probs = torch.clamp(selected_prediction_probs, min=0.001, max=0.99)
        # import IPython; IPython.embed(); exit()
        return torch.log(selected_prediction_probs)*action_ids["attention_mask"]