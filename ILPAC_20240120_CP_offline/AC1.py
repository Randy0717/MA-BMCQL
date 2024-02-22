import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.distributions import Categorical

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    print('PyTorch is using CPU.')

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, act_dim),
            nn.Softmax(dim=-1),
        )

        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

        self.actor = self._init_layers(obs_dim, act_dim, hidden_size)
        self.critic = self._init_layers(obs_dim, 1, hidden_size)

    def _init_layers(self, in_dim, out_dim, hidden_size):
        layers = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_dim),
        )

        # 使用标准正态分布进行随机初始化
        for m in layers.modules():
            if isinstance(m, nn.Linear):
                init.normal_(m.weight, mean=0, std=0.01)  # 使用正态分布初始化权重
                init.constant_(m.bias, 0)  # 使用常数初始化偏差

        if out_dim == 1:
            # 对于critic的输出层，我们通常使用标量，所以没有softmax
            pass
        else:
            # 对于actor的输出层，我们通常使用softmax进行归一化
            layers.add_module('softmax', nn.Softmax(dim=-1))

        return layers

    def actor_forward(self, obs):
        prob = self.actor(obs)
        return prob

    def critic_forward(self, obs):
        value = self.critic(obs)
        return value

class A2CAgent:
    def __init__(self, hidden_size=256, critic_lr=1e-2, actor_lr=1e-2, gamma=0.99, tau = 0.005):
        self.obs_dim = 14
        self.act_dim = 2
        self.gamma = gamma
        self.training_steps = []
        self.entropy_alpha = 0
        self.tau = tau

        self.network = ActorCritic(self.obs_dim, self.act_dim, hidden_size).to(device)
        self.target_network = ActorCritic(self.obs_dim, self.act_dim, hidden_size).to(device)
        self.target_network.load_state_dict(self.network.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.network.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.network.critic.parameters(), lr=critic_lr)

    def get_action(self, obs):
        if type(obs) is list:
            obs = torch.tensor(obs, dtype=torch.float32).to(device)
        prob = self.network.actor_forward(obs)
        dist = Categorical(prob)
        action = dist.sample()
        value = self.network.critic_forward(obs)
        return action, value

    def update(self, M):
        batch_size = 512

        indices = np.random.randint(0, len(M), size=batch_size)
        batch = [M[i] for i in indices]

        states, actions, rewards, delay, next_states = zip(*batch)

        obs = torch.tensor(states, dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.float32).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        delays = torch.tensor(delay, dtype=torch.float32).to(device)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, next_states)), dtype=torch.bool).to(device)
        non_final_next_obs = torch.tensor([s for s in next_states if s is not None], dtype=torch.float32).to(device)

        # Compute values and next_values
        values = self.network.critic_forward(obs)
        next_values = torch.zeros(batch_size, device=device)
        next_values[non_final_mask] = self.target_network.critic_forward(non_final_next_obs).squeeze()
        deltas = rewards + self.gamma ** delays * next_values - values

        next_values_for_advantage = torch.zeros(batch_size, device=device)
        next_values_for_advantage[non_final_mask] = self.network.critic_forward(non_final_next_obs).squeeze()
        advantages = rewards + self.gamma ** delays * next_values_for_advantage - values


        # Update Critic
        critic_loss = deltas.pow(2).mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        for target_param, train_param in zip(self.target_network.critic.parameters(), self.network.critic.parameters()):
            target_param.data.copy_(self.tau * train_param.data + (1.0 - self.tau) * target_param.data)

        probs = self.network.actor_forward(obs)
        dists = Categorical(probs)
        values = self.network.critic_forward(obs)

        actor_loss = -(dists.log_prob(actions) * deltas.detach()).mean()
        entropy_loss = -self.entropy_alpha * dists.entropy().mean()  # 添加熵项
        loss = actor_loss + entropy_loss
        self.training_steps.append(float(critic_loss))

        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

    def save(self, path):
        torch.save(self.network.state_dict(), path)

    def load(self, path):
        self.network.load_state_dict(torch.load(path))
