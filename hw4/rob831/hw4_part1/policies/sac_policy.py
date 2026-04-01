from rob831.hw4_part1.policies.MLP_policy import MLPPolicy
import torch
import numpy as np
from rob831.hw4_part1.infrastructure import sac_utils
from rob831.hw4_part1.infrastructure import pytorch_util as ptu
from torch import nn
from torch import optim
import itertools

class MLPPolicySAC(MLPPolicy):
    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=3e-4,
                 training=True,
                 log_std_bounds=[-20,2],
                 action_range=[-1,1],
                 init_temperature=1.0,
                 **kwargs
                 ):
        super(MLPPolicySAC, self).__init__(ac_dim, ob_dim, n_layers, size, discrete, learning_rate, training, **kwargs)
        self.log_std_bounds = log_std_bounds
        self.action_range = action_range
        self.init_temperature = init_temperature
        self.learning_rate = learning_rate

        self.log_alpha = torch.tensor(np.log(self.init_temperature)).to(ptu.device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.learning_rate)

        self.target_entropy = -ac_dim

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def get_action(self, obs: np.ndarray, sample=True) -> np.ndarray:
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]
        observation = ptu.from_numpy(observation)
        action_distribution = self.forward(observation)
        if sample:
            action = action_distribution.sample()
        else:
            action = action_distribution.mean
        action = action.clamp(*self.action_range)
        return ptu.to_numpy(action)[0]

    def forward(self, observation: torch.FloatTensor):
        mean = self.mean_net(observation)
        log_std = self.logstd.expand_as(mean)  # broadcast logstd
        log_std = torch.clamp(log_std, *self.log_std_bounds)
        std = torch.exp(log_std)
        action_distribution = sac_utils.SquashedNormal(mean, std)
        return action_distribution

    def update(self, obs, critic):
        dist = self.forward(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)

        q1, q2 = critic(obs, action)
        q = torch.min(q1, q2)

        actor_loss = (self.alpha.detach() * log_prob - q.unsqueeze(-1)).mean()

        self.optimizer.zero_grad()
        actor_loss.backward()
        self.optimizer.step()

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        return actor_loss.item(), alpha_loss.item(), self.alpha