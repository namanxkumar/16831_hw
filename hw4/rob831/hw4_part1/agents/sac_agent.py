from collections import OrderedDict

from rob831.hw4_part1.infrastructure.replay_buffer import ReplayBuffer
from rob831.hw4_part1.infrastructure.utils import *
from .base_agent import BaseAgent
import gym
from rob831.hw4_part1.policies.sac_policy import MLPPolicySAC
from rob831.hw4_part1.critics.sac_critic import SACCritic
import rob831.hw4_part1.infrastructure.pytorch_util as ptu
import torch

class SACAgent(BaseAgent):
    def __init__(self, env: gym.Env, agent_params):
        super(SACAgent, self).__init__()

        self.env = env
        self.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        self.agent_params = agent_params
        self.gamma = self.agent_params['gamma']
        self.critic_tau = 0.005
        self.learning_rate = self.agent_params['learning_rate']

        self.actor = MLPPolicySAC(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            self.agent_params['discrete'],
            self.agent_params['learning_rate'],
            action_range=self.action_range,
            init_temperature=self.agent_params['init_temperature']
        )
        self.actor_update_frequency = self.agent_params['actor_update_frequency']
        self.critic_target_update_frequency = self.agent_params['critic_target_update_frequency']

        self.critic = SACCritic(self.agent_params)
        self.critic_target = copy.deepcopy(self.critic).to(ptu.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.training_step = 0
        self.replay_buffer = ReplayBuffer(max_size=100000)

    def update_critic(self, ob_no, ac_na, next_ob_no, re_n, terminal_n):
        ob_no = ptu.from_numpy(ob_no)
        ac_na = ptu.from_numpy(ac_na)
        next_ob_no = ptu.from_numpy(next_ob_no)
        re_n = ptu.from_numpy(re_n)
        terminal_n = ptu.from_numpy(terminal_n)

        # compute target Q
        with torch.no_grad():
            next_dist = self.actor(next_ob_no)
            next_action = next_dist.rsample()
            next_log_prob = next_dist.log_prob(next_action).sum(-1, keepdim=True)
            target_q1, target_q2 = self.critic_target(next_ob_no, next_action)
            target_q = torch.min(target_q1, target_q2).unsqueeze(-1)
            target_q = re_n.unsqueeze(-1) + self.gamma * (1 - terminal_n.unsqueeze(-1)) * (target_q - self.actor.alpha.detach() * next_log_prob)

        q1, q2 = self.critic(ob_no, ac_na)
        critic_loss = torch.nn.functional.mse_loss(q1.unsqueeze(-1), target_q) + torch.nn.functional.mse_loss(q2.unsqueeze(-1), target_q)

        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        return critic_loss.item()

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        loss = OrderedDict()

        # update critic
        for _ in range(self.agent_params.get('num_critic_updates_per_agent_update', 1)):
            critic_loss = self.update_critic(ob_no, ac_na, next_ob_no, re_n, terminal_n)
            loss['Critic Loss'] = critic_loss

        # update actor
        if self.training_step % self.actor_update_frequency == 0:
            actor_loss, alpha_loss, alpha = self.actor.update(ptu.from_numpy(ob_no), self.critic)
            loss['Actor Loss'] = actor_loss
            loss['Alpha Loss'] = alpha_loss
            loss['Alpha'] = alpha

        # soft update target critic
        if self.training_step % self.critic_target_update_frequency == 0:
            from rob831.hw4_part1.infrastructure.sac_utils import soft_update_params
            soft_update_params(self.critic, self.critic_target, self.critic_tau)

        self.training_step += 1
        return loss

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_random_data(batch_size)
