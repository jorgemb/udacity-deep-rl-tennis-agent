import numpy as np
import torch

from AbstractAgent import AbstractAgent
from DDPGAgent import DDPGAgent, Critic
from rllib.ReplayBuffer import ReplayBuffer


class NaiveDDPG(AbstractAgent):
    def __init__(self, state_size, action_size, total_agents, *, gamma=1.0, alpha=0.1, seed=-1, **kwargs) -> None:
        super().__init__(state_size, action_size, gamma=gamma, alpha=alpha, seed=seed, **kwargs)

        self.total_agents = total_agents

        self.agents = [
            DDPGAgent(state_size, action_size, **kwargs)
        ]

    def start(self, state):
        actions = np.zeros((self.total_agents, self.action_size))
        for i, agent in enumerate(self.agents):
            actions[i] = agent.start(state[i])

        return actions

    def step(self, state, reward, learn=True):
        actions = np.zeros((self.total_agents, self.action_size))
        for i, agent in enumerate(self.agents):
            actions[i] = agent.step(state[i], reward[i], learn)

        return actions

    def end(self, reward):
        for i, agent in enumerate(self.agents):
            agent.end(reward[i])


class SharedReplay(NaiveDDPG):
    def __init__(self, state_size, action_size, total_agents, *, gamma=1.0, alpha=0.1, seed=-1, **kwargs) -> None:
        super().__init__(state_size, action_size, total_agents, gamma=gamma, alpha=alpha, seed=seed, **kwargs)

        self.shared_replay = None
        self.config = kwargs
        self.create_shared_replay()

    def create_shared_replay(self):
        self.shared_replay = ReplayBuffer(
            self.action_size,
            self.config.get('buffer_size'),
            self.config.get('batch_size'),
            self.device)
        for agent in self.agents:
            agent.replay = self.shared_replay

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['shared_replay']
        return state

    def __setstate__(self, state):
        data = super().__setstate__(state)
        self.create_shared_replay()


class SharedCritic(NaiveDDPG):
    def __init__(self, state_size, action_size, total_agents, *, gamma=1.0, alpha=0.1, seed=-1, **kwargs) -> None:
        super().__init__(state_size, action_size, total_agents, gamma=gamma, alpha=alpha, seed=seed, **kwargs)

        self.create_shared_critic()

    def create_shared_critic(self):
        target_critic = Critic(self.state_size, self.action_size, self.seed).to(self.device)
        local_critic = Critic(self.state_size, self.action_size, self.seed).to(self.device)
        critic_optimizer = torch.optim.Adam(local_critic.parameters(), lr=self.alpha)

        for agent in self.agents:
            agent.target_critic = target_critic
            agent.local_critic = local_critic
            critic_optimizer = critic_optimizer

    def __setstate__(self, state):
        super().__setstate__(state)

        self.create_shared_critic()
