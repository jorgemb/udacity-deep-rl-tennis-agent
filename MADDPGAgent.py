import numpy as np
import torch

from AbstractAgent import AbstractAgent
from DDPGAgent import DDPGAgent, Critic, ModifiedDDPGAgent
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
        reward = np.asarray(reward, dtype=float)
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


class MADDPGAgent(AbstractAgent):
    def __init__(self, state_size, action_size, total_agents, *, gamma=1.0, alpha=0.1, seed=-1,
                 **kwargs) -> None:
        super().__init__(state_size, action_size, gamma=gamma, alpha=alpha, seed=seed, **kwargs)

        self.buffer_size = kwargs.get('buffer_size')
        self.batch_size = kwargs.get('batch_size')
        self.replay = ReplayBuffer(
            action_size,
            buffer_size=self.buffer_size,
            batch_size=self.batch_size,
            device=self.device
        )

        self.total_agents = total_agents
        self.agents = [
            ModifiedDDPGAgent(state_size, action_size, total_agents, i, self.replay, **kwargs)
            for i in range(total_agents)
        ]

        self.last_state = None
        self.last_action = None

    def start(self, state):
        self.last_state = np.asarray(state, dtype=float)
        self.last_action = np.zeros((self.total_agents, self.action_size))

        for i, agent in enumerate(self.agents):
            self.last_action[i] = agent.start(self.last_state[i])

        return self.last_action

    def step(self, state, reward, learn=True):
        # Modify reward for the time that agents manage to keep the ball in the air
        reward = np.asarray(reward, dtype=float)
        reward += 0.01

        # Add to memory
        new_state = np.asarray(state, dtype=float)
        self.replay.add(self.last_state, self.last_action, reward, new_state, False)

        self.last_state = new_state
        self.last_action = np.zeros((self.total_agents, self.action_size))

        for i, agent in enumerate(self.agents):
            self.last_action[i] = agent.step(new_state[i], reward, learn)

        return self.last_action

    def end(self, reward):
        reward = np.asarray(reward, dtype=float)
        self.replay.add(self.last_state, self.last_action, reward, self.last_state, True)
        for agent in self.agents:
            agent.end(reward)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['replay']
        return state

    def __setstate__(self, state):
        super().__setstate__(state)

        self.replay = ReplayBuffer(
            self.action_size,
            buffer_size=self.buffer_size,
            batch_size=self.batch_size,
            device=self.device
        )

        for agent in self.agents:
            agent.shared_replay = self.replay

