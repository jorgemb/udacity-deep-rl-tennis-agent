from AbstractAgent import AbstractAgent
import numpy as np

from DDPGAgent import ModifiedDDPGAgent, DDPGAgent
from rllib.ReplayBuffer import ReplayBuffer


class NaiveMADDPGAgent(AbstractAgent):
    def __init__(self, state_size, action_size, total_agents, *, gamma=1.0, alpha=0.1, seed=-1, **kwargs) -> None:
        super().__init__(state_size, action_size, gamma=gamma, alpha=alpha, seed=seed, **kwargs)

        # Create agents
        self.total_agents = total_agents
        self.agents = [
            DDPGAgent(state_size, action_size, **kwargs) for i in range(total_agents)
        ]

    def start(self, state):
        actions = np.zeros((self.total_agents, self.action_size))

        for i, agent in enumerate(self.agents):
            actions[i] = agent.start(state[i])

        return actions

    def step(self, state, reward, learn=True):
        actions = np.zeros((self.total_agents, self.action_size))

        for i, agent in enumerate(self.agents):
            actions[i] = agent.step(state[i], reward, learn)

        return actions

    def end(self, reward):
        for agent in self.agents:
            agent.end(reward)


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

        # Use the same actor on both agents
        # local_actor = self.agents[0].local_actor
        # target_actor = self.agents[0].target_actor
        # actor_optimizer = self.agents[0].actor_optimizer

        # for a in self.agents:
        #     a.local_actor = local_actor
        #     a.target_actor = target_actor
        #     a.actor_optimizer = actor_optimizer

    def start(self, state):
        self.last_state = np.asarray(state, dtype=float)
        self.last_action = np.zeros((self.total_agents, self.action_size))

        for i, agent in enumerate(self.agents):
            self.last_action[i] = agent.start(state)

        return self.last_action

    def step(self, state, reward, learn=True):
        # Add to memory
        new_state = np.asarray(state, dtype=float)
        self.replay.add(self.last_state, self.last_action, reward, new_state, False)

        self.last_state = new_state
        self.last_action = np.zeros((self.total_agents, self.action_size))

        for i, agent in enumerate(self.agents):
            self.last_action[i] = agent.step(state, reward, learn)

        return self.last_action

    def end(self, reward):
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
