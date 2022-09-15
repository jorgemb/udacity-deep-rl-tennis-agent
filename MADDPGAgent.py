import numpy as np
import torch

import dobleuber.Agent
from AbstractAgent import AbstractAgent
from DDPGAgent import DDPGAgent, Critic, ModifiedDDPGAgent, PriorityDDPG
from rllib.ReplayBuffer import ReplayBuffer


class NaiveDDPG(AbstractAgent):
    def __init__(self, state_size, action_size, total_agents, *, gamma=1.0, alpha=0.1, seed=-1, **kwargs) -> None:
        super().__init__(state_size, action_size, gamma=gamma, alpha=alpha, seed=seed, **kwargs)

        self.total_agents = total_agents

        self.agents = [
            DDPGAgent(state_size, action_size, **kwargs)
            for i in range(self.total_agents)
        ]

    def start(self, state):
        actions = np.zeros((self.total_agents, self.action_size))
        for i, agent in enumerate(self.agents):
            actions[i] = agent.start(state[i])

        return actions

    def step(self, state, reward, learn=True):
        # Try random
        learn = False

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

        # DEPENDENCY: Add the other_agents variable to each agent
        for agent in self.agents:
            agent.other_agents = self.agents

    def start(self, state):
        self.last_state = np.asarray(state, dtype=float)
        self.last_action = np.zeros((self.total_agents, self.action_size))

        for i, agent in enumerate(self.agents):
            self.last_action[i] = agent.start(self.last_state[i])

        return self.last_action

    def step(self, state, reward, learn=True):
        # Modify reward for the time that agents manage to keep the ball in the air
        reward = np.asarray(reward, dtype=float)

        # Add to memory
        new_state = np.asarray(state, dtype=float)
        self.replay.add(self.last_state, self.last_action, reward, new_state, False)

        self.last_state = new_state
        self.last_action = np.zeros((self.total_agents, self.action_size))

        for i, agent in enumerate(self.agents):
            self.last_action[i] = agent.step(new_state[i], reward[i], learn)

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
            agent.other_agents = self.agents


class SingleDDPG(AbstractAgent):
    def __init__(self, state_size, action_size, total_agents, *, gamma=1.0, alpha=0.1, seed=-1, **kwargs) -> None:
        super().__init__(state_size, action_size, gamma=gamma, alpha=alpha, seed=seed, **kwargs)

        self.total_agents = total_agents
        self.agent = DDPGAgent(total_agents * state_size, total_agents * action_size, **kwargs)

    def start(self, state):
        state = state.flatten()
        return self.agent.start(state)

    def step(self, state, reward, learn=True):
        state = state.flatten()
        reward = np.sum(reward)

        return self.agent.step(state, reward, learn)

    def end(self, reward):
        reward = np.sum(reward)
        return self.agent.end(reward)


class SingleAgent(AbstractAgent):

    def __init__(self, state_size, action_size, total_agents, *, gamma=1.0, alpha=0.1, seed=-1, **kwargs) -> None:
        super().__init__(state_size, action_size, gamma=gamma, alpha=alpha, seed=seed, **kwargs)

        self.agent = DDPGAgent(state_size, action_size, **kwargs)
        self.total_agents = total_agents

    def start(self, state):
        self.agent.start(state[0])
        actions = np.random.random((self.total_agents, self.action_size))

        return np.clip(actions, -1, 1)

    def step(self, state, reward, learn=True):
        actions = np.zeros((self.total_agents, self.action_size), dtype=float)

        for i in range(self.total_agents):
            actions[i, :] = self.agent.step(state[i], reward[i], learn)

        return actions

    def end(self, reward):
        return self.agent.end(reward[0])


class DobleUber(AbstractAgent):
    def __init__(self, state_size, action_size, total_agents, *, gamma=1.0, alpha=0.1, seed=-1, **kwargs) -> None:
        super().__init__(state_size, action_size, gamma=gamma, alpha=alpha, seed=seed, **kwargs)

        self.agent = dobleuber.Agent.Agent(state_size, action_size, total_agents, self.seed)

        self.total_agents = total_agents
        self.last_state = None
        self.last_action = None

    def start(self, state):
        self.last_state = state
        self.last_action = self.agent.act(self.last_state)

        return self.last_action

    def step(self, state, reward, learn=True):
        if learn:
            self.agent.step(self.last_state, self.last_action, reward, state, np.zeros((self.total_agents,)))

        self.last_state = state
        self.last_action = self.agent.act(self.last_state)

        return self.last_action

    def end(self, reward):
        self.agent.step(self.last_state, self.last_action, reward, self.last_state, np.ones((self.total_agents,)))

    def __getstate__(self):
        state = self.__dict__.copy()
        state['agent'].memory = None
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self.agent.create_memory()


class PriorityAgent(NaiveDDPG):
    def __init__(self, state_size, action_size, total_agents, *, gamma=1.0, alpha=0.1, seed=-1, **kwargs) -> None:
        super().__init__(state_size, action_size, total_agents, gamma=gamma, alpha=alpha, seed=seed, **kwargs)

        self.agents = [
            PriorityDDPG(state_size, action_size, **kwargs)
            for _ in range(total_agents)
        ]


class PriorityAgent_SharedCritic(PriorityAgent):
    def __init__(self, state_size, action_size, total_agents, *, gamma=1.0, alpha=0.1, seed=-1, **kwargs) -> None:
        super().__init__(state_size, action_size, total_agents, gamma=gamma, alpha=alpha, seed=seed, **kwargs)

        self.share_critic()

    def share_critic(self):
        local_critic = self.agents[0].local_critic
        target_critic = self.agents[0].target_critic
        critic_optimizer = self.agents[0].critic_optimizer

        for agent in self.agents:
            agent.local_critic = local_critic
            agent.target_critic = target_critic
            agent.critic_optimizer = critic_optimizer

    def __setstate__(self, state):
        super().__setstate__(state)
        self.share_critic()


class PriorityAgent_SharedActor(PriorityAgent):

    def __init__(self, state_size, action_size, total_agents, *, gamma=1.0, alpha=0.1, seed=-1, **kwargs) -> None:
        super().__init__(state_size, action_size, total_agents, gamma=gamma, alpha=alpha, seed=seed, **kwargs)

    def share_actor(self):
        local_actor = self.agents[0].local_actor
        target_actor = self.agents[0].target_actor
        actor_optimizer = self.agents[0].actor_optimizer

        for agent in self.agents:
            agent.local_actor = local_actor
            agent.target_actor = target_actor
            actor_optimizer = actor_optimizer

    def __setstate__(self, state):
        super().__setstate__(state)

        self.share_actor()
