from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from rllib import random_process, schedule, ReplayBuffer
from AbstractAgent import AbstractAgent


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return -lim, lim


class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc_units=256):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        # self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc_units)
        self.fc2 = nn.Linear(fc_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = F.relu(self.fc1(state))
        return F.tanh(self.fc2(x))


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fcs1_units=256, fc2_units=256, fc3_units=128):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        # self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units + action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = F.leaky_relu(self.fcs1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        return self.fc4(x)


class DDPGAgent(AbstractAgent):
    def __init__(self, state_size, action_size, *, gamma=1.0, alpha=0.1, seed=-1, **kwargs) -> None:
        super().__init__(state_size, action_size, gamma=gamma, alpha=alpha, seed=seed, **kwargs)

        # Create actor-critic networks
        # Actor
        self.local_actor = Actor(state_size, action_size, self.seed).to(self.device)
        self.target_actor = Actor(state_size, action_size, self.seed).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.local_actor.parameters(), lr=self.alpha)

        # Critic
        self.local_critic = Critic(state_size, action_size, self.seed).to(self.device)
        self.target_critic = Critic(state_size, action_size, self.seed).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.local_critic.parameters(), lr=self.alpha)

        self.random_process = random_process.OrnsteinUhlenbeckProcess(
            size=(self.action_size,),
            std=schedule.LinearSchedule(0.2)
        )

        self.last_state = None
        self.last_action = None
        self.step_n = 0

        # Hyperparameters
        self.buffer_size = kwargs.get('buffer_size')
        self.batch_size = kwargs.get('batch_size')
        self.tau = kwargs.get('tau')
        self.learn_every = kwargs.get('learn_every', 10)

        self.replay = ReplayBuffer.ReplayBuffer(
            action_size,
            buffer_size=self.buffer_size,
            batch_size=self.batch_size,
            device=self.device
        )

    def take_action(self, state: np.array, add_noise: bool = True):
        """
        Takes an action using the local actor or a random value
        @param state:
        @param add_noise:
        @return:
        """
        # Get action without training
        s = torch.from_numpy(state).float().to(self.device)
        self.local_actor.eval()
        with torch.no_grad():
            action = self.local_actor(s).cpu().data.numpy()
        self.local_actor.train()

        if add_noise:
            action += self.random_process.sample()
        return np.clip(action, -1, 1)

    def start(self, state):
        self.step_n = 1

        # Take and action and store
        self.last_state = np.asarray(state, dtype=float)
        self.last_action = self.take_action(state, True)
        return self.last_action

    def step(self, state, reward, learn=True):
        # Store the SARS information in the replay buffer
        next_state = np.asarray(state, dtype=float)
        self.replay.add(self.last_state, self.last_action, reward, next_state, False)

        # Get next action
        self.last_state = next_state
        self.last_action = self.take_action(state, add_noise=True)
        self.step_n += 1

        # Check if we should do a learning step
        if learn and len(self.replay) > self.batch_size and self.step_n % self.learn_every == 0:
            self.learn_step()

        return self.last_action

    def learn_step(self):
        # Get sample
        states, actions, rewards, next_states, dones = self.replay.sample()

        # Update critic
        a_next = self.target_actor(next_states)
        q_target = self.target_critic(next_states, a_next)
        q_target = self.gamma * (1 - dones) * q_target + rewards
        q_target = q_target.detach()

        # .. compute expected
        q_expected = self.local_critic(states, actions)
        critic_loss = F.mse_loss(q_expected, q_target)

        # .. step with optimizer
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        a_predicted = self.local_actor(states)
        actor_loss = -self.local_critic(states, a_predicted).mean()

        # .. step with optimizer
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update the networks
        self.soft_update(self.target_actor, self.local_actor)
        self.soft_update(self.target_critic, self.local_critic)

    def end(self, reward):
        # Store the SARS information in the replay buffer
        self.replay.add(self.last_state, self.last_action, reward, self.last_state, True)

    def soft_update(self, target, src):
        for target_param, param in zip(target.parameters(), src.parameters()):
            target_param.detach_()
            target_param.copy_(target_param * (1.0 - self.tau) +
                               param * self.tau)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['replay']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.replay = ReplayBuffer.ReplayBuffer(
            self.action_size,
            buffer_size=self.buffer_size,
            batch_size=self.batch_size,
            device=self.device
        )
