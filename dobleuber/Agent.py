"""
Author: Wbert Adrián Castro Vera (dobleuber)
Source: https://github.com/dobleuber/DeepReinforcementLearningUdacity
License: <unspecified>
"""

import random
from math import ceil

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from dobleuber.Actor import Actor
from dobleuber.Critic import Critic
from dobleuber.Noise import OUNoise
from dobleuber.ReplayBuffer import ReplayBuffer

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR_ACTOR = 1e-4  # learning rate of the actor
LR_CRITIC = 1e-4
# learning rate of the critic
WEIGHT_DECAY = 0  # L2 weight decay

# prioritized experience replay
UPDATE_NN_EVERY = 1  # how often to update the network
UPDATE_MEM_EVERY = 20  # how often to update the priorities
UPDATE_MEM_PAR_EVERY = 3000  # how often to update the hyperparameters
EXPERIENCES_PER_SAMPLING = ceil(BATCH_SIZE * UPDATE_MEM_EVERY / UPDATE_NN_EVERY)

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')


class Agent:
    """
    Interacts with and learns from the environment.
    """

    def __init__(self, state_size, action_size, num_agents, random_seed):
        """
        Initialize an Agent

        Params
        ======
            state_size (int): state dimension
            action_size (int): action dimension
            num_agents (int): simultaneous running agents
            random_seed (int): random seed
        """

        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        random.seed(random_seed)

        # Actor Network and its target network
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network and its target network
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise object
        self.noise = OUNoise((num_agents, action_size), random_seed)

        # Replay Memory
        self.random_seed = random_seed
        # self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, EXPERIENCES_PER_SAMPLING, device, random_seed)
        self.memory = None
        self.create_memory()

        # Initialize time step (for updating every UPDATE_NN_EVERY steps)
        self.t_step_nn = 0
        # Initialize time step (for updating every UPDATE_MEM_PAR_EVERY steps)
        self.t_step_mem_par = 0
        # Initialize time step (for updating every UPDATE_MEM_EVERY steps)
        self.t_step_mem = 0

    def create_memory(self):
        self.memory = ReplayBuffer(self.action_size, BUFFER_SIZE, BATCH_SIZE, EXPERIENCES_PER_SAMPLING, device,
                                   self.random_seed)

    def step(self, state, action, reward, next_state, done):
        """
        Save experience in replay memory, and use prioritized sample from buffer to learn.
        """

        # Save memory
        for i in range(self.num_agents):
            self.memory.add(state[i, :], action[i, :], reward[i], next_state[i, :], done[i])

        # Learn every UPDATE_NN_EVERY time steps.
        self.t_step_nn = (self.t_step_nn + 1) % UPDATE_NN_EVERY
        self.t_step_mem = (self.t_step_mem + 1) % UPDATE_MEM_EVERY
        self.t_step_mem_par = (self.t_step_mem_par + 1) % UPDATE_MEM_PAR_EVERY

        if self.t_step_mem_par == 0:
            self.memory.update_parameters()
        if self.t_step_nn == 0:
            # Learn from memory if enough samples exist
            if self.memory.experience_count > EXPERIENCES_PER_SAMPLING:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

        if self.t_step_mem == 0:
            self.memory.update_memory_sampling()

    def act(self, states, add_noise=True):
        """
        Returns actions for given state as per current policy.
        """
        states = torch.from_numpy(states).float().to(device)
        actions = np.zeros((self.num_agents, self.action_size))
        self.actor_local.eval()
        with torch.no_grad():
            for i, state in enumerate(states):
                action = self.actor_local(state).cpu().data.numpy()
                actions[i, :] = action

        self.actor_local.train()
        if add_noise:
            actions += self.noise.sample()

        return np.clip(actions, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """
        Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones, indices = experiences

        # update Critic
        # Get next predicted state, actions, and Q values
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)

        # Compute Q targets for current state
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Compute Critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update Actor
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()

        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

        # Update priorities
        delta = abs(Q_targets - Q_expected).detach().numpy()
        self.memory.update_priorities(delta, indices)

    @staticmethod
    def soft_update(local_model, target_model, tau):
        """
        Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """

        for target_model_param, local_model_param in zip(target_model.parameters(), local_model.parameters()):
            target_model_param.data.copy_(tau * local_model_param.data + (1. - tau) * target_model_param.data)
