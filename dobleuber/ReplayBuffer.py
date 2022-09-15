"""
Author: Wbert Adrián Castro Vera (dobleuber)
Source: https://github.com/dobleuber/DeepReinforcementLearningUdacity
License: <unspecified>
"""

import operator
import random
from collections import namedtuple

import numpy as np
import torch

ALPHA = 0.5
ALPHA_DECAY_RATE = 0.99
BETA = 0.5
BETA_GROWTH_RATE = 1.001


class ReplayBuffer:
    """
    Fixed-size buffer to store experience tuples.
    """

    def __init__(self, action_size, buffer_size, batch_size, experiences_per_sampling, device, seed):
        """
        Initialize the replay buffer.
        """
        self.action_size = action_size
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.experiences_per_sampling = experiences_per_sampling
        self.device = device

        self.alpha = ALPHA
        self.alpha_decay_rate = ALPHA_DECAY_RATE
        self.beta = BETA
        self.beta_growth_rate = BETA_GROWTH_RATE
        self.experience_count = 0

        self.experience = namedtuple('Experience', field_names=['state', 'action', 'reward', 'next_state', 'done'])
        self.data = namedtuple('Data', field_names=['priority', 'probability', 'weight', 'index'])

        indexes = []
        datas = []
        for i in range(buffer_size):
            indexes.append(i)
            d = self.data(0, 0, 0, i)
            datas.append(d)

        self.memory = {key: self.experience for key in indexes}
        self.memory_data = {key: data for key, data in zip(indexes, datas)}
        self.sampled_batches = []
        self.current_batch = 0
        self.priorities_sum_alpha = 0
        self.priorities_max = 1
        self.weights_max = 1

        random.seed(seed)

    def update_priorities(self, tds, indices):
        """
        Updates memory entries priorities
        :param tds: temporal differences
        :param indices: memory indices
        """
        for td, index in zip(tds, indices):
            N = min(self.experience_count, self.buffer_size)

            updated_priority = td[0]
            if updated_priority > self.priorities_max:
                self.priorities_max = updated_priority

            updated_weight = 1

            old_priority = self.memory_data[index].priority
            self.priorities_sum_alpha += updated_priority ** self.alpha - old_priority ** self.alpha
            updated_probability = td[0] ** self.alpha / self.priorities_sum_alpha
            data = self.data(updated_priority, updated_probability, updated_weight, index)
            self.memory_data[index] = data

    def update_memory_sampling(self):
        """
        Randomly sample X batches of experiences from memory.
        """
        self.current_batch = 0
        values = list(self.memory_data.values())
        random_values = random.choices(
            self.memory_data,
            [data.probability for data in values],
            k=self.experiences_per_sampling
        )

        self.sampled_batches = [random_values[i:i + self.batch_size]
                                for i in range(0, len(random_values), self.batch_size)]

    def update_parameters(self):
        """
        Update memory parameters
        """
        self.alpha *= self.alpha_decay_rate
        self.beta *= self.beta_growth_rate

        if self.beta > 1:
            self.beta = 1

        N = min(self.experience_count, self.buffer_size)
        self.priorities_sum_alpha = 0

        sum_prob_before = 0
        for element in self.memory_data.values():
            sum_prob_before += element.probability
            self.priorities_sum_alpha += element.priority ** self.alpha

        sum_prob_after = 0
        for element in self.memory_data.values():
            probability = element.priority ** self.alpha / self.priorities_sum_alpha
            sum_prob_after += probability
            weight = 1
            d = self.data(element.priority, probability, weight, element.index)
            self.memory_data[element.index] = d

    def add(self, state, action, reward, next_state, done):
        """
        Add experience to the memory
        """
        self.experience_count += 1
        index = self.experience_count % self.buffer_size

        if self.experience_count > self.buffer_size:
            temp = self.memory_data[index]
            self.priorities_sum_alpha -= temp.priority ** self.alpha

            if temp.priority == self.priorities_max:
                self.memory_data[index].priority = 0
                self.priorities_max = max(self.memory_data.items(), key=operator.itemgetter(1)).priority

        priority = self.priorities_max
        weight = self.weights_max
        self.priorities_sum_alpha += priority ** self.alpha
        probability = priority ** self.alpha / self.priorities_sum_alpha
        e = self.experience(state, action, reward, next_state, done)
        self.memory[index] = e

        d = self.data(priority, probability, weight, index)
        self.memory_data[index] = d

    def sample(self):
        """
        Gets a random batch of experience from memory
        """
        sampled_batch = self.sampled_batches[self.current_batch]
        self.current_batch += 1
        experiences = []
        weights = []
        indices = []

        for data in sampled_batch:
            experiences.append(self.memory.get(data.index))
            weights.append(data.weight)
            indices.append(data.index)

        device = self.device
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])) \
            .float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)) \
            .float().to(device)

        return states, actions, rewards, next_states, dones, indices

    def __len__(self):
        """
        Returns the current size of the memory
        """
        return len(self.memory)
