import random
from abc import abstractmethod
from os import urandom

import numpy as np
import torch


class AbstractAgent:
    def __init__(self, state_size, action_size, *, gamma=1.0, alpha=0.1, seed=-1, **kwargs) -> None:
        """ Initializes the Agent with environment information and hyperparameters

        Args:
            space_size (int): Size of the state space
            action_size (int): Size of the action space
            gamma (float, optional): Discount rate. Defaults to 1.0.
            alpha (float, optional): Learning rate. Defaults to 0.1.
            seed (int, optional): Seed. If -1 then a random seed is used.
        """
        self.state_size = state_size
        self.action_size = action_size

        self.gamma = gamma
        self.alpha = alpha

        self.epsilon = 1.0
        self.epsilon_min = 0.05

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.device = "cpu"

        if seed == -1:
            self.seed = int.from_bytes(urandom(4), byteorder="little")
        else:
            self.seed = seed

        random.seed(self.seed)
        torch.manual_seed(self.seed)

    @abstractmethod
    def start(self, state):
        """Marks the start of an episode with the initial state. Returns the initial action.

        Args:
            state (array_like): Initial state

        Returns:
            int: Initial action
        """
        pass

    @abstractmethod
    def step(self, state, reward, learn=True):
        """Performs a step in the simulation, provides the reward of the last action and the next state.

        Args:
            state (array_like): New state
            reward (float): Reward from previous action

        Returns:
            int: Next action
        """
        pass

    @abstractmethod
    def end(self, reward):
        """Finishes an episode, provides the last reward that was provided.

        Args:
            reward (float): Reward from previous action
        """
        pass

    def __getstate__(self):
        """Return a state for pickling
        """
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        """Restores the instance attributes from a state

        Args:
            state (dict): Dictionary of elements
        """
        self.__dict__.update(state)

    def agent_name(self):
        return self.__class__.__name__


class RandomAgent(AbstractAgent):
    def __init__(self, state_size, action_size, *, gamma=1, alpha=0.1, seed=-1, **kwargs) -> None:
        super().__init__(state_size, action_size, gamma=gamma, alpha=alpha, seed=seed, **kwargs)

    def random_action(self):
        return np.random.uniform(-1, 1, self.action_size)

    def start(self, state):
        return self.random_action()

    def end(self, reward):
        pass

    def step(self, reward, state, learn=True):
        return self.random_action()
