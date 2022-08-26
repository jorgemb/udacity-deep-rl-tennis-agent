from AbstractAgent import AbstractAgent

class MADDPGAgent(AbstractAgent):
    def __init__(self, state_size, action_size, *, gamma=1.0, alpha=0.1, seed=-1, **kwargs) -> None:
        super().__init__(state_size, action_size, gamma=gamma, alpha=alpha, seed=seed, **kwargs)

    def start(self, state):
        pass

    def step(self, state, reward, learn=True):
        pass

    def end(self, reward):
        pass

    def __getstate__(self):
        return super().__getstate__()

    def __setstate__(self, state):
        super().__setstate__(state)