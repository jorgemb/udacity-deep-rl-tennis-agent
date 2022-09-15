#!/usr/bin/env python
# coding: utf-8

import time

import numpy as np
import torch
from unityagents import UnityEnvironment

import wandb
from AbstractAgent import AbstractAgent, RandomAgent
from MADDPGAgent import NaiveDDPG, SharedReplay, SharedCritic, MADDPGAgent, SingleDDPG, DobleUber, SingleAgent, \
    PriorityAgent, PriorityAgent_SharedCritic, PriorityAgent_SharedActor


def create_agent(state_space, action_space, **kwargs):
    """
    Create the list of agents to test.
    @param state_space:
    @param action_space:
    @return: Created agent
    """
    agent_name = kwargs.get('agent')

    if agent_name == 'RandomAgent':
        return RandomAgent(state_space, action_space * 2, **kwargs)
    elif agent_name == 'NaiveDDPG':
        return NaiveDDPG(state_space, action_space, 2, **kwargs)
    elif agent_name == 'SharedReplay':
        return SharedReplay(state_space, action_space, 2, **kwargs)
    elif agent_name == 'SharedCritic':
        return SharedCritic(state_space, action_space, 2, **kwargs)
    elif agent_name == 'MADDPGAgent':
        return MADDPGAgent(state_space, action_space, 2, **kwargs)
    elif agent_name == 'SingleDDPG':
        return SingleDDPG(state_space, action_space, 2, **kwargs)
    elif agent_name == 'SingleAgent':
        return SingleAgent(state_space, action_space, 2, **kwargs)
    elif agent_name == 'DobleUber':
        return DobleUber(state_space, action_space, 2, **kwargs)
    elif agent_name == 'PriorityAgent':
        return PriorityAgent(state_space, action_space, 2, **kwargs)
    elif agent_name == 'PriorityAgent_SharedCritic':
        return PriorityAgent_SharedCritic(state_space, action_space, 2, **kwargs)
    elif agent_name == 'PriorityAgent_SharedActor':
        return PriorityAgent_SharedActor(state_space, action_space, 2, **kwargs)
    else:
        raise f'Unknown agent: {agent_name}'


def do_experiment(environment, brain_name, agent: AbstractAgent, total_episodes: int, print_every: int, learn=True):
    """Performs an experiment on the given agent.

    Args:
        environment (any): Environment to use
        agent (AbstractAgent): Agent that follows the "Agent" interface
        total_episodes (int): Amount of episodes to perform
        print_every (int): How often to print the episode information

    Returns:
        (array_like, array_like): Scores and times that the agent took per episode
    """
    scores = np.zeros(total_episodes)
    times = np.zeros(total_episodes)

    for i in range(total_episodes):
        start_time = time.time()
        scores[i] = do_episode(environment, brain_name, agent, learn)
        times[i] = time.time() - start_time

        # Log data
        ep = i + 1
        wandb.log({
            "episode": ep,
            "score": scores[i],
            "time": times[i]
        })

        if ep % print_every == 0:
            print(
                f"{agent.agent_name()} :: ({ep}/{total_episodes}) AVG {np.average(scores[max(0, i - print_every):ep])}")

        # Early stop
        if np.average(scores[max(0, i - 100):ep]) >= 0.5:
            print(f'Reached target at episode {ep}')
            break

    return scores, times


def do_episode(environment, brain_name, agent, learn=True):
    """Performs a single episode using the given environment and agent

    Args:
        environment (env): Environment that will perform the simulation
        agent (Agent): Agent that will traverse the environment

    Returns:
        (float, int): Total score and steps of the episode
    """
    # episode_score = 0
    agents_score = np.zeros((2, ))
    env_info = environment.reset(train_mode=learn)[brain_name]

    # Start the agent
    state = env_info.vector_observations
    next_action = agent.start(state)

    # Take the first action
    env_info = environment.step(next_action)[brain_name]

    while not env_info.local_done[0]:
        # Take a step from the agent
        rewards = env_info.rewards
        # episode_score += np.max(rewards)
        agents_score += rewards
        state = env_info.vector_observations

        next_action = agent.step(state, rewards, learn=learn)

        # Perform action
        env_info = environment.step(next_action)[brain_name]

    # Register last reward to the agent
    rewards = env_info.rewards
    # episode_score += np.max(rewards)
    agents_score += rewards
    agent.end(rewards)

    return np.max(agents_score)


if __name__ == '__main__':
    graphics = False

    # Load environment and get initial brain
    env = UnityEnvironment(file_name='Tennis/Tennis.x86_64', no_graphics=not graphics)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # Initialize environment for use of the agent
    env_info = env.reset(train_mode=graphics)[brain_name]
    action_space = brain.vector_action_space_size
    state_space = env_info.vector_observations.shape

    with wandb.init(project='nanorl-p3', entity='jorelmb') as run:
        episodes = wandb.config.episodes
        print_every = 100

        agent = create_agent(state_space[1], action_space, **wandb.config)
        scores, _ = do_experiment(env, brain_name, agent, episodes, print_every, learn=not graphics)
        agent_name = agent.agent_name()

        # Print scores
        print(f"Last score: {scores[-1]:.4f}, Average: {np.average(scores[max(0, len(scores) - print_every):]):.4f}")

        # Save agent
        torch.save(agent, f"agents/{agent_name}-{run.id}.pt")

    # Close environment
    env.close()
