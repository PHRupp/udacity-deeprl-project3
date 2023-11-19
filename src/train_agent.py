
from collections import deque
from typing import List

import numpy as np
import torch

from base_agent import BaseAgent
from ddpg_agent import DDPGAgent
from utils import logger


def create_agents(
    state_size,
    action_size,
    seed,
    lr_actor,
    lr_critic,
    buffer_size,
    train_batch_size,
    discount_factor,
    tau,  # update of best parameters
    update_iteration,
    weight_decay,
    num_updates_per_interval,
    noise_decay,
):
    agent1 = DDPGAgent(
        state_size=state_size,
        action_size=action_size,
        seed=seed,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        buffer_size=buffer_size,
        train_batch_size=train_batch_size,
        discount_factor=discount_factor,
        TAU=tau,  # update of best parameters
        update_iteration=update_iteration,
        weight_decay=weight_decay,
        num_updates_per_interval=num_updates_per_interval,
        noise_decay=noise_decay,
    )
    agent2 = DDPGAgent(
        state_size=state_size,
        action_size=action_size,
        seed=int(seed / 2.0),
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        buffer_size=buffer_size,
        train_batch_size=train_batch_size,
        discount_factor=discount_factor,
        TAU=tau,  # update of best parameters
        update_iteration=update_iteration,
        weight_decay=weight_decay,
        num_updates_per_interval=num_updates_per_interval,
        noise_decay=noise_decay
    )

    # share the replay buffer
    agent2.set_replay_buffer(agent1.replay_buffer)

    return agent1, agent2


def train_episode(
    env,
    i_episode,
    agent1: BaseAgent,
    agent2: BaseAgent,
    max_timesteps: int = 1000,
    brain: str = 'TennisBrain',
):
    state_brain = env.reset()
    state1, state2 = state_brain[brain].__dict__['vector_observations']
    score1 = 0
    score2 = 0

    # loop through all time steps within episode
    for t in range(max_timesteps):
        action1 = agent1.act(state1, add_noise=True)
        action2 = agent2.act(state2, add_noise=True)

        state_brain = env.step(vector_action=[action1, action2])
        next_state1, next_state2 = state_brain[brain].__dict__['vector_observations']
        reward1, reward2 = state_brain[brain].__dict__['rewards']
        done1, done2 = state_brain[brain].__dict__['local_done']

        logger.debug(
            '{}:{} --- {} {:.2f} --- {} {:.2f}'.format(
                i_episode,
                t,
                str(action1),
                reward1,
                str(action2),
                reward2,
            )
        )

        agent1.step(state1, action1, reward1, next_state1, done1)
        agent2.step(state2, action2, reward2, next_state2, done2)

        state1 = next_state1
        state2 = next_state2

        score1 += reward1
        score2 += reward2

        # Done reached
        if done1 or done2:
            break

    agent1.reset()
    agent2.reset()

    return score1, score2


def train(
    env,
    agent1: BaseAgent,
    agent2: BaseAgent,
    num_episodes: int = 2000,
    max_timesteps: int = 1000,
    threshold: float = 30.0,
    brain: str = 'TennisBrain',
) -> List[float]:
    """ code to train an agent
    :param env: environment that agent will train in
    :param agent: (BaseAgent) agent to be trained
    :param num_episodes: (int) maximum number of training episodes
    :param max_timesteps: (int) maximum number of time steps per episode
    :param eps_start: (float) starting value of epsilon assuming epsilon-greedy approach
    :param eps_end: (float) absolute minimum value of epsilon
    :param eps_decay: (float) multiplicative factor (per episode) for decreasing epsilon
    :param threshold: (float) required threshold score required for training to be complete
    :return:
    """
    scores1: List[float] = []
    scores2: List[float] = []
    avg_scores1: List[float] = []
    avg_scores2: List[float] = []
    scores_window1 = deque(maxlen=100)
    scores_window2 = deque(maxlen=100)

    # loop through each episode
    for i_episode in range(1, num_episodes + 1):
        score1, score2 = train_episode(
            env,
            i_episode,
            agent1,
            agent2,
            max_timesteps,
            brain,
        )

        # save the scores
        scores_window1.append(score1)
        scores_window2.append(score2)
        scores1.append(score1)
        scores2.append(score2)

        avg_score1 = np.mean(scores_window1)
        avg_score2 = np.mean(scores_window2)
        avg_scores1.append(avg_score1)
        avg_scores2.append(avg_score2)

        score_str = 'Episode {}\tAvg Scores: {:.2f} | {:.2f}'
        out_s = score_str.format(i_episode, avg_score1, avg_score2)

        logger.info(out_s)
        print(out_s)

        # If the avg score of latest window is above threshold, then stop training and save model
        if np.max([avg_score1, avg_score2]) >= threshold:
            solved_str = '\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f} | {:.2f}'
            logger.info(solved_str.format(i_episode - 100, avg_score1, avg_score2))
            torch.save(agent1.actor_model_current.state_dict(), 'models\\checkpoint1_actor.pth')
            torch.save(agent1.critic_model_current.state_dict(), 'models\\checkpoint1_critic.pth')
            torch.save(agent2.actor_model_current.state_dict(), 'models\\checkpoint2_actor.pth')
            torch.save(agent2.critic_model_current.state_dict(), 'models\\checkpoint2_critic.pth')
            break

    return scores1, scores2, avg_scores1, avg_scores2
