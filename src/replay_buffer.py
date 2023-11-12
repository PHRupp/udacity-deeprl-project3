import random
from collections import namedtuple, deque
from typing import Iterable

import numpy as np
import torch


class ReplayBuffer:
    """Buffer to store agent experiences as tuples."""

    def __init__(self, action_size: int, buffer_size: int, batch_size: int, seed: int, device):
        """ Initialize the Replay buffer

        :param action_size: (int) number of dimensions for the action space
        :param buffer_size: (int) size of the entire buffer
        :param batch_size: (int) size of the batch used for training
        :param seed: (int) random seed
        :param device: (int) torch device
        """
        self.action_size = action_size
        self.replay_buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple(
            "Experience",
            field_names=[
                "state",
                "action",
                "reward",
                "next_state",
                "done",
            ],
        )
        self.seed = seed
        random.seed(seed)
        self.device = device

    def add_experience(
        self,
        state: Iterable[float],
        action: Iterable[float],
        reward: float,
        next_state: Iterable[float],
        done: bool,
    ):
        """ Adds new experience to the buffer
        :param state: Iterable[float] of state_size dimensions containing state space at time T
        :param action: int Chosen action index at time T
        :param reward: Reward received from taking action A with state S at time T
        :param next_state: Iterable[float] of state_size dimensions containing state space at time T+1
        :param done: boolean indicating episode done condition: True = done, False = not done
        """
        self.replay_buffer.append(
            self.experience(
                state,
                action,
                reward,
                next_state,
                done,
            )
        )

    def sample(self) -> tuple:
        """ Sample all experiences and grab a random set from them

        :return:
            tuple[
                torch array = states,
                torch array = actions,
                torch array = rewards,
                torch array = next_states,
                torch array = dones,
            ]
        """
        exps = random.sample(self.replay_buffer, k=self.batch_size)
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        # Grab all the experiences and form them into a torch object
        [
            [
                states.append(e.state),
                actions.append(e.action),
                rewards.append(e.reward),
                next_states.append(e.next_state),
                dones.append(e.done),
            ]
            for e in exps if e is not None
        ]

        states = torch.from_numpy(np.vstack(states)).float().to(self.device)
        actions = torch.from_numpy(np.vstack(actions)).float().to(self.device)
        rewards = torch.from_numpy(np.vstack(rewards)).float().to(self.device)
        next_states = torch.from_numpy(np.vstack(next_states)).float().to(self.device)
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float().to(self.device)

        return states, actions, rewards, next_states, dones

    def has_enough_data(self) -> bool:
        """
        Return whether replay buffer has enough data for training

        :return: True = buffer has enough data, False = not enough data
        """
        return len(self) > self.batch_size

    def __len__(self) -> int:
        """Return the current size of internal memory."""
        return len(self.replay_buffer)
