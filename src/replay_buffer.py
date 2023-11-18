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
                "state1",
                "state2",
                "action1",
                "action2",
                "reward1",
                "reward2",
                "next_state1",
                "next_state2",
                "done1",
                "done2",
            ],
        )
        self.seed = seed
        random.seed(seed)
        self.device = device

    def add_experience(
        self,
        state1: Iterable[float],
        state2: Iterable[float],
        action1: float,
        action2: float,
        reward1: float,
        reward2: float,
        next_state1: Iterable[float],
        next_state2: Iterable[float],
        done1: bool,
        done2: bool,
    ):
        """ Adds new experience to the buffer
        :param stateN: Iterable[float] of state_size dimensions containing state space at time T
        :param actionN: int Chosen action index at time T
        :param rewardN: Reward received from taking action A with state S at time T
        :param next_stateN: Iterable[float] of state_size dimensions containing state space at time T+1
        :param doneN: boolean indicating episode done condition: True = done, False = not done
        """
        self.replay_buffer.append(
            self.experience(
                state1,
                state2,
                action1,
                action2,
                reward1,
                reward2,
                next_state1,
                next_state2,
                done1,
                done2,
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
        states1, states2 = ([], [])
        actions1, actions2 = ([], [])
        rewards1, rewards2 = ([], [])
        next_states1, next_states2 = ([], [])
        dones1, dones2 = ([], [])

        # Grab all the experiences and form them into a torch object
        [
            [
                states1.append(e.state1),
                states2.append(e.state2),
                actions1.append(e.action1),
                actions2.append(e.action2),
                rewards1.append(e.reward1),
                rewards2.append(e.reward2),
                next_states1.append(e.next_state1),
                next_states2.append(e.next_state2),
                dones1.append(e.done1),
                dones2.append(e.done2),
            ]
            for e in exps if e is not None
        ]

        states1 = torch.from_numpy(np.vstack(states1)).float().to(self.device)
        states2 = torch.from_numpy(np.vstack(states2)).float().to(self.device)
        actions1 = torch.from_numpy(np.vstack(actions1)).float().to(self.device)
        actions2 = torch.from_numpy(np.vstack(actions2)).float().to(self.device)
        rewards1 = torch.from_numpy(np.vstack(rewards1)).float().to(self.device)
        rewards2 = torch.from_numpy(np.vstack(rewards2)).float().to(self.device)
        next_states1 = torch.from_numpy(np.vstack(next_states1)).float().to(self.device)
        next_states2 = torch.from_numpy(np.vstack(next_states2)).float().to(self.device)
        dones1 = torch.from_numpy(np.vstack(dones1).astype(np.uint8)).float().to(self.device)
        dones2 = torch.from_numpy(np.vstack(dones2).astype(np.uint8)).float().to(self.device)

        return states1, states2, actions1, actions2, rewards1, rewards2, next_states1, next_states2, dones1, dones2

    def has_enough_data(self) -> bool:
        """
        Return whether replay buffer has enough data for training

        :return: True = buffer has enough data, False = not enough data
        """
        return len(self) > self.batch_size

    def __len__(self) -> int:
        """Return the current size of internal memory."""
        return len(self.replay_buffer)

    def clear_buffer(self):
        self.replay_buffer.clear()
