
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from base_agent import BaseAgent
from networks import ActorNetwork, CriticNetwork
from noise import OUNoise
from replay_buffer import ReplayBuffer
from utils import soft_update


class DDPGAgent(BaseAgent):
    """Interacts with and learns from the environment."""

    def __init__(
        self,
        state_size,
        action_size,
        seed,
        device,
        lr_actor=1e-4,
        lr_critic=1e-3,
        buffer_size=int(1e5),
        train_batch_size=64,
        discount_factor=0.99,
        TAU=1e-3,  # update of best parameters
        update_iteration=4,
        weight_decay=0,
        num_updates_per_interval=1,
        noise_decay=0.995,
        agent_name='',
    ):
        """Initialize the DQN agent

        :param state_size: number of dimensions within state space
        :param action_size: number of dimensions within action space
        :param seed: random seed
        :param device: device for computation
        :param lr_actor: learning rate of actor
        :param lr_critic: learning rate of critic
        :param buffer_size: total size of the replay buffer
        :param train_batch_size: size of the batch taken from buffer
        :param discount_factor: reward discount factor
        :param TAU: best model parameter updates
        :param update_iteration: update the model at every Nth iteration
        :param weight_decay: L2 weight decay?
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed
        random.seed(seed)
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.buffer_size = buffer_size
        self.train_batch_size = train_batch_size
        self.discount_factor = discount_factor
        self.TAU = TAU
        self.update_iteration = update_iteration
        self.weight_decay = weight_decay
        self.num_updates_per_interval = num_updates_per_interval
        self.noise_iteration = 1
        self.noise_decay = noise_decay
        self.agent_name = agent_name
        self.device = device

        # Actor Network
        self.actor_model_current = ActorNetwork(state_size, action_size, seed).to(self.device)
        self.actor_model_best = ActorNetwork(state_size, action_size, seed).to(self.device)
        self.actor_optimizer = optim.Adam(
            self.actor_model_current.parameters(),
            lr=lr_actor,
        )

        # Critic Network
        self.critic_model_current = CriticNetwork(state_size, action_size, seed).to(self.device)
        self.critic_model_best = CriticNetwork(state_size, action_size, seed).to(self.device)
        self.critic_optimizer = optim.Adam(
            self.critic_model_current.parameters(),
            lr=lr_critic,
            weight_decay=self.weight_decay,
        )

        # Initialize update iteration
        self.update_num = 0

        # Noise process
        self.noise = OUNoise(action_size, self.seed, decay=self.noise_decay)

    def step(self, replay_buffer: ReplayBuffer):
        """ Step the agent which may update the underlying model using SARSA data

        :param state: Iterable[float] of state_size dimensions containing state space at time T
        :param action: Iterable[float] Chosen action index at time T
        :param reward: Reward received from taking action A with state S at time T
        :param next_state: Iterable[float] of state_size dimensions containing state space at time T+1
        :param done: boolean indicating episode done condition: True = done, False = not done
        """
        """
        # store experiences in replay buffer
        self.replay_buffer.add_experience(
            state,
            action,
            reward,
            next_state,
            done
        )

        # learn at update_iteration
        self.update_num += 1
        if self.update_num == self.update_iteration:
            self.update_num = 0
        """
        # update model with random replays from buffer
        if replay_buffer.has_enough_data():
            s1, s2, a1, a2, r1, r2, ns1, ns2, d1, d2 = replay_buffer.sample()
            for i in range(self.num_updates_per_interval):
                self.model_update(
                    states=s1 if self.agent_name == 'agent1' else s2,
                    actions=a1 if self.agent_name == 'agent1' else a2,
                    rewards=r1 if self.agent_name == 'agent1' else r2,
                    next_states=ns1 if self.agent_name == 'agent1' else ns2,
                    dones=d1 if self.agent_name == 'agent1' else d2,
                )

            # Reduce the noise
            self.noise.decay_noise_params(self.noise_iteration)
            self.noise_iteration += 1
            self.noise.reset()

    def act(self, state, add_noise: bool = True):
        """Returns the action selected

        :param state: Iterable[float] of state_size dimensions containing state space at time T
        :param add_noise: (bool) epsilon value for randomly selecting action
        """

        # convert the numpy state into a torch expected format
        state = torch.from_numpy(state).float().to(self.device)

        # set model to evaluation mode
        self.actor_model_current.eval()

        # ensure all gradients are detached from graphs
        with torch.no_grad():

            # calculate the actions from the model and current state
            action = self.actor_model_current(state).cpu().data.numpy()

        # set model to train mode
        self.actor_model_current.train()

        # Epsilon-greedy action selection
        if add_noise:
            noise = self.noise.sample()
            #logger.debug("Noise: %s", str(noise))
            action += noise

        return np.clip(action, -1, 1)

    def model_update(self, states, actions, rewards, next_states, dones):
        """Update value parameters using given batch of experience tuples.

        :param experiences: SARSA named tuples
        """
        # Split into components
        #states, actions, rewards, next_states, dones = experiences

        # max predicted Q-vals from best model
        actions_next = self.actor_model_best(next_states)
        Q_best_next = self.critic_model_best(next_states, actions_next)

        # Q-best-vals for current states
        Q_best = rewards + (self.discount_factor * Q_best_next * (1 - dones))

        # expected Q-vals from current model
        Q_expected = self.critic_model_current(states, actions)

        # get loss using mean squared error
        critic_loss = F.mse_loss(Q_expected, Q_best)

        # reduce loss using optimizer along gradient descent
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm(self.critic_model_current.parameters(), 1)
        self.critic_optimizer.step()

        # Compute actor loss
        actions_pred = self.actor_model_current(states)
        actor_loss = -self.critic_model_current(states, actions_pred).mean()

        # Minimize the loss for the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update_all()

    def soft_update_all(self):
        soft_update(self.critic_model_current, self.critic_model_best, self.TAU)
        soft_update(self.actor_model_current, self.actor_model_best, self.TAU)
