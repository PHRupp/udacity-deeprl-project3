
import numpy as np
import torch
import torch.nn.functional as F

from ddpg_agent import DDPGAgent
from utils import logger


class MADDDPGAgent:
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
        tau=1e-3,  # update of best parameters
        update_iteration=4,
        weight_decay=0,
        num_updates_per_interval=1,
        noise_decay=0.995,
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
        self.num_updates_per_interval = num_updates_per_interval
        self.noise_iteration = 1
        self.agent1 = DDPGAgent(
            state_size=state_size,
            action_size=action_size,
            seed=seed,
            device=device,
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
            agent_name='agent1',
        )
        self.agent2 = DDPGAgent(
            state_size=state_size,
            action_size=action_size,
            seed=int(seed / 2.0),
            device=device,
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
            agent_name='agent2',
        )
    
    def step(self, replay_buffer):
        """ Step the agent which may update the underlying model using SARSA data

        :param replay_buffer: ReplayBuffer storing the SARS experiences
        """

        # update model with random replays from buffer
        if replay_buffer.has_enough_data():
            s1, s2, a1, a2, r1, r2, ns1, ns2, d1, d2 = replay_buffer.sample()
            for i in range(self.num_updates_per_interval):
                self.model_update(s1, s2, a1, a2, r1, r2, ns1, ns2, d1, d2)

            # Reduce the noise
            self.agent1.noise.decay_noise_params(self.noise_iteration)
            self.agent2.noise.decay_noise_params(self.noise_iteration)
            self.noise_iteration += 1
            self.agent1.noise.reset()
            self.agent2.noise.reset()
            logger.debug("MODEL UPDATE!")
        else:
            logger.warn('NOT ENOUGH BATCH SIZE!')

        return

    def act(self, state1, state2, add_noise: bool = True):
        """Returns the action selected

        :param state1: Iterable[float] of state_size dimensions containing state space at time T for agent1
        :param state2: Iterable[float] of state_size dimensions containing state space at time T for agent2
        :param add_noise: (bool) epsilon value for randomly selecting action
        """
        action1 = self.agent1.act(state1, add_noise=add_noise)
        action2 = self.agent2.act(state2, add_noise=add_noise)
        return action1, action2

    def model_update_agent1(self, s1, s2, a1, a2, r1, ns1, ns2, d1, an1, an2, ap1, ap2):

        # max predicted Q-vals from best model
        Q_best_next = self.agent1.critic_model_best(ns1, ns2, an1, an2)
        #Q_best_next = self.agent1.critic_model_best(ns1, an1)

        # Q-best-vals for current states
        Q_best = r1 + (self.agent1.discount_factor * Q_best_next * (1 - d1))

        # expected Q-vals from current model
        Q_expected = self.agent1.critic_model_current(s1, s2, a1, a2)
        #Q_expected = self.agent1.critic_model_current(s1, a1)

        # get loss using mean squared error
        critic_loss = F.mse_loss(Q_expected, Q_best)

        # reduce loss using optimizer along gradient descent
        self.agent1.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm(self.agent1.critic_model_current.parameters(), 1)
        self.agent1.critic_optimizer.step()

        # Compute actor loss
        actor_loss = -self.agent1.critic_model_current(s1, s2, ap1, ap2).mean()
        #actor_loss = -self.agent1.critic_model_current(s1, ap1).mean()

        # Minimize the loss for the actor
        self.agent1.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.agent1.actor_optimizer.step()

        # Update the models
        self.agent1.soft_update_all()

    def model_update_agent2(self, s1, s2, a1, a2, r2, ns1, ns2, d2, an1, an2, ap1, ap2):

        # max predicted Q-vals from best model
        Q_best_next = self.agent2.critic_model_best(ns1, ns2, an1, an2)
        #Q_best_next = self.agent2.critic_model_best(ns2, an2)

        # Q-best-vals for current states
        Q_best = r2 + (self.agent2.discount_factor * Q_best_next * (1 - d2))

        # expected Q-vals from current model
        Q_expected = self.agent2.critic_model_current(s1, s2, a1, a2)
        #Q_expected = self.agent2.critic_model_current(s2, a2)

        # get loss using mean squared error
        critic_loss = F.mse_loss(Q_expected, Q_best)

        # reduce loss using optimizer along gradient descent
        self.agent2.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm(self.agent2.critic_model_current.parameters(), 1)
        self.agent2.critic_optimizer.step()

        # Compute actor loss
        actor_loss = -self.agent2.critic_model_current(s1, s2, ap1, ap2).mean()
        #actor_loss = -self.agent2.critic_model_current(s2, ap2).mean()

        # Minimize the loss for the actor
        self.agent2.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.agent2.actor_optimizer.step()

        # Update the models
        self.agent2.soft_update_all()

    def model_update(self, s1, s2, a1, a2, r1, r2, ns1, ns2, d1, d2):
        """Update value parameters using given batch of experience tuples.

        :param s1: Iterable[float] of state_size dimensions containing state space at time T for agent1
        :param s2: Iterable[float] of state_size dimensions containing state space at time T for agent2
        :param a1: Iterable[float] Chosen action index at time T for agent1
        :param a2: Iterable[float] Chosen action index at time T for agent2
        :param r1: Reward received from taking action A with state S at time T for agent1
        :param r2: Reward received from taking action A with state S at time T for agent2
        :param ns1: Iterable[float] of state_size dimensions containing state space at time T+1 for agent1
        :param ns2: Iterable[float] of state_size dimensions containing state space at time T+1 for agent2
        :param d1: boolean indicating episode done condition: True = done, False = not done for agent1
        :param d2: boolean indicating episode done condition: True = done, False = not done for agent2
        """

        # predict actions needed for computing actor loss
        actions_pred1 = self.agent1.actor_model_current(s1).detach()
        actions_pred2 = self.agent2.actor_model_current(s2).detach()

        # max predicted Q-vals from best model
        actions_next1 = self.agent1.actor_model_best(ns1).detach()
        actions_next2 = self.agent2.actor_model_best(ns2).detach()

        self.model_update_agent1(s1, s2, a1, a2, r1, ns1, ns2, d1, actions_next1, actions_next2, actions_pred1, actions_pred2)
        self.model_update_agent2(s1, s2, a1, a2, r2, ns1, ns2, d2, actions_next1, actions_next2, actions_pred1, actions_pred2)
