
from typing import Tuple

from maddpg_agent import MADDDPGAgent
from replay_buffer import ReplayBuffer
from utils import logger


def run_episode(
    env,
    i_episode,
    maddpg: MADDDPGAgent,
    replay_buffer: ReplayBuffer,
    max_timesteps: int = 1000,
    brain: str = 'TennisBrain',
) -> Tuple[float, float]:

    state_brain = env.reset()
    maddpg.reset()
    state1, state2 = state_brain[brain].__dict__['vector_observations']
    experiences = []
    score1, score2 = (0, 0)
    final_reward1, final_reward2 = (0, 0)

    # loop through all time steps within episode
    for t in range(max_timesteps):
        action1, action2 = maddpg.act(state1, state2, add_noise=True)

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

        # capture experiences
        final_reward1 = reward1
        final_reward2 = reward2
        experiences.append((state1, state2, action1, action2, reward1, reward2, next_state1, next_state2, done1, done2))

        # update the states and total scores
        state1 = next_state1
        state2 = next_state2
        score1 += reward1
        score2 += reward2

        # Done reached
        if done1 or done2:
            break
    """
    for s1, s2, a1, a2, r1, r2, ns1, ns2, d1, d2 in experiences:
        replay_buffer.add_experience(s1, s2, a1, a2, r1, r2, ns1, ns2, d1, d2)
    """
    # add all the experiences to the buffer but use accumulated discounted rewards
    G_t1 = final_reward1
    G_t2 = final_reward2
    for s1, s2, a1, a2, r1, r2, ns1, ns2, d1, d2 in reversed(experiences):
        G_t1 = r1 + maddpg.agent1.discount_factor * G_t1
        G_t2 = r2 + maddpg.agent2.discount_factor * G_t2
        replay_buffer.add_experience(s1, s2, a1, a2, G_t1, G_t2, ns1, ns2, d1, d2)

    return score1, score2
