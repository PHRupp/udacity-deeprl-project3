
import json
import traceback as tb
from collections import deque
from os.path import join
from typing import List

import numpy as np
import pandas as pd
import torch
from unityagents import UnityEnvironment

from maddpg_agent import MADDDPGAgent
from replay_buffer import ReplayBuffer
from train_agent import run_episode
from utils import logger, save_plots


projects_dir = "C:\\Users\\pathr\\PycharmProjects\\"
app_path = "Tennis_Windows_x86_64\\Tennis.exe"
app_path = join(projects_dir, app_path)
env = UnityEnvironment(file_name=app_path)
params = {
    'STATE_SIZE': 8*3,
    'ACTION_SIZE': 2,
    'SEED': 8675309,
    'LR_ACTOR': 0.0005,
    'LR_CRITIC': 0.0025,
    'BUFFER_SIZE': int(1e5),
    'TRAIN_BATCH_SIZE': 256,
    'DISCOUNT_FACTOR': 0.99,
    'TAU': 1e-3,
    'WEIGHT_DECAY': 0.0001,
    'UPDATE_ITERATION': 1,
    'NUM_UPDATES_PER_INTERVAL': 10,
    'NOISE_DECAY': 0.999,
    'NUM_EPISODES': 100000,
    'MAX_TIMESTEPS': 1000,
    'THRESHOLD': 2.0,
    'PLOT_ITER': 100,
    'RANDOM_THRESH': 200,
}
logger.info('PARAMETERS:\n%s', json.dumps(params, indent=4))

scores1: List[float] = []
scores2: List[float] = []
avg_scores1: List[float] = []
avg_scores2: List[float] = []
scores_window1 = deque(maxlen=100)
scores_window2 = deque(maxlen=100)

torch_device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(torch_device)

try:
    maddpg = MADDDPGAgent(
        state_size=params['STATE_SIZE'],
        action_size=params['ACTION_SIZE'],
        seed=params['SEED'],
        device=device,
        lr_actor=params['LR_ACTOR'],
        lr_critic=params['LR_CRITIC'],
        buffer_size=params['BUFFER_SIZE'],
        train_batch_size=params['TRAIN_BATCH_SIZE'],
        discount_factor=params['DISCOUNT_FACTOR'],
        tau=params['TAU'],  # update of best parameters
        update_iteration=params['UPDATE_ITERATION'],
        weight_decay=params['WEIGHT_DECAY'],
        num_updates_per_interval=params['NUM_UPDATES_PER_INTERVAL'],
        noise_decay=params['NOISE_DECAY'],
    )

    # start from negative number to collect lots of random data before training begins
    update_iter = 0

    # replay buffer for storing SARS
    replay_buffer = ReplayBuffer(
        params['BUFFER_SIZE'],
        params['TRAIN_BATCH_SIZE'],
        params['SEED'],
        device,
    )

    # loop through each episode
    for i_episode in range(1, params['NUM_EPISODES'] + 1):

        run_random = (i_episode < params['RANDOM_THRESH'])

        if update_iter == params['UPDATE_ITERATION']:
            update_iter = 0
            maddpg.step(replay_buffer)
            #maddpg.agent1.step(replay_buffer)
            #maddpg.agent2.step(replay_buffer)
            #replay_buffer.clear_buffer()

        update_iter += 1
        score1, score2 = run_episode(
            env,
            i_episode,
            maddpg,
            replay_buffer,
            params['MAX_TIMESTEPS'],
            random=run_random,
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

        score_str = 'Episode: {}\tAvg Scores: [{:.3f}, {:.3f}]\tScores: [{:.2f}, {:.2f}]\tRandom: {}'
        out_s = score_str.format(i_episode, avg_score1, avg_score2, score1, score2, run_random)

        logger.info(out_s)
        print(out_s)

        if (i_episode % params['PLOT_ITER']) == 0:
            save_plots(scores1, avg_scores1, scores2, avg_scores2)

        # If the avg score of latest window is above threshold, then stop training and save model
        if np.max([avg_score1, avg_score2]) >= params['THRESHOLD']:
            solved_str = '\nEnvironment solved in {:d} episodes!\tAverage Score: {:.3f} | {:.3f}'
            logger.info(solved_str.format(i_episode - 100, avg_score1, avg_score2))
            torch.save(maddpg.agent1.actor_model_current.state_dict(), 'models\\checkpoint1_actor.pth')
            torch.save(maddpg.agent1.critic_model_current.state_dict(), 'models\\checkpoint1_critic.pth')
            torch.save(maddpg.agent2.actor_model_current.state_dict(), 'models\\checkpoint2_actor.pth')
            torch.save(maddpg.agent2.critic_model_current.state_dict(), 'models\\checkpoint2_critic.pth')
            break

except Exception:
    logger.critical(tb.format_exc())

except KeyboardInterrupt:
    logger.critical(tb.format_exc())

pd.DataFrame.from_dict(
    {
        'scores1': scores1,
        'scores2': scores2,
        'avg_scores1': avg_scores1,
        'avg_scores2': avg_scores2,
    },
).to_feather('models\\results.feather')

save_plots(scores1, avg_scores1, scores2, avg_scores2, file='models\\final_agents.png')

logger.info('Exiting...')
env.close()
exit(0)
