
import json
import traceback as tb
from os.path import join

import matplotlib.pyplot as plt
from unityagents import UnityEnvironment

from train_agent import *
from utils import logger


projects_dir = "C:\\Users\\pathr\\PycharmProjects\\"
app_path = "Tennis_Windows_x86_64\\Tennis.exe"
app_path = join(projects_dir, app_path)
env = UnityEnvironment(file_name=app_path)
params = {
    'STATE_SIZE': 8*3,
    'ACTION_SIZE': 2,
    'SEED': 8675309,
    'LR_ACTOR': 1e-4,
    'LR_CRITIC': 1e-3,
    'BUFFER_SIZE': int(1e5),
    'TRAIN_BATCH_SIZE': 512,
    'DISCOUNT_FACTOR': 0.99,
    'TAU': 1e-3,
    'WEIGHT_DECAY': 0.0001,
    'UPDATE_ITERATION': 10,
    'NUM_UPDATES_PER_INTERVAL': 10,
    'NOISE_DECAY': 1,
    'NUM_EPISODES': 2500,
    'MAX_TIMESTEPS': 50,
    'THRESHOLD': 2.0,
}
logger.info('PARAMETERS:\n%s', json.dumps(params, indent=4))

try:
    agent1, agent2 = create_agents(
        state_size=params['STATE_SIZE'],
        action_size=params['ACTION_SIZE'],
        seed=params['SEED'],
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

    scores1: List[float] = []
    scores2: List[float] = []
    avg_scores1: List[float] = []
    avg_scores2: List[float] = []
    scores_window1 = deque(maxlen=100)
    scores_window2 = deque(maxlen=100)

    # loop through each episode
    for i_episode in range(1, params['NUM_EPISODES'] + 1):
        score1, score2 = train_episode(
            env,
            i_episode,
            agent1,
            agent2,
            params['MAX_TIMESTEPS'],
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

        score_str = 'Episode: {}\tAvg Scores: [{:.2f}, {:.2f}]\tScores: [{:.2f}, {:.2f}]'
        out_s = score_str.format(i_episode, avg_score1, avg_score2, score1, score2)

        logger.info(out_s)
        print(out_s)

        # If the avg score of latest window is above threshold, then stop training and save model
        if np.max([avg_score1, avg_score2]) >= params['THRESHOLD']:
            solved_str = '\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f} | {:.2f}'
            logger.info(solved_str.format(i_episode - 100, avg_score1, avg_score2))
            torch.save(agent1.actor_model_current.state_dict(), 'models\\checkpoint1_actor.pth')
            torch.save(agent1.critic_model_current.state_dict(), 'models\\checkpoint1_critic.pth')
            torch.save(agent2.actor_model_current.state_dict(), 'models\\checkpoint2_actor.pth')
            torch.save(agent2.critic_model_current.state_dict(), 'models\\checkpoint2_critic.pth')
            break

except Exception:
    logger.critical(tb.format_exc())

except KeyboardInterrupt:
    logger.critical(tb.format_exc())

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores1)), scores1)
plt.plot(np.arange(len(avg_scores1)), avg_scores1)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.title('Agent 1')
plt.show()

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores2)), scores2)
plt.plot(np.arange(len(avg_scores2)), avg_scores2)
plt.ylabel('Avg Score')
plt.xlabel('Episode #')
plt.title('Agent 2')
plt.show()

logger.info('Exiting...')
env.close()
exit(0)
