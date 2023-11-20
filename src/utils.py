
import logging
import traceback as tb

import matplotlib.pyplot as plt
import numpy as np

verbose = False

logger = logging.getLogger("myLog")

FileOutputHandler = logging.FileHandler('logs.log', mode='w')

lvl = logging.DEBUG if verbose else logging.INFO
logger.setLevel(level=lvl)

formatter = logging.Formatter(fmt='%(levelname)s: %(message)s')

FileOutputHandler.setFormatter(formatter)

logger.addHandler(FileOutputHandler)
logger.propagate = False


def save_plots(scores1, avg_scores1, scores2, avg_scores2, file="models\\agents.png"):

    try:
        # plot the scores
        fig = plt.figure(figsize=(14, 6))
        ax1 = fig.add_subplot(121)
        ax1.plot(np.arange(len(scores1)), scores1)
        ax1.plot(np.arange(len(avg_scores1)), avg_scores1)
        ax1.set_ylabel('Score')
        ax1.set_xlabel('Episode #')
        ax1.set_title('Agent 1')
        ax2 = fig.add_subplot(122)
        ax2.plot(np.arange(len(scores2)), scores2)
        ax2.plot(np.arange(len(avg_scores2)), avg_scores2)
        ax2.set_ylabel('Score')
        ax2.set_xlabel('Episode #')
        ax2.set_title('Agent 2')
        plt.savefig(file)
    except Exception:
        logger.error(tb.format_exc())
