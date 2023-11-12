
import logging

verbose = False

logger = logging.getLogger("myLog")

FileOutputHandler = logging.FileHandler('logs.log', mode='w')

lvl = logging.DEBUG if verbose else logging.INFO
logger.setLevel(level=lvl)

formatter = logging.Formatter(fmt='%(levelname)s: %(message)s')

FileOutputHandler.setFormatter(formatter)

logger.addHandler(FileOutputHandler)
logger.propagate = False

