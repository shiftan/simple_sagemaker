import logging
import sys

from task_toolkit import algo_lib

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout)
    algo_lib.setDebugLevel()
    logger.info("Starting worker...")

    args = algo_lib.parseArgs()
    stateDir = algo_lib.initMultiWorkersState(args)

    logger.info("Doing something...")

    algo_lib.markCompleted(args)
    logger.info("finished!")
