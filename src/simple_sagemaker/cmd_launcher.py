import logging
import subprocess
import sys
from pathlib import Path

from worker_toolkit import worker_lib

logger = logging.getLogger(__name__)


def worker():
    logging.basicConfig(stream=sys.stdout)

    # parse the arguments
    worker_config = worker_lib.WorkerConfig()

    cmd_line = worker_config.hps["SSM_CMD_LINE"]
    logger.info(f"Launching: {cmd_line}")
    ran_cmd = subprocess.run(cmd_line, shell=True)
    if ran_cmd.returncode == 0:
        # mark the task as completed
        worker_config.markCompleted()

    logger.info(f"finished with {ran_cmd.returncode} return code!")
    return ran_cmd.returncode


if __name__ == "__main__":
    worker()
