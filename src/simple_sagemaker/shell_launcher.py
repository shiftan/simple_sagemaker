import logging
import os
import shutil
import subprocess
import sys

from worker_toolkit import worker_lib

logger = logging.getLogger(__name__)


def worker():
    logging.basicConfig(stream=sys.stdout)

    # Parse the arguments + initialize state
    worker_config = worker_lib.WorkerConfig()

    # Fill the environment varaible with missing parameters
    os.environ["SSM_STATE"] = worker_config.state
    os.environ["SSM_INSTANCE_STATE"] = worker_config.instance_state

    # Delete the current file + toolkit as both got injected
    os.remove(__file__)
    shutil.rmtree("./worker_toolkit")

    # Run the shell / cmd line command
    if "SSM_CMD_LINE" in worker_config.hps:
        cmd_line = worker_config.hps["SSM_CMD_LINE"]
        logger.info(f"Launching: {cmd_line}")
        shell_cmd = subprocess.run(cmd_line)
    elif "SSM_SHELL_CMD_LINE" in worker_config.hps:
        cmd_line = worker_config.hps["SSM_SHELL_CMD_LINE"]
        logger.info(f"Launching a shell: {cmd_line}")
        shell_cmd = subprocess.run(cmd_line, shell=True)
    # Mark the job as completed if exit code is 0
    if shell_cmd.returncode == 0:
        # mark the task as completed
        worker_config.markCompleted()

    logger.info(f"finished with {shell_cmd.returncode} return code!")
    return shell_cmd.returncode


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    worker()
