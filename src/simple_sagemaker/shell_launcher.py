import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from worker_toolkit import worker_lib

logger = logging.getLogger(__name__)


def worker():
    logging.basicConfig(stream=sys.stdout)

    # Parse the arguments + initialize state
    worker_config = worker_lib.WorkerConfig()

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
        shell_cmd = subprocess.run(cmd_line, shell=True, executable="/bin/bash")

    logger.info(f"finished with {shell_cmd.returncode} return code!")

    # wait_for_state_sync(worker_config)
    return shell_cmd.returncode


def wait_for_state_sync(worker_config):
    max_secs = 60 * 5  # 5 mins max
    wait_secs = 5
    state_path = Path(worker_config.state)
    max_change_time = max(map(os.path.getmtime, state_path.rglob("*")))
    for i in range(max_secs // wait_secs):
        time.sleep(wait_secs)
        new_max = max(map(os.path.getmtime, state_path.rglob("*")))
        if new_max == max_change_time:
            return
        max_change_time = new_max
    logger.warning(
        f"It seems like sage maker is still uploading after {max_secs} secs..."
    )


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    retcode = worker()
    sys.exit(retcode)
