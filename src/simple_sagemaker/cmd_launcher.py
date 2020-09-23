import logging
import sys
from pathlib import Path
import subprocess

from worker_toolkit import worker_lib

logger = logging.getLogger(__name__)


def listDir(path, ignore_patterns=[]):
    logger.info(f"*** START listing files in {path}")
    for file in sorted(Path(path).rglob("*")):
        if (not ignore_patterns) or all(
            [pattern not in str(file) for pattern in ignore_patterns]
        ):
            logger.info(f"[{['Dir ', 'File'][file.is_dir()]}] {file}")
    logger.info(f"*** END file listing {path}")


def worker():
    logging.basicConfig(stream=sys.stdout)

    # parse the arguments
    worker_config = worker_lib.WorkerConfig()
    listDir("/opt/ml")
    logger.info(f"Current directory: {Path('.').resolve()}")
    listDir(".")

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
