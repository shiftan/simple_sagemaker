import logging
import os
import subprocess
import sys

from worker_toolkit import worker_lib

logger = logging.getLogger(__name__)


def listDir(path, recursive=True):
    logger.info(f"*** START listing files in {path}")
    cmd_args = ["ls", "-la", path]
    if recursive:
        cmd_args.append("-R")
    process = subprocess.run(cmd_args, stdout=subprocess.PIPE, universal_newlines=True)
    logger.info(process.stdout)
    logger.info(f"*** END file listing {path}")


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logger.info("Starting ...")

    worker_config = worker_lib.WorkerConfig()

    print(os.environ)

    listDir("/opt/")
    open(os.environ["SSM_STATE"] + "/state", "wt").write("state")
    open(os.environ["SSM_OUTPUT"] + "/output", "wt").write("output")

    worker_config.markCompleted()
    # just to show the final directory structue
    logger.info("finished!")
