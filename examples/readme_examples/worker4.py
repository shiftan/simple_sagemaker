import logging
import subprocess
import sys

from task_toolkit import algo_lib

logger = logging.getLogger(__name__)


def listDir(path):
    logger.info(f"*** START listing files in {path}")
    logger.info(
        subprocess.run(
            ["ls", "-la", "-R", path], stdout=subprocess.PIPE, universal_newlines=True
        ).stdout
    )
    logger.info(f"*** END file listing {path}")


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout)
    algo_lib.setDebugLevel()
    args = algo_lib.parseArgs()
    listDir(args.input_data)
    listDir(args.input_bucket)
