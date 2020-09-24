import logging
import subprocess
import sys

from worker_toolkit import worker_lib

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
    worker_config = worker_lib.WorkerConfig(False)
    listDir(worker_config.channel_data)
    listDir(worker_config.channel_bucket)
