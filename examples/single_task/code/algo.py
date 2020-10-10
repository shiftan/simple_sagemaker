import logging
import os
import shutil
import subprocess
import sys

import transformers
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


def logBefore(worker_config):
    # show the given arguments and environment
    logger.info(f"Argv: {sys.argv}")
    logger.info(f"Env: {os.environ}")
    # show a library that was installed due to requirements.txt
    logger.info(f"transformers: {transformers}")
    # just to show the initial directory structue
    listDir("/opt/ml")
    listDir(worker_config.state)


def logAfter(worker_config):
    # just to show the final directory structue
    listDir("/opt/ml")
    listDir(worker_config.state)


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout)
    logger.info("Starting algo...")

    # parse the arguments
    worker_config = worker_lib.WorkerConfig()

    # importing internal and external dependencies
    from external_dependency import lib1  # noqa: F401
    from internal_dependency import lib2  # noqa: F401

    logBefore(worker_config)

    # copy the entire input dir to the output dir
    output_data_dir = os.path.join(
        worker_config.output_data_dir, worker_config.current_host
    )
    shutil.copytree(worker_config.input_dir, f"{output_data_dir}/input_dir_copy")
    # copy state dir
    shutil.copytree(worker_config.state, f"{output_data_dir}/state_copy")
    # cteaye a file
    open(f"{output_data_dir}/output_data_dir", "wt").write("output_data_dir")

    # create one file in the output dir
    output_dir = os.path.join(worker_config.output_dir, worker_config.current_host)
    os.makedirs(output_dir, exist_ok=True)
    open(f"{output_dir}/output_dir", "wt").write("output_dir")

    # create one file in the output model dir
    modelDir = os.path.join(worker_config.model_dir, worker_config.current_host)
    os.makedirs(modelDir, exist_ok=True)
    open(f"{modelDir}/model_dir", "wt").write("model_dir")

    open(
        f"{worker_config.instance_state}/state_{worker_config.current_host}", "wt"
    ).write(f"state_{worker_config.current_host}")

    # just to show the final directory structue
    logger.info("finished!")
    logAfter(worker_config)
    # The task is marked as completed
