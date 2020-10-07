import logging
import os
import shutil
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


def logBefore(worker_config):
    # show the given arguments and environment
    logger.info(f"Argv: {sys.argv}")
    logger.info(f"Env: {os.environ}")
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
    logBefore(worker_config)

    output_data_dir = os.path.join(
        worker_config.output_data_dir, worker_config.current_host
    )

    # create some data in the state dir
    if worker_config.hps["stage"] == 1:
        # put some files in the state directory
        for i in range(10):
            open(
                f"{worker_config.instance_state}/state_{worker_config.current_host}_{i+1}",
                "wt",
            ).write("state")

        # put something in the model
        modelDir = os.path.join(worker_config.model_dir, worker_config.current_host)
        os.makedirs(modelDir, exist_ok=True)
        open(f"{modelDir}/model_dir", "wt").write("model_dir")

    elif worker_config.hps["stage"] == 2:
        logger.info("Doing nothing...")

    # copy all input channels to the output dir
    for channel_name in worker_config.channels:
        input_dir = worker_config.__getattr__(f"channel_{channel_name}")
        shutil.copytree(input_dir, f"{output_data_dir}/{channel_name}_copy")
    shutil.copytree(worker_config.state, f"{output_data_dir}/state_copy")

    logger.info("finished!")
    logAfter(worker_config)
    # The task is marked as completed
