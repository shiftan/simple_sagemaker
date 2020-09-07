import logging
import os
import shutil
import subprocess
import sys

from task_toolkit import algo_lib

logger = logging.getLogger(__name__)


def listDir(path, recursive=True):
    logger.info(f"*** Listing files in {path}")
    args = ["ls", "-la", path]
    if recursive:
        args.append("-R")
    process = subprocess.run(args, stdout=subprocess.PIPE, universal_newlines=True)
    logger.info(process.stdout)


def logBefore(args):
    # show the given arguments and environment
    logger.info(f"Argv: {sys.argv}")
    logger.info(f"Env: {os.environ}")
    # just to show the initial directory structue
    listDir("/opt/ml")
    listDir(args.state)


def logAfter(args):
    # just to show the final directory structue
    listDir("/opt/ml")
    listDir(args.state)


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout)
    algo_lib.setDebugLevel()
    logger.info("Starting algo...")

    # parse the arguments
    args = algo_lib.parseArgs()
    logBefore(args)

    outputDataDir = os.path.join(args.output_data_dir, args.current_host)
    stateDir = algo_lib.initMultiWorkersState(args)

    # create some data in the state dir
    if args.hps["stage"] == 1:
        # put some files in the state directory
        for i in range(10):
            open(f"{stateDir}/state_{args.current_host}_{i+1}", "wt").write("state")

        # put something in the model
        modelDir = os.path.join(args.model_dir, args.current_host)
        os.makedirs(modelDir, exist_ok=True)
        open(f"{modelDir}/model_dir", "wt").write("model_dir")

    elif args.hps["stage"] == 2:
        logger.info("Doing nothing...")

    # copy all input channels to the output dir
    for channelName in args.channel_names:
        inputDir = args.__getattribute__(f"input_{channelName}")
        shutil.copytree(inputDir, f"{outputDataDir}/{channelName}_copy")
    shutil.copytree(args.state, f"{outputDataDir}/state_copy")

    algo_lib.markCompleted(args)

    logger.info("finished!")
    logAfter(args)
