import logging
import os
import shutil
import subprocess
import sys

import transformers
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
    # show a library that was installed due to requirements.txt
    logger.info(f"transformers: {transformers}")
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

    # importing internal and external dependencies
    from external_dependency import lib1  # noqa: F401
    from internal_dependency import lib2  # noqa: F401

    logBefore(args)

    # copy the entire input dir to the output dir
    outputDataDir = os.path.join(args.output_data_dir, args.current_host)
    shutil.copytree(args.input_dir, f"{outputDataDir}/input_dir_copy")
    # copy state dir
    shutil.copytree(args.state, f"{outputDataDir}/state_copy")
    # cteaye a file
    open(f"{outputDataDir}/output_data_dir", "wt").write("output_data_dir")

    # create one file in the output dir
    outputDir = os.path.join(args.output_dir, args.current_host)
    os.makedirs(outputDir, exist_ok=True)
    open(f"{outputDir}/output_dir", "wt").write("output_dir")

    # create one file in the output model dir
    modelDir = os.path.join(args.model_dir, args.current_host)
    os.makedirs(modelDir, exist_ok=True)
    open(f"{modelDir}/model_dir", "wt").write("model_dir")

    # delete other instances state, write file in instance state folder.
    stateDir = algo_lib.initMultiWorkersState(args)
    open(f"{stateDir}/state_{args.current_host}", "wt").write(
        f"state_{args.current_host}"
    )
    algo_lib.markCompleted(args)

    # just to show the final directory structue
    logger.info("finished!")
    logAfter(args)
