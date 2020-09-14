import logging
import os
import shutil
import subprocess
import sys

import transformers
from task_toolkit import algo_lib

logger = logging.getLogger(__name__)


def listDir(path, recursive=True):
    logger.info(f"*** START listing files in {path}")
    args = ["ls", "-la", path]
    if recursive:
        args.append("-R")
    process = subprocess.run(args, stdout=subprocess.PIPE, universal_newlines=True)
    logger.info(process.stdout)
    logger.info(f"*** END file listing {path}")


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
    output_data_dir = os.path.join(args.output_data_dir, args.current_host)
    shutil.copytree(args.input_dir, f"{output_data_dir}/input_dir_copy")
    # copy state dir
    shutil.copytree(args.state, f"{output_data_dir}/state_copy")
    # cteaye a file
    open(f"{output_data_dir}/output_data_dir", "wt").write("output_data_dir")

    # create one file in the output dir
    output_dir = os.path.join(args.output_dir, args.current_host)
    os.makedirs(output_dir, exist_ok=True)
    open(f"{output_dir}/output_dir", "wt").write("output_dir")

    # create one file in the output model dir
    modelDir = os.path.join(args.model_dir, args.current_host)
    os.makedirs(modelDir, exist_ok=True)
    open(f"{modelDir}/model_dir", "wt").write("model_dir")

    # delete other instances state, write file in instance state folder.
    state_dir = algo_lib.initMultiWorkersState(args)
    open(f"{state_dir}/state_{args.current_host}", "wt").write(
        f"state_{args.current_host}"
    )
    algo_lib.markCompleted(args)

    # just to show the final directory structue
    logger.info("finished!")
    logAfter(args)
