import logging
import sys
import time
from pathlib import Path

# a library that was installed due to requirements.txt
import transformers  # noqa: F401

# importing an internal dependency
from internal_dependency import lib2  # noqa: F401
from task_toolkit import task_lib

logger = logging.getLogger(__name__)


def listDir(path, ignore_patterns=[]):
    logger.info(f"*** START listing files in {path}")
    for file in sorted(Path(path).rglob("*")):
        if (not ignore_patterns) or all(
            [pattern not in str(file) for pattern in ignore_patterns]
        ):
            logger.info(f"[{['Dir ', 'File'][file.is_dir()]}] {file}")
    logger.info(f"*** END file listing {path}")


def worker1(args, state_dir):
    # a library that is pre-installed in the docker image, as defined in the Dockerfile
    import pandas  # noqa: F401

    logger.info("{pandas} is pre-installed in this image")

    # update the state
    (Path(state_dir) / args.current_host).write_text(f"state_{args.current_host}")
    # "process" input data into model output
    for file in Path(args.input_data).rglob("*"):
        relp = file.relative_to(args.input_data)
        path = Path(args.model_dir) / (f"{relp}_proc_by_{args.current_host}")
        path.write_text(f"{file.read_text()} processed by {args.current_host}")
    # write to output dir
    (Path(args.output_data_dir) / f"output_{args.current_host}").write_text(
        f"output_{args.current_host}"
    )


def worker2(args, state_dir):
    # importing an external dependency
    from external_dependency import lib1  # noqa: F401

    logger.info("Score=10;")
    time.sleep(60)  # sleep to be able to see the two scores
    logger.info("Score=20;")


def show_inputs(args, state_dir):
    # just to show the initial directory structue
    for channel_name in args.channel_names:
        input_path = args.__getattribute__(f"input_{channel_name}")
        logger.info(f"input channel {channel_name} is at {input_path}")

    listDir("/opt/ml", ["__pycache__"])
    listDir(args.state)


def show_output(args, state_dir):
    # show the final directory structue
    listDir("/opt/ml", ["/opt/ml/input", "/opt/ml/code", "__pycache__"])
    listDir(args.state)


def worker():
    logging.basicConfig(stream=sys.stdout)
    task_lib.setDebugLevel()
    # parse the arguments
    args = task_lib.parseArgs()
    # get the instance specific state path
    state_dir = task_lib.initMultiWorkersState(args)
    show_inputs(args, state_dir)

    if int(args.hps["task_type"]) == 1:
        worker1(args, state_dir)
    elif int(args.hps["task_type"]) == 2:
        worker2(args, state_dir)

    # mark the task as completed
    task_lib.markCompleted(args)
    show_output(args, state_dir)

    logger.info("finished!")


if __name__ == "__main__":
    worker()
