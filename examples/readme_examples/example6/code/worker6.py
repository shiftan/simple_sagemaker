import logging
import sys
import time
from pathlib import Path

# a library that was installed due to requirements.txt
import transformers  # noqa: F401

# importing an internal dependency
from internal_dependency import lib2  # noqa: F401
from worker_toolkit import worker_lib

logger = logging.getLogger(__name__)


def listDir(path, ignore_patterns=[]):
    logger.info(f"*** START listing files in {path}")
    for file in sorted(Path(path).rglob("*")):
        if (not ignore_patterns) or all(
            [pattern not in str(file) for pattern in ignore_patterns]
        ):
            logger.info(f"[{['Dir ', 'File'][file.is_file()]}] {file}")
    logger.info(f"*** END file listing {path}")


def worker1(worker_config):
    # Libraries that were pre-installed in the docker image, as defined in the Dockerfile
    import pandas  # noqa: F401
    import sklearn  # noqa: F401

    logger.info("{pandas} is pre-installed in this image")

    # update the state
    (Path(worker_config.instance_state) / worker_config.current_host).write_text(
        f"state_{worker_config.current_host}"
    )
    # "process" input data into model output
    for file in Path(worker_config.channel_data).rglob("*"):
        relp = file.relative_to(worker_config.channel_data)
        path = Path(worker_config.model_dir) / (
            f"{relp}_proc_by_{worker_config.current_host}"
        )
        path.write_text(f"{file.read_text()} processed by {worker_config.current_host}")
    # write to output dir
    (
        Path(worker_config.output_data_dir) / f"output_{worker_config.current_host}"
    ).write_text(f"output_{worker_config.current_host}")


def worker2(worker_config):
    # importing an external dependency
    from external_dependency import lib1  # noqa: F401

    logger.info("Score=10;")
    time.sleep(60)  # sleep to be able to see the two scores
    logger.info("Score=20;")


def show_inputs(worker_config):
    # just to show the initial directory structue
    for channel_name in worker_config.channels:
        input_path = worker_config.__getattr__(f"channel_{channel_name}")
        logger.info(f"input channel {channel_name} is at {input_path}")

    listDir("/opt/ml", ["__pycache__"])
    listDir(worker_config.state)


def show_output(worker_config):
    # show the final directory structue
    listDir("/opt/ml", ["/opt/ml/input", "/opt/ml/code", "__pycache__"])
    listDir(worker_config.state)


def worker():
    logging.basicConfig(stream=sys.stdout)
    # parse the arguments
    worker_config = worker_lib.WorkerConfig()
    # get the instance specific state path
    show_inputs(worker_config)

    if int(worker_config.hps["task_type"]) == 1:
        worker1(worker_config)
    elif int(worker_config.hps["task_type"]) == 2:
        worker2(worker_config)

    show_output(worker_config)

    logger.info("finished!")
    # The task is marked as completed


if __name__ == "__main__":
    worker()
