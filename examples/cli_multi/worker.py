import logging
import sys
from pathlib import Path

from worker_toolkit import worker_lib

logger = logging.getLogger(__name__)


def task1(worker_config):
    # update the state per running instance
    open(
        f"{worker_config.instance_state}/state_{worker_config.current_host}", "wt"
    ).write("state")
    # write to the model output directory
    for file in Path(worker_config.channel_data).rglob("*"):
        if file.is_file():
            relp = file.relative_to(worker_config.channel_data)
            path = Path(worker_config.model_dir) / (
                str(relp) + "_proc_by_" + worker_config.current_host
            )
            path.write_text(
                file.read_text() + " processed by " + worker_config.current_host
            )
    open(f"{worker_config.model_dir}/output_{worker_config.current_host}", "wt").write(
        "output"
    )


def task2(worker_config):
    logger.info(
        f"Input task2_data: {list(Path(worker_config.channel_task2_data).rglob('*'))}"
    )


def main():
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    logger.info("Starting worker...")
    # parse the arguments
    worker_config = worker_lib.WorkerConfig()

    logger.info(f"Hyperparams: {worker_config.hps}")
    logger.info(
        f"Input data files: {list(Path(worker_config.channel_data).rglob('*'))}"
    )
    logger.info(f"State files: { list(Path(worker_config.state).rglob('*'))}")

    if int(worker_config.hps["task_type"]) == 1:
        task1(worker_config)
    elif int(worker_config.hps["task_type"]) == 2:
        task2(worker_config)

    logger.info("finished!")
    # The task is marked as completed


if __name__ == "__main__":
    main()
