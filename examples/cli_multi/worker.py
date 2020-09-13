import logging
import sys
from pathlib import Path

from task_toolkit import algo_lib

logger = logging.getLogger(__name__)


def task1(args, state_dir):
    # update the state per running instance
    open(f"{state_dir}/state_{args.current_host}", "wt").write("state")
    # write to the model output directory
    for file in Path(args.input_data).rglob("*"):
        if file.is_file():
            relp = file.relative_to(args.input_data)
            path = Path(args.model_dir) / (str(relp) + "_proc_by_" + args.current_host)
            path.write_text(file.read_text() + " processed by " + args.current_host)
    open(f"{args.model_dir}/output_{args.current_host}", "wt").write("output")


def task2(args, state_dir):
    logger.info(f"Input task2_data: {list(Path(args.input_task2_data).rglob('*'))}")


def main():
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    algo_lib.setDebugLevel()

    logger.info("Starting worker...")
    # parse the arguments
    args = algo_lib.parseArgs()

    state_dir = algo_lib.initMultiWorkersState(args)

    logger.info(f"Hyperparams: {args.hps}")
    logger.info(f"Input data files: {list(Path(args.input_data).rglob('*'))}")
    logger.info(f"State files: { list(Path(args.state).rglob('*'))}")

    if int(args.hps["task_type"]) == 1:
        task1(args, state_dir)
    elif int(args.hps["task_type"]) == 2:
        task2(args, state_dir)

    # mark the task as completed
    algo_lib.markCompleted(args)
    logger.info("finished!")


if __name__ == "__main__":
    main()
