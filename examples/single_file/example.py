import logging
import shutil
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

dockerFileContent = """
# __BASE_IMAGE__ is automatically replaced with the correct base image
FROM __BASE_IMAGE__
RUN pip3 install pandas==1.1 scikit-learn==0.21.3
"""
file_path = Path(__file__).parent


def runner(project_name="simple-sagemaker-sf", prefix="", postfix="", output_path=None):
    from simple_sagemaker.sm_project import SageMakerProject

    sm_project = SageMakerProject(project_name, prefix=prefix)
    # define the code parameters
    sm_project.setDefaultCodeParams(
        source_dir=None, entry_point=__file__, dependencies=[]
    )
    # define the instance parameters
    sm_project.setDefaultInstanceParams(instance_count=2, max_run_mins=15)
    # docker image
    sm_project.setDefaultImageParams(
        aws_repo_name="task_repo",
        repo_name="task_repo",
        image_tag="latest",
        docker_file_path_or_content=dockerFileContent,
    )
    image_uri = sm_project.buildOrGetImage(
        instance_type=sm_project.defaultInstanceParams.instance_type
    )

    # *** Task 1 - process input data
    task1_name = "single-file-task1" + postfix
    # set the input data
    input_data_path = file_path / "data"
    # run the task
    sm_project.runTask(
        task1_name,
        image_uri,
        input_distribution="ShardedByS3Key",  # distribute the input files among the workers
        hyperparameters={"worker": 1, "arg": "hello world!", "task": 1},
        input_data_path=str(input_data_path) if input_data_path.is_dir() else None,
        clean_state=True,  # clean the current state, also forces re-running
    )
    # download the results
    if not output_path:
        output_path = file_path / "output"
    shutil.rmtree(output_path, ignore_errors=True)
    sm_project.downloadResults(task1_name, Path(output_path) / "output1")

    # *** Task 2 - process the results of Task 1
    task2_name = "single-file-task2" + postfix
    # set the input
    additional_inputs = {
        "task2_data": sm_project.getInputConfig(task1_name, "model"),
        "task2_data_dist": sm_project.getInputConfig(
            task1_name, "model", distribution="ShardedByS3Key"
        ),
    }
    # run the task
    sm_project.runTask(
        task2_name,
        image_uri,
        hyperparameters={"worker": 1, "arg": "hello world!", "task": 2},
        clean_state=True,  # clean the current state, also forces re-running
        additional_inputs=additional_inputs,
    )
    # download the results
    sm_project.downloadResults(task2_name, Path(output_path) / "output2")

    return sm_project


def worker():
    from worker_toolkit import worker_lib

    logger.info("Starting worker...")
    # parse the arguments
    worker_config = worker_lib.WorkerConfig()

    logger.info(f"Hyperparams: {worker_config.hps}")
    logger.info(
        f"Input data files: {list(Path(worker_config.channel_data).rglob('*'))}"
    )
    logger.info(f"State files: { list(Path(worker_config.state).rglob('*'))}")

    if int(worker_config.hps["task"]) == 1:
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
        open(
            f"{worker_config.model_dir}/output_{worker_config.current_host}", "wt"
        ).write("output")
    elif int(worker_config.hps["task"]) == 2:
        logger.info(
            f"Input task2_data: {list(Path(worker_config.channel_task2_data).rglob('*'))}"
        )
        logger.info(
            f"Input task2_data_dist: {list(Path(worker_config.channel_task2_data_dist).rglob('*'))}"
        )

    logger.info("finished!")
    # The task is marked as completed


def main():
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    if "--worker" in sys.argv:
        worker()
    else:
        runner()


if __name__ == "__main__":
    main()
