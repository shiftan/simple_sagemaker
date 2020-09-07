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
filePath = Path(__file__).parent


def runner(projectName="simple-sagemaker-sf", prefix="", postfix="", outputPath=None):
    from simple_sagemaker.sm_project import SageMakerProject

    smProject = SageMakerProject(prefix + projectName + postfix)
    # define the code parameters
    smProject.setDefaultCodeParams(sourceDir=None, entryPoint=__file__, dependencies=[])
    # define the instance parameters
    smProject.setDefaultInstanceParams(instanceCount=2)
    # docker image
    smProject.setDefaultImageParams(
        awsRepoName="task_repo",
        repoName="task_repo",
        imgTag="latest",
        dockerFilePathOrContent=dockerFileContent,
    )
    imageUri = smProject.buildOrGetImage(
        instanceType=smProject.defaultInstanceParams.instanceType
    )
    # ceate the IAM role
    smProject.createIAMRole()

    # *** Task 1 - process input data
    task1Name = "task1"
    # set the input data
    inputDataPath = filePath.parent / "data"
    # run the task
    smProject.runTask(
        task1Name,
        imageUri,
        distribution="ShardedByS3Key",  # distribute the input files among the workers
        hyperparameters={"worker": 1, "arg": "hello world!", "task": 1},
        inputDataPath=str(inputDataPath) if inputDataPath.is_dir() else None,
        cleanState=True,  # clean the current state, also forces re-running
    )
    # download the results
    if not outputPath:
        outputPath = filePath.parent / "output"
    shutil.rmtree(outputPath, ignore_errors=True)
    smProject.downloadResults(task1Name, Path(outputPath) / "output1")

    # *** Task 2 - process the results of Task 1
    task2Name = "task2"
    # set the input
    additionalInputs = {
        "task2_data": smProject.getInputConfig(task1Name, model=True),
        "task2_data_dist": smProject.getInputConfig(
            task1Name, model=True, distribution="ShardedByS3Key"
        ),
    }
    # run the task
    smProject.runTask(
        task2Name,
        imageUri,
        hyperparameters={"worker": 1, "arg": "hello world!", "task": 2},
        cleanState=True,  # clean the current state, also forces re-running
        additionalInputs=additionalInputs,
    )
    # download the results
    smProject.downloadResults(task2Name, Path(outputPath) / "output2")

    return smProject


def worker():
    from task_toolkit import algo_lib

    algo_lib.setDebugLevel()

    logger.info("Starting worker...")
    # parse the arguments
    args = algo_lib.parseArgs()

    stateDir = algo_lib.initMultiWorkersState(args)

    logger.info(f"Hyperparams: {args.hps}")
    logger.info(f"Input data files: {list(Path(args.input_data).rglob('*'))}")
    logger.info(f"State files: { list(Path(args.state).rglob('*'))}")

    if int(args.hps["task"]) == 1:
        # update the state per running instance
        open(f"{stateDir}/state_{args.current_host}", "wt").write("state")
        # write to the model output directory
        for file in Path(args.input_data).rglob("*"):
            relp = file.relative_to(args.input_data)
            path = Path(args.model_dir) / (str(relp) + "_proc_by_" + args.current_host)
            path.write_text(file.read_text() + " processed by " + args.current_host)
        open(f"{args.model_dir}/output_{args.current_host}", "wt").write("output")
    elif int(args.hps["task"]) == 2:
        logger.info(f"Input task2_data: {list(Path(args.input_task2_data).rglob('*'))}")
        logger.info(
            f"Input task2_data_dist: {list(Path(args.input_task2_data_dist).rglob('*'))}"
        )

    # mark the task as completed
    algo_lib.markCompleted(args)
    logger.info("finished!")


def main():
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    if "--worker" in sys.argv:
        worker()
    else:
        runner()


if __name__ == "__main__":
    main()
