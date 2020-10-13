import logging
import os
import shutil
import sys

logger = logging.getLogger(__name__)

file_path = os.path.split(__file__)[0]
if "TOX_ENV_NAME" not in os.environ:
    srcPath = os.path.abspath(os.path.join(file_path, "..", "..", "src"))
    sys.path.append(srcPath)
from simple_sagemaker.sm_project import SageMakerProject  # noqa: E402


def setDefaultParams(sm_project):
    # docker image params
    aws_repo_name = "task_repo"  # remote (ECR) rpository name
    repo_name = "task_repo"  # local repository name
    image_tag = "latest"  # tag for local & remote images
    docker_file_path = os.path.join(
        file_path, "..", "single_task", "docker"
    )  # path of the local Dockerfile
    sm_project.setDefaultImageParams(
        aws_repo_name, repo_name, image_tag, docker_file_path
    )

    # job code path, entrypoint and params
    source_dir = os.path.join(file_path, "code")
    entry_point = "algo_multi.py"
    dependencies = []
    sm_project.setDefaultCodeParams(source_dir, entry_point, dependencies)

    # instances type an count
    instance_type = "ml.m5.large"
    training_instance_count = 2
    volume_size = (
        30  # Size in GB of the EBS volume to use for storing input data during training
    )
    use_spot_instances = True  # False
    max_run_mins = 15
    sm_project.setDefaultInstanceParams(
        instance_type,
        training_instance_count,
        volume_size,
        use_spot_instances,
        max_run_mins,
    )


def runner(
    project_name="simple-sagemaker-example-multi",
    prefix="",
    postfix="",
    output_path=None,
):
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    sm_project = SageMakerProject(project_name, prefix=prefix)
    setDefaultParams(sm_project)
    image_uri = sm_project.buildOrGetImage(
        instance_type=sm_project.defaultInstanceParams.instance_type
    )

    # task name
    task_name = (
        "multi-task1" + postfix
    )  # must satisfy regular expression pattern: ^[a-zA-Z0-9](-*[a-zA-Z0-9])*
    # input data params
    input_data_path = os.path.join(
        file_path, "..", "single_task", "input_data"
    )  # Can also provide a URI to an S3 bucket, e.g. next commented line
    # input_data_path = sagemaker.s3.s3_path_join("s3://", "sagemaker-us-east-1-XXXXXXXXXXXX", "task3", "input")
    distribution = "ShardedByS3Key"  # or "FullyReplicated" which is the default
    model_uri = (
        None  # Can be used to supply model data as an additional input, local/s3
    )
    hyperparameters = {"stage": 1}
    sm_project.runTask(
        task_name,
        image_uri,
        hyperparameters,
        input_data_path,
        model_uri=model_uri,
        input_distribution=distribution,
        clean_state=True,
    )

    if not output_path:
        output_path = os.path.join(file_path, "output")
    # delete the output directory
    outputDir1 = os.path.join(output_path, "output1")
    shutil.rmtree(outputDir1, ignore_errors=True)
    sm_project.downloadResults(task_name, outputDir1)

    task_name2 = "multi-task2"
    hyperparameters = {"stage": 2}
    additional_inputs = dict()
    additional_inputs["task1_state1"] = sm_project.getInputConfig(task_name, "state")
    additional_inputs["task1_state2"] = sm_project.getInputConfig(
        task_name, "state", distribution="ShardedByS3Key"
    )
    additional_inputs["task1_state3"] = sm_project.getInputConfig(
        task_name, "output", distribution="ShardedByS3Key"
    )
    model_uri = sm_project.tasks[task_name].getOutputTargetUri(model=True)
    sm_project.runTask(
        task_name2,
        image_uri,
        hyperparameters,
        input_data_path,
        model_uri=model_uri,
        input_distribution=distribution,
        additional_inputs=additional_inputs,
        clean_state=True,
    )

    # delete the output directory
    output_dir2 = os.path.join(output_path, "output2")
    shutil.rmtree(output_dir2, ignore_errors=True)
    sm_project.downloadResults(task_name2, output_dir2)

    return sm_project


if __name__ == "__main__":
    runner()
