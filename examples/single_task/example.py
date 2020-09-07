import logging
import os
import shutil
import sys
from time import gmtime, strftime

logger = logging.getLogger(__name__)

filePath = os.path.split(__file__)[0]
if "TOX_ENV_NAME" not in os.environ:
    srcPath = os.path.abspath(os.path.join(filePath, "..", "..", "src"))
    sys.path.append(srcPath)
from simple_sagemaker.sm_project import SageMakerProject  # noqa: E402


def setDefaultParams(smProject):
    # docker image params
    awsRepoName = "task_repo"  # remote (ECR) rpository name
    repoName = "task_repo"  # local repository name
    imgTag = "latest"  # tag for local & remote images
    dockerFilePath = os.path.join(filePath, "docker")  # path of the local Dockerfile
    smProject.setDefaultImageParams(awsRepoName, repoName, imgTag, dockerFilePath)

    # job code path, entrypoint and params
    sourceDir = os.path.join(filePath, "code")
    entryPoint = "algo.py"
    dependencies = [os.path.join(filePath, "external_dependency")]
    smProject.setDefaultCodeParams(sourceDir, entryPoint, dependencies)

    # instances type an count
    instanceType = "ml.m5.large"
    trainingInstanceCount = 2
    volumeSize = (
        30  # Size in GB of the EBS volume to use for storing input data during training
    )
    useSpotInstances = True  # False
    maxRun = 24 * 60 * 60
    maxWait = None
    if useSpotInstances:
        maxWait = maxRun  # should be >= maxRun
    smProject.setDefaultInstanceParams(
        instanceType,
        trainingInstanceCount,
        volumeSize,
        useSpotInstances,
        maxRun,
        maxWait,
    )


def buildImage(smProject, fallbackUri=None):
    try:
        # build a local image
        imageUri = smProject.buildOrGetImage(
            instanceType=smProject.defaultInstanceParams.instanceType
        )
        # use an AWS pre-built image
        # imageUri = "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.6.0-cpu-py3"
    except:  # noqa: E722
        logger.exception("Couldn't build image")
        if not fallbackUri:
            raise
        logger.info(f"falling back to {fallbackUri}")
        # for debugging whe're
        imageUri = fallbackUri

    return imageUri


def runner(
    projectName="simple-sagemaker-example", prefix="", postfix="", outputPath=None
):
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    smProject = SageMakerProject(prefix + projectName + postfix)

    setDefaultParams(smProject)
    imageUri = buildImage(
        smProject, "667232328135.dkr.ecr.us-east-1.amazonaws.com/task_repo:latest"
    )
    smProject.createIAMRole()

    # task name
    taskName = (
        "Task1"  # must satisfy regular expression pattern: ^[a-zA-Z0-9](-*[a-zA-Z0-9])*
    )
    # input data params
    inputDataPath = os.path.join(
        filePath, "input_data"
    )  # Can also provide a URI to an S3 bucket, e.g. next commented line
    # inputDataPath = sagemaker.s3.s3_path_join("s3://", "sagemaker-us-east-1-667232328135", "task3", "input")
    distribution = "ShardedByS3Key"  # or "FullyReplicated" which is the default
    modelUri = None  # Can be used to supply model data as an additional input, local/s3
    hyperparameters = {"arg1": 5, "arg2": "hello"}

    smProject.runTask(
        taskName,
        imageUri,
        hyperparameters,
        inputDataPath,
        modelUri=modelUri,
        distribution=distribution,
        cleanState=True,
    )

    # delete the output directory
    if not outputPath:
        outputPath = os.path.join(filePath, "output")
    shutil.rmtree(outputPath, ignore_errors=True)
    smProject.downloadResults(taskName, outputPath)

    return smProject


if __name__ == "__main__":
    pyVersionString = f"py{sys.version_info.major}{sys.version_info.minor}"
    timeString = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    smProject = runner(postfix=f"_{timeString}_{pyVersionString}", prefix="tests/")
