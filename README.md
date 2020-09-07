# Simple Sagemaker 
a *simple* python wrapper around AWS SageMaker, to easily empower your data science projects

**Note: the (initial) work is still in progress...**

The idea is simple - 

You define a series of *tasks* within a *project*, provide the code, define how the input and outputs flows through the *tasks*, set the running instance(s) parameters, and let simple-sagemaker do the rest

## Example
A [single file example](./examples/single_file/code/example.py) can be found in the [examples directory](./examples).
First, define the runner:
```python
dockerFileContent = """
# __BASE_IMAGE__ is automatically replaced with the correct base image
FROM __BASE_IMAGE__
RUN pip3 install pandas==1.1 scikit-learn==0.21.3
"""
filePath = Path(__file__).parent


def runner(projectName="simple-sagemaker-sf", prefix="", postfix="", outputPath=None):
    from simple_sagemaker.sm_project import SageMakerProject

    smProject = SageMakerProject(projectName=projectName)
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
```
An additional *task* that depends on the previous one can now be scheduled as well:
```python
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
    smProject.downloadResults(task1Name, Path(outputPath) / "output2")

    return smProject
```

Then, the worker code (note: the same function is used for the two different tasks, depending on the `task` hyperparameter):
```python
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
```

To pack everything in a single file, we use the command line argumen `--worker` (as defined in the `runner` function) to distinguish between runner and worker runs
```python
import logging
import shutil
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

...

def main():
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    if "--worker" in sys.argv:
        worker()
    else:
        runner()


if __name__ == "__main__":
    main()
```
Running the file, with a sibling directory named `data` with a sample file [as on the example](./examples/single_file/data), prduces the following outputs for Task 1:
```
INFO:__main__:Hyperparams: {'arg': 'hello world!', 'task': 1, 'worker': 1}
INFO:__main__:Input data files: [PosixPath('/opt/ml/input/data/data/sample_data1.txt')]
INFO:__main__:State files: [PosixPath('/state/algo-1')]
INFO:task_toolkit.algo_lib:Marking instance algo-1 completion
INFO:task_toolkit.algo_lib:Creating instance specific state dir
INFO:__main__:finished!
```

```
INFO:__main__:Hyperparams: {'arg': 'hello world!', 'task': 1, 'worker': 1}
INFO:__main__:Input data files: [PosixPath('/opt/ml/input/data/data/sample_data2.txt')]
INFO:__main__:State files: [PosixPath('/state/algo-2')]
INFO:task_toolkit.algo_lib:Marking instance algo-2 completion
INFO:task_toolkit.algo_lib:Creating instance specific state dir
INFO:__main__:finished!
```

And the following for Task 2:
```
INFO:__main__:Hyperparams: {'arg': 'hello world!', 'task': 2, 'worker': 1}
INFO:__main__:Input data files: [PosixPath('task_toolkit'), PosixPath('example.py'), PosixPath('task_toolkit/algo_lib.py'), PosixPath('task_toolkit/__pycache__'), PosixPath('task_toolkit/__init__.py'), PosixPath('task_toolkit/__pycache__/__init__.cpython-38.pyc'), PosixPath('task_toolkit/__pycache__/algo_lib.cpython-38.pyc')]
INFO:__main__:State files: [PosixPath('/state/algo-1')]
INFO:__main__:Input task2_data: [PosixPath('/opt/ml/input/data/task2_data/model.tar.gz')]
INFO:__main__:Input task2_data_dist: [PosixPath('/opt/ml/input/data/task2_data_dist/model.tar.gz')]
INFO:task_toolkit.algo_lib:Marking instance algo-1 completion
INFO:task_toolkit.algo_lib:Creating instance specific state dir
```

```
INFO:__main__:Hyperparams: {'arg': 'hello world!', 'task': 1, 'worker': 1}
INFO:__main__:Input data files: [PosixPath('/opt/ml/input/data/data/sample_data2.txt')]
INFO:__main__:State files: [PosixPath('/state/algo-2')]
INFO:task_toolkit.algo_lib:Marking instance algo-2 completion
INFO:task_toolkit.algo_lib:Creating instance specific state dir
INFO:__main__:finished!

```

As mentioned, the complete code can be found in [this directory](./examples/single_file), 

## Main features
1. A pure python implementation, i.e. no shell script are required
2. Save [up to 90% of the cost](https://docs.aws.amazon.com/sagemaker/latest/dg/model-managed-spot-training.html) - spot instances are used the default! (see [pricing](https://aws.amazon.com/sagemaker/pricing))
2. Except for having an AWS account, There's no assumptions on AWS pre-configuration nor AWS knowledge (well, almost :)
    - A single lime for IAM role creation
    - A single line for building a docker image and uploading to AWS (ECS service)
3. Abstraction of how data is maintianed on AWS (S3 service)
    - *State* is automaticall maintained between consequetive execution of *jobs* that belongs to the same *task*
    - A simple way to define how data flows between *tasks* of the same *project*, i.e. how the first *task*'s outputs is used as an input for the second *task*

Except for the above, the rest (mostly) behaves "normally" as defined by AWS, e.g.
- (Amazon SageMaker Developer Guide)[https://docs.aws.amazon.com/sagemaker/latest/dg/whatis.html]
- (Amazon SageMaker Python SDK @ Read the Docs)[https://sagemaker.readthedocs.io/en/stable/index.html]

## Definitions

- A *prjoect* is a series of related *tasks*
- A *task* is defined by providing a local package path, entrypoint, and list of additional local dependencies
    - a *job* is a instance SageMaker Job that is executing it
- A *task* 

## Details

As [documented](https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms.html), there're a few to run anything with SageMaker:

1. Use a built-in algorithms
2. Use a pre-built container image
3. Extending an existing container 
    - [Example](https://github.com/awslabs/amazon-sagemaker-examples/tree/master/advanced_functionality/pytorch_extending_our_containers)
4. Bringing a fully customized container 
    - [Examples](https://github.com/awslabs/amazon-sagemaker-examples/tree/master/advanced_functionality/custom-training-containers)

This project currently uses the 3rd option (currently only PyTorch is implemented), as it's the simplest one that still allows full customization of the environment. 
Future work extend the project to allow the container image to be based on any existing image

## Requirements

1. Python 3.6+
2. An AWS account with Administrator (???) credentials
3. ???

## Highlighted Features
One stop shop Python based solution, all you need is AWS credentials and you can simply
1. Fully customize the docker image (based on a pre-built one)
2. Provide the input, maintain state, and set the output for any task
3. Define input distribution method
4. Get the output data and logs
5. Save money by using spot instances
6. And many many other features which are supported by SageMaker....

## How to run

1. Configure the AWS credentials for boto, see https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html
2. 

# Development
## Pushing a code change
1. Develop ...
2. Format & lint
```bash
tox -e cf
tox -e lint
```
3. Cleanup
```bash
tox -e clean
```
3. Test
```bash
tox
```
4. Generate & test coveraeg
```bash
tox -e report
```
5. [Optionally] - bump the version string on /src/simple_sagemaker/__init__ to allow the release of a new version
5. Push your code to a development branch
    - Every push is tested for linting
6. Create a pull request to the master branch
    - Every master push is fully tested
7 If the tests succeed, the new version is publihed on P
