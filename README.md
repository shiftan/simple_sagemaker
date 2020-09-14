# Simple Sagemaker 
A **very simple** way to run your python code on the cloud (AWS).

**Note: the (initial) work is still in progress...**

Lets start with a very basic example. 
Assuming one would like to run the following `worker1.py` on the cloud:

```python
import torch
for i in range(torch.cuda.device_count()):
    print(f"Device {i}: {torch.cuda.get_device_properties(i)}")
```
It's as easy as running the following command to get it running on a *ml.m5.large* *spot* instance:
```bash
ssm -p simple-sagemaker-example-cli -t task1 -e worker1.py -o ./output/example1
```
The output, including logs will be save to `./output/example1`. The relevant part from the log file is:
```
...
-***- Device 0: _CudaDeviceProperties(name='Tesla V100-SXM2-16GB', major=7, minor=0, total_memory=16160MB, multi_processor_count=80)
...
```

## More ssm CLI examples (below)
- [Passing command line arguments](#Passing-command-line-arguments)
- [Task state and output](#Task-state-and-output)
- [Providing input data](#Providing-input-data)
- [Chaining tasks](#Chaining-tasks)
- [Defining code dependencies](#Defining-code-dependencies)
- [Configuring the docker image](#Configuring-the-docker-image)


# Main features
1. Except for having an AWS account, no assumptions, AWS pre-configuration nor AWS knowledge is assumed (well, almost :), including
    - IAM role creation
    - Building and uploading a docker image to AWS (ECS service)
    - Synchronizing local source code / input data to a S3 bucket
    - Downloading the results from S3
2. Save [up to 90% of the cost](https://docs.aws.amazon.com/sagemaker/latest/dg/model-managed-spot-training.html) - spot instances are used by default! (see [pricing](https://aws.amazon.com/sagemaker/pricing))
3. Abstraction of how data is maintianed on AWS (S3 service)
    - No need to mess with S3 paths, the data is automatically
    - *State* is automaticall maintained between consequetive execution of *jobs* that belongs to the same *task*
4. A simple way to define how data flows between *tasks* of the same *project*, i.e. how the first *task*'s outputs is used as an input for the second *task*
5. The API is mostly wrapped by a command line interface (named ***ssm***), to make your life even easier
6. Easily customize the docker image (based on a pre-built one)

On top of that, the rest (mostly) behaves "normally" as defined by AWS, e.g.
- (Amazon SageMaker Developer Guide)[https://docs.aws.amazon.com/sagemaker/latest/dg/whatis.html]
- (Amazon SageMaker Python SDK @ Read the Docs)[https://sagemaker.readthedocs.io/en/stable/index.html]


# Background
*Simple Sagemaker* is a thin warpper around SageMaker's training jobs, that makes distribution of python code on [any supported instance type](https://aws.amazon.com/sagemaker/pricing/) **very simple**. 

The solutions is composed of two partsm one on each side: a **runner** on the client machine, and a **worker** which is the distributed code on AWS. 
* The **runner** is the main part of this package, can mostly be controlled by using the **ssm** command line interface (CLI), or be fully customised using code
* The **worker** is basically your code, but a small `task_tollkit` library is injected to it, for extracting the environment configuration, i.e. input/output/state paths and running parameters.

## Definitions

- A *prjoect* is a series of related *tasks*
- A *task* is a logical step that runs on input and provide output. It's defined by providing a local package path, entrypoint, and list of additional local dependencies
    - A SageMaker *job* is a *task* instance, i.e. a *job* is created each time you run the task
    - State is maintained between consecutive execution of the same *task*

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

## How to run

1. Configure the AWS credentials for boto, see https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html
2. 


## Passing command line arguments
Any extra argument passed to the command line in assumed to be an hypermarameter. 
To get access to all environment arguments, call `algo_lib.parseArgs()`. For example, see the following worker code `worker2.py`:
```python
from task_toolkit import algo_lib

args = algo_lib.parseArgs()
print(args.hps["msg"])
```
Running command:
```bash
ssm -p simple-sagemaker-example-cli -t task2 -e worker2.py --msg "Hello, world!" -o ./output/example2
```
Output from the log file
```
Invoking script with the following command:

/opt/conda/bin/python worker2.py --msg Hello, world!

Hello, world!
```
## Task state and output

### State
State is maintained between executions of the same task, i.e. between execution jobs that belongs to the same task.
The local path is available in `args.state`. 
When running multiple instances, the data is merged into a single directory, so in order to avoid collisions, `algo_lib.initMultiWorkersState(args)` initializes a per instance sub directory. On top of that, `algo_lib` provides an additional important API to mark the task as completed: `algo_lib.markCompleted(args)`. If all instances of the job mark it as completed, the task is assumed to be completed by that job, which allows:
1. To skip it next time (unlesss eforced otherwise)
2. To use its output as input for other tasks (see below: [chaining tasks](#Chaining-tasks))

### Output
There're 3 main output mechanisms:
1. Logs - any output writen to standard output
2. Output data - args.output_data_dir is compressed into a tar.gz file, only the main instance data is kept
3. Model - args.model_dir is compressed into a tar.gz file, data from all instance is merged, so be carful with collisions.

A complete example can be seen in `worker3.py`:
```python
import os

from task_toolkit import algo_lib

args = algo_lib.parseArgs()

open(os.path.join(args.output_data_dir, "output_data_dir"), "wt").write(
    "output_data_dir file"
)
open(os.path.join(args.model_dir, "model_dir"), "wt").write("model_dir file")
open(os.path.join(args.state, "state_dir"), "wt").write("state_dir file")

# Mark the tasks as completed, to allow other tasks using its output, and to avoid re-running it (unless enforced)
algo_lib.markCompleted(args)
```
Running command:
```bash
ssm -p simple-sagemaker-example-cli$ -t task3 -e worker3.py -o ./output/example3
```
Output from the log file
```
Invoking script with the following command:

/opt/conda/bin/python worker2.py --msg Hello, world!

Hello, world!
```

## Providing input data
local path
s3 bucket

## Chaining tasks
TBD

## Configuring the docker image
TBD

## Defining code dependencies
TBD

---

Simple Sagemaker is a lightweight wrapper around AWS Sage Maker machine learning python wrapper around AWS SageMaker, to easily empower your data science projects

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
file_path = Path(__file__).parent


def runner(project_name="simple-sagemaker-sf", prefix="", postfix="", output_path=None):
    from simple_sagemaker.sm_project import SageMakerProject

    sm_project = SageMakerProject(project_name=project_name)
    # define the code parameters
    sm_project.setDefaultCodeParams(source_dir=None, entryPoint=__file__, dependencies=[])
    # define the instance parameters
    sm_project.setDefaultInstanceParams(instance_count=2)
    # docker image
    sm_project.setDefaultImageParams(
        aws_repo_name="task_repo",
        repo_name="task_repo",
        img_tag="latest",
        docker_file_path_or_content=dockerFileContent,
    )
    image_uri = sm_project.buildOrGetImage(
        instance_type=sm_project.defaultInstanceParams.instance_type
    )
    # ceate the IAM role
    sm_project.createIAMRole()

    # *** Task 1 - process input data
    task1_name = "task1"
    # set the input data
    input_data_path = file_path.parent / "data"
    # run the task
    sm_project.runTask(
        task1_name,
        image_uri,
        distribution="ShardedByS3Key",  # distribute the input files among the workers
        hyperparameters={"worker": 1, "arg": "hello world!", "task": 1},
        input_data_path=str(input_data_path) if input_data_path.is_dir() else None,
        clean_state=True,  # clean the current state, also forces re-running
    )
    # download the results
    if not output_path:
        output_path = file_path.parent / "output"
    shutil.rmtree(output_path, ignore_errors=True)
    sm_project.downloadResults(task1_name, Path(output_path) / "output1")
```
An additional *task* that depends on the previous one can now be scheduled as well:
```python
    # *** Task 2 - process the results of Task 1
    task2_name = "task2"
    # set the input
    additional_inputs = {
        "task2_data": sm_project.getInputConfig(task1_name, model=True),
        "task2_data_dist": sm_project.getInputConfig(
            task1_name, model=True, distribution="ShardedByS3Key"
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
    sm_project.downloadResults(task1_name, Path(output_path) / "output2")

    return sm_project
```

Then, the worker code (note: the same function is used for the two different tasks, depending on the `task` hyperparameter):
```python
def worker():
    from task_toolkit import algo_lib

    algo_lib.setDebugLevel()

    logger.info("Starting worker...")
    # parse the arguments
    args = algo_lib.parseArgs()

    state_dir = algo_lib.initMultiWorkersState(args)

    logger.info(f"Hyperparams: {args.hps}")
    logger.info(f"Input data files: {list(Path(args.input_data).rglob('*'))}")
    logger.info(f"State files: { list(Path(args.state).rglob('*'))}")

    if int(args.hps["task"]) == 1:
        # update the state per running instance
        open(f"{state_dir}/state_{args.current_host}", "wt").write("state")
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
4. Generate & test coverage
```bash
tox -e report
```
5. [Optionally] - bump the version string on /src/simple_sagemaker/__init__ to allow the release of a new version
5. Push your code to a development branch
    - Every push is tested for linting + some
6. Create a pull request to the master branch
    - Every master push is fully tested
7. If the tests succeed, the new version is publihed to [PyPi](https://pypi.org/project/simple-sagemaker/)
