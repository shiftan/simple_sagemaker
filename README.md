# Simple Sagemaker 
A **simpler** and **cheaper** way to distribute work (python/shell/training) work on machines of your choice in the (AWS) cloud.

**Note: this (initial) work is still in progress. Only SageMaker's [PyTorch](https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/index.html) and [TensorFlow](https://sagemaker.readthedocs.io/en/stable/frameworks/tensorflow/index.html) frameworks are currently supported. But, these frameworks are enough to distribute any type of work, including shell commands, just without the specific customization.**

## Requirements
1. Python 3.6+
2. An AWS account + region and credentials configured for boto3, as explained on the [Boto3 docs](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html)
3. (Optional) The [Docker Engine](https://docs.docker.com/get-docker/), to be able to customize a docker image
4. (Optional) The [Docker Compose](https://docs.docker.com/compose/install/), for local testing

## Getting started
To install *Simple Sagemaker*
```
pip install simple-sagemaker
```
Then, to get the shell command `cat /proc/cpuinfo && nvidia-smi` run on a single ml.p3.2xlarge instance, run the following `ssm` command (documentation of the `ssm` CLI is given [below](#cli)):
```bash
ssm shell -p simple-sagemaker-example-cli-shell -t shell-task -o ./output --cmd_line "cat /proc/cpuinfo && nvidia-smi"
```

Output including the logs with script stdout is downloaded to `./output`.

```bash
$ cat ./output/logs/logs0
processor: 0
model name: Intel(R) Xeon(R) CPU E5-2686 v4 @ 2.30GHz
cpu cores: 4
....
processor: 2
....
processor: 7
....
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 440.33.01    Driver Version: 440.33.01    CUDA Version: 10.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla V100-SXM2...  On   | 00000000:00:1E.0 Off |                    0 |
| N/A   46C    P0    27W / 300W |      0MiB / 16160MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
....
```

Similarly, to run the following `worker1.py` on two *ml.p3.2xlarge* *spot* instances
```python
import torch

for i in range(torch.cuda.device_count()):
    print(f"-***- Device {i}: {torch.cuda.get_device_properties(i)}")
```
Just run the below `ssm` command:
```bash
ssm run -p simple-sagemaker-example-cli -t task1 -e worker1.py -o ./output/example1 --it ml.p3.2xlarge --ic 2
```
The output is saved to `./output/example1`, logs to `./output/example1/logs/logs0` and `./output/example1/logs/logs1`:
```bash
$ cat ./output/example1/logs/logs0
...
-***- Device 0: _CudaDeviceProperties(name='Tesla V100-SXM2-16GB', major=7, minor=0, total_memory=16160MB, multi_processor_count=80)
...
```
It's recommended to review the [fully featured advanced example](#A-fully-featured-advanced-example), as a demonstration of most features.

## More examples (below)
CLI based examples:
- [Passing command line arguments](#Passing-command-line-arguments)
- [Task state and output](#Task-state-and-output)
- [Providing input data](#Providing-channel-data)
- [Chaining tasks](#Chaining-tasks)
- [Configuring the docker image](#Configuring-the-docker-image)
- [Defining code dependencies](#Defining-code-dependencies)

API based example:
- [Single file example](#Single-file-example)

## Motivation
Well, I couldn't find a simple and cheap way to run my existing code on cloud. And, I wanted to try having my own OS project including documentation and a full automated testing and publishing piepline, so here it is :) 
Please refer to [this blog post](https://medium.com/@shiftan/a-very-simple-and-cheap-way-to-run-your-processing-job-on-the-cloud-c76af579f9e9?postPublishedType=repub).

# Background
*Simple Sagemaker* is a thin wrapper around SageMaker's training **jobs**, that makes distribution of work (python/shell) on [any supported instance type](https://aws.amazon.com/sagemaker/pricing/) **very simple**. 

The distribution solution is composed of two parts, one on each side: a **runner** on the client machine that manages the distribution process, and a **worker** which is the code being distributed on the cloud.
* The **runner** is the main part of this package, can mostly be controlled by using the `ssm` command line interface (CLI), or be fully customized by using the python API.
* The **worker** is basically the work (shell/python code) being distributed. Python code may be adapted to use a small `task_toolkit` library (that is automatically injected to the **worker**) for getting the environment configuration (`WorkerConfig`, see [below](#Configuration)), i.e. input/output/state paths, running parameters. Shell command can access the same parameters on the command line, and completion is determined by the exit code (i.e. 0 is a success) etc.

The **runner** is used to configure **tasks** and **projects**: 
- A **task** is a logical step that runs on a defined input and provide output. It's defined by providing a local code path, entrypoint, and a list of additional local dependencies
- A SageMaker **job** is a **task** instance, i.e. a single **job** is created each time a **task** is executed
    - State is maintained between consecutive execution of the same **task** (see more [below](#Task-state-and-output))
    - If a **task** was completed, by returnin 0 retcode from all instances, it'll be skipped automatically on the next time (unless enforced otherwise)
- A **project** is a series of related **tasks**, with possible dependencies
    - The output of a completed task can be consumed as input by a consecutive task

# Main features
1. "Simpler" - Except for holding an AWS account credentials, no other pre-configuration nor knowledge is assumed (well, almost :). Behind the scenes you get:
    - Jobs IAM role creation, including policies for accessing needed S3 buckets
    - Building and uploading a customized docker image to AWS (ECS service)
    - Synchronizing local source code / input data to a S3 bucket
    - Downloading the results from S3
    - ...
2. "Cheaper" - ["pay only for what you use"](https://aws.amazon.com/sagemaker/pricing/), and save [up to 90% of the cost](https://docs.aws.amazon.com/sagemaker/latest/dg/model-managed-spot-training.html) with spot instances, which got used by default!
3. Abstraction of how data is maintained on AWS (S3 service)
    - No need to mess with S3 paths, the data is automatically
    - State is automatically maintained between consecutive execution of **jobs** that belongs to the same **task**
4. A simple way to define how data flows between **tasks** of the same **project**, e.g. how the first **task**'s outputs is used as an input for a second **task**
5. (Almost) no code changes are to the existing code - the API is mostly wrapped by a command line interface (named `ssm`) to control the execution (a.k.a implement the **runner**, see below)
    - In most cases it's only about 1 line for getting the environment configuration (e.g. input/output/state paths and running parameters, see [below](#Configuration)) and passing it on to the original code
6. Easy customization of the docker image (based on a pre-built one)
7. The rest of the SageMaker advantages, which (mostly) behaves "normally" as defined by AWS, e.g.
    - (Amazon SageMaker Developer Guide)[https://docs.aws.amazon.com/sagemaker/latest/dg/whatis.html]
    - (Amazon SageMaker Python SDK @ Read the Docs)[https://sagemaker.readthedocs.io/en/stable/index.html]


## High level flow diagram
![High level flow diagram](https://github.com/shiftan/simple_sagemaker/blob/master/docs/high_level_flow.svg?raw=true "High level flow")

# Worker environment
The worker entry point (`entry_point` parameter), directory (`source_dir` for python code / .sh script, `dir_files` for shell script), 
along with all dependencies (`dependencies` parameter) are getting copied to a single directory (`/opt/ml/code`) on each instance, 
and the entry point is then executed. 
On top of the above, for python code tasks, the `task_toolkit` library is also added as a dependency in this folder.

## Configuration
The worker can access the environment configuration parameters in two ways:
1. The environment variables, e.g. `SM_NUM_CPUS` represents the number of CPUs.
2. Using the `worker_lib` library: initialize a `WorkerConfig` instance, `worker_config = worker_lib.WorkerConfig()`, and then all params can be accessible from the `worker_config` variable, e.g.  `worker_lib.num_cpus` is the number of CPUs.

The complete list of configuration parameters:

| Description | Environment variable | `worker_config` field name | Example |
| ----------- | ----------- | ----------- | ----------- |
| The name of the current running **job** | SAGEMAKER_JOB_NAME | job_name | 'task1-2020-09-23-17-12-46-0JNcrR6H'
| **Input channels**:|
| Names of the input channels | SM_CHANNELS | channels | ['data']
| The data input channel | SM_CHANNEL_DATA | channel_data | '/opt/ml/input/data/data'
| Path where the input model (given by `model_uri` parameter) is located | SM_CHANNEL_MODEL | channel_model | '/opt/ml/input/data/model'
| Generally - path where the channel [ChannelName] is located | SM_CHANNEL_[ChannelName] | channel_[ChannelName] | '/opt/ml/input/data/[ChannelName]'
| Additional command line parameters / hyperparameters | SM_HPS | hps | {'arg': 'hello world!', 'task': 1, 'worker': 1}
| **State**:|
| The root path of where state should be stored | SSM_STATE | state | '/state'
| The instance specific state path | SSM_INSTANCE_STATE | instance_state | '/state/algo-1'
| **Output**:|
| The path where output data should be stored | SM_OUTPUT_DATA_DIR | output_data_dir | '/opt/ml/output/data'
| Path where model output should be stored | SM_MODEL_DIR | model_dir | '/opt/ml/model'
| TensorBoard output | SSM_TENSORBOARD_DIR | tensorboard_dir | '/opt/ml/output/tensorboard'
| **System**:|
| The number of available CPUs on this instance | SM_NUM_CPUS | num_cpus | 2
| The number of available GPUs  instance| SM_NUM_GPUS | num_gpus | 1
| Name of the current host | SM_CURRENT_HOST | current_host | 'algo-1'
| Names of all other hosts that are running on this **job** | SM_HOSTS | hosts | ['algo-1', 'algo-2']
| The name of the network interface | SM_NETWORK_INTERFACE_NAME | network_interface_name | 'eth0'
| The number of instance running for this **job** | SSM_NUM_NODES | num_nodes | 'eth0'
| The rank of the current instance | SSM_HOST_RANK | host_rank | 'eth0'

## State
State is maintained between executions of the same **task**, i.e. between **jobs** that belongs to the same **task**.
The local path is available in `worker_config.state`. 
When running multiple instances, the state data is merged into a single directory (post execution).  To avoid collisions, set the `per_instance_state` parameter of `WorkerConfig` constructor to `True` (the default behavior), which initializes a per instance sub directory, and keep it in `worker_config.instance_state`. On top of that, the return value plays an important part: returning 0 means the **job** is completed. If all instances of a **job** marked it as completed, the **task** is assumed to be completed by that **job**, which allows:
1. To skip it next time (unless enforced otherwise by using `--force_running` or a newer run of the same **task** failed)
2. To use its output as input for other **tasks** (see below: ["Chaining tasks"](#Chaining-tasks))

## Output
On top of the state, there're 3 main other output mechanisms:
1. Logs - any output written to standard output / error
2. Output data - any data in `worker_config.output_data_dir` is compressed into a output.tar.gz. Only the main instance output data is kept.
3. Model - any data in `worker_config.model_dir` is compressed into a model.tar.gz. As data from all instance is merged, be carful with collisions.


# Data maintenance on S3
All data, including input, code, state and output, is maintained on S3. The bucket to use can be defined, or the default one is used.
The files and directories structure is as follows:
```
[Bucket name]/[Project name]/[Task name]
|-- state
|-- input
|-- [Job name]
|   |-- output
|   |   |-- model.tar.gz
|   |   `-- output.tar.gz
|   `-- source/sourcedir.tar.gz
|-- [Job name 2]
|        ...
```
- state - the task state, shared between all jobs, i.e. task executions
- input - the task input, shared as well
- [Job name] - a per job specific folder
    - model.tar.gz - model output data, merged from *all instances*
    - output.tar.gz - the *main instance* output data (other outputs are ignored)
    - sourcedir.tar.gz - source code and dependencies
- [Job name 2] - another execution of the same task

# Local mode
SageMaker offers partial ["local mode"](https://sagemaker.readthedocs.io/en/stable/overview.html#local-mode) support in order to test locally. The basic mode runs just the docker locally, while keep using S3 for input/output, and there's the `local_code` mode that does everything locally.
To use the basic mode with *Simple Sagemaker*, `local` or `local_gpu` as instance type and `local_mode = True` for `SageMakerProject` constructor (this is done automatically with `ssm` CLI).
Notes:
- sagemaker[local] is required: `pip install sagemaker[local]`
- Local mode doesn't support all features, e.g. state isn't supported. More notes and exclusions can be seen on the [documentation](["local_mode"](https://sagemaker.readthedocs.io/en/stable/overview.html#local-mode) )
- `local_code` mode isn't currently supported by *Simple Sagemaker*

# Distributed training
Sagemaker's PyTorch and TensorFlow pre-built images has extra customization for distributed training. Make sure to configure `framework`, 
`framework_version` and `py_version` to use the image that matches your needs ([the full list is here](https://github.com/aws/deep-learning-containers/blob/master/available_images.md)). For TensorFlow you'll need to use the `distribution` parameters. For more details on the built in support see:
- PyTorch - [Distributed PyTorch Training](https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/using_pytorch.html#distributed-pytorch-training)
- TensorFlow - [Distributed TensorFlow Training](https://sagemaker.readthedocs.io/en/stable/frameworks/tensorflow/using_tf.html#distributed-training). 

# Processing tasks
TBD
For now, take a look [on the processing cli examples](https://github.com/shiftan/simple_sagemaker/tree/master/examples/processing_cli/run.sh), and the `ssm process -h` output.

# CLI
The `ssm` CLI supports 4 commands:
- run - to run a python / .sh script based task
- shell - to run a shell based task
- data - to manage (download/clear state) the data of an existing task
- process - to run a processing command, script or generic
```bash
$ ssm -h
usage: ssm [-h] {run,shell,data,process} ...

positional arguments:
  {run,shell,data,process}
    run                 Run a python / .sh script task
    shell               Run a shell task
    data                Manage task data
    process             Run a processing task

optional arguments:
  -h, --help            show this help message and exit
```
```bash  
$ ssm run -h
usage: ssm run [-h] --project_name PROJECT_NAME --task_name TASK_NAME
               [--bucket_name BUCKET_NAME] [--source_dir SOURCE_DIR]
               --entry_point ENTRY_POINT
               [--dependencies DEPENDENCIES [DEPENDENCIES ...]]
               [--instance_type INSTANCE_TYPE]
               [--instance_count INSTANCE_COUNT] [--volume_size VOLUME_SIZE]
               [--no_spot] [--use_spot_instances]
               [--max_wait_mins MAX_WAIT_MINS] [--max_run_mins MAX_RUN_MINS]
               [--aws_repo_name AWS_REPO_NAME] [--repo_name REPO_NAME]
               [--image_tag IMAGE_TAG]
               [--docker_file_path_or_content DOCKER_FILE_PATH_OR_CONTENT]
               [--framework {pytorch,tensorflow}]
               [--framework_version FRAMEWORK_VERSION]
               [--py_version PY_VERSION]
               [--input_path INPUT_PATH [INPUT_PATH ...]]
               [--model_uri MODEL_URI] [--input_s3 INPUT_S3 [INPUT_S3 ...]]
               [--input_task INPUT_TASK [INPUT_TASK ...]] [--force_running]
               [--distribution DISTRIBUTION] [--clean_state] [--keep_state]
               [--metric_definitions name regexp] [--enable_sagemaker_metrics]
               [--tag key value] [--output_path OUTPUT_PATH]
               [--download_state] [--download_model] [--download_output]

optional arguments:
  -h, --help            show this help message and exit
  --project_name PROJECT_NAME, -p PROJECT_NAME
                        Project name. (default: None)
  --task_name TASK_NAME, -t TASK_NAME
                        Task name. (default: None)
  --bucket_name BUCKET_NAME, -b BUCKET_NAME
                        S3 bucket name (a default one is used if not given).
                        (default: None)

Code:
  --source_dir SOURCE_DIR, -s SOURCE_DIR
                        Path (absolute, relative or an S3 URI) to a directory
                        with any other source code dependencies aside from the
                        entry point file. If source_dir is an S3 URI, it must
                        point to a tar.gz file. Structure within this
                        directory are preserved when running on Amazon
                        SageMaker. (default: None)
  --entry_point ENTRY_POINT, -e ENTRY_POINT
                        Path (absolute or relative) to the local Python source
                        file or a .sh script which should be executed as the
                        entry point. If source_dir is specified, then
                        entry_point must point to a file located at the root
                        of source_dir. (default: None)
  --dependencies DEPENDENCIES [DEPENDENCIES ...], -d DEPENDENCIES [DEPENDENCIES ...]
                        A list of paths to directories (absolute or relative)
                        with any additional libraries that will be exported to
                        the container The library folders will be copied to
                        SageMaker in the same folder where the entrypoint is
                        copied. (default: None)

Instance:
  --instance_type INSTANCE_TYPE, --it INSTANCE_TYPE
                        Type of EC2 instance to use. (default: ml.m5.large)
  --instance_count INSTANCE_COUNT, --ic INSTANCE_COUNT
                        Number of EC2 instances to use. (default: 1)
  --volume_size VOLUME_SIZE, -v VOLUME_SIZE
                        Size in GB of the EBS volume to use for storing input
                        data. Must be large enough to store input data.
                        (default: 30)
  --no_spot             Use on demand instances (default: True)
  --use_spot_instances  Specifies whether to use SageMaker Managed Spot
                        instances. (default: True)
  --max_wait_mins MAX_WAIT_MINS
                        Timeout in minutes waiting for spot instances. After
                        this amount of time Amazon SageMaker will stop waiting
                        for Spot instances to become available. If 0 is
                        specified and spot instances are used, its set to
                        max_run_mins (default: 0)
  --max_run_mins MAX_RUN_MINS
                        Timeout in minutes for running. After this amount of
                        time Amazon SageMaker terminates the job regardless of
                        its current status. (default: 1440)

Image:
  --aws_repo_name AWS_REPO_NAME, --ar AWS_REPO_NAME
                        Name of ECS repository. (default: None)
  --repo_name REPO_NAME, --rn REPO_NAME
                        Name of local repository. (default: None)
  --image_tag IMAGE_TAG
                        Image tag. (default: latest)
  --docker_file_path_or_content DOCKER_FILE_PATH_OR_CONTENT, --df DOCKER_FILE_PATH_OR_CONTENT
                        Path to a directory containing the DockerFile. The
                        base image should be set to `__BASE_IMAGE__` within
                        the Dockerfile, and is automatically replaced with the
                        correct base image. (default: None)
  --framework {pytorch,tensorflow}, -f {pytorch,tensorflow}
                        The framework to use, see https://github.com/aws/deep-
                        learning-containers/blob/master/available_images.md
                        (default: pytorch)
  --framework_version FRAMEWORK_VERSION, --fv FRAMEWORK_VERSION
                        The framework version (default: None)
  --py_version PY_VERSION, --pv PY_VERSION
                        The python version (default: None)

Running:
  --force_running       Force running the task even if its already completed.
                        (default: False)
  --distribution DISTRIBUTION
                        Tensorflows distribution policy, see https://sagemake
                        r.readthedocs.io/en/stable/frameworks/tensorflow/using
                        _tf.html#distributed-training. (default: None)
  --tag key value       Tag to be attached to the jobs executed for this task.
                        (default: None)

I/O:
  --input_path INPUT_PATH [INPUT_PATH ...], -i INPUT_PATH [INPUT_PATH ...]
                        INPUT: PATH [DISTRIBUTION] [SUBDIR] Local/s3 path for
                        the input data. If a local path is given, it will be
                        synced to the task folder on the selected S3 bucket
                        before launching the task. (default: None)
  --model_uri MODEL_URI
                        URI where a pre-trained model is stored, either
                        locally or in S3. If specified, the estimator will
                        create a channel pointing to the model so the training
                        job can download it. This model can be a
                        ‘model.tar.gz’ from a previous training job, or other
                        artifacts coming from a different source. (default:
                        None)
  --input_s3 INPUT_S3 [INPUT_S3 ...], --iis INPUT_S3 [INPUT_S3 ...]
                        INPUT_S3: INPUT_NAME S3_URI [DISTRIBUTION] [SUBDIR]
                        Additional S3 input sources (a few can be given).
                        (default: None)
  --input_task INPUT_TASK [INPUT_TASK ...], --iit INPUT_TASK [INPUT_TASK ...]
                        INPUT_TASK: INPUT_NAME TASK_NAME TYPE [DISTRIBUTION]
                        [SUBDIR] Use an output of a completed task in the same
                        project as an input source (a few can be given). Type
                        should be one of ['state', 'model', 'source',
                        'output']. (default: None)
  --clean_state, --cs   Clear the task state before running it. The task will
                        be running again even if it was already completed
                        before. (default: False)
  --keep_state, --ks    Keep the current task state. If the task is already
                        completed, its current output will be taken without
                        running it again. (default: True)
  --metric_definitions name regexp, --md name regexp
                        Name and regexp for a metric definition, a few can be
                        given. See https://docs.aws.amazon.com/sagemaker/lates
                        t/dg/training-metrics.html. (default: None)
  --enable_sagemaker_metrics, -m
                        Enables SageMaker Metrics Time Series. See https://doc
                        s.aws.amazon.com/sagemaker/latest/dg/training-
                        metrics.html. (default: False)

Download:
  --output_path OUTPUT_PATH, -o OUTPUT_PATH
                        Local path to download the outputs to. (default: None)
  --download_state      Download the state once task is finished (default:
                        False)
  --download_model      Download the model once task is finished (default:
                        False)
  --download_output     Download the output once task is finished (default:
                        False)

Anything after "--" (followed by a space) will be passed as-is to the executed
script command line
```
```bash  
$ ssm process -h
usage: ssm process [-h] --project_name PROJECT_NAME --task_name TASK_NAME
                   [--bucket_name BUCKET_NAME] [--code CODE]
                   [--entrypoint ENTRYPOINT [ENTRYPOINT ...]]
                   [--dependencies DEPENDENCIES [DEPENDENCIES ...]]
                   [--command COMMAND [COMMAND ...]]
                   [--instance_type INSTANCE_TYPE]
                   [--instance_count INSTANCE_COUNT]
                   [--volume_size VOLUME_SIZE] [--max_run_mins MAX_RUN_MINS]
                   [--aws_repo_name AWS_REPO_NAME] [--repo_name REPO_NAME]
                   [--image_tag IMAGE_TAG]
                   [--docker_file_path_or_content DOCKER_FILE_PATH_OR_CONTENT]
                   [--framework FRAMEWORK]
                   [--framework_version FRAMEWORK_VERSION]
                   [--input_path INPUT_PATH [INPUT_PATH ...]]
                   [--input_s3 INPUT_S3 [INPUT_S3 ...]]
                   [--input_task INPUT_TASK [INPUT_TASK ...]]
                   [--force_running] [--clean_state] [--keep_state]
                   [--tag key value] [--env key value]
                   [--arguments ARGUMENTS [ARGUMENTS ...]]
                   [--output_path OUTPUT_PATH] [--download_state]
                   [--download_model] [--download_output]

optional arguments:
  -h, --help            show this help message and exit
  --project_name PROJECT_NAME, -p PROJECT_NAME
                        Project name. (default: None)
  --task_name TASK_NAME, -t TASK_NAME
                        Task name. (default: None)
  --bucket_name BUCKET_NAME, -b BUCKET_NAME
                        S3 bucket name (a default one is used if not given).
                        (default: None)

Code:
  --code CODE           An S3 URI or a local path to a file with the framework
                        script to run. (default: None)
  --entrypoint ENTRYPOINT [ENTRYPOINT ...], -e ENTRYPOINT [ENTRYPOINT ...]
                        The entrypoint for the processing job (default: None).
                        This is in the form of a list of strings that make a
                        command (default: None)
  --dependencies DEPENDENCIES [DEPENDENCIES ...], -d DEPENDENCIES [DEPENDENCIES ...]
                        A list of paths to directories (absolute or relative)
                        with any additional libraries that will be exported to
                        the container The library folders will be copied to
                        SageMaker in the same folder where the entrypoint is
                        copied. (default: None)
  --command COMMAND [COMMAND ...]
                        The command to run, along with any command-line flags
                        (defaults to: "python3"). (default: None)

Instance:
  --instance_type INSTANCE_TYPE, --it INSTANCE_TYPE
                        Type of EC2 instance to use. (default: ml.m5.large)
  --instance_count INSTANCE_COUNT, --ic INSTANCE_COUNT
                        Number of EC2 instances to use. (default: 1)
  --volume_size VOLUME_SIZE, -v VOLUME_SIZE
                        Size in GB of the EBS volume to use for storing input
                        data. Must be large enough to store input data.
                        (default: 30)
  --max_run_mins MAX_RUN_MINS
                        Timeout in minutes for running. After this amount of
                        time Amazon SageMaker terminates the job regardless of
                        its current status. (default: 1440)

Image:
  --aws_repo_name AWS_REPO_NAME, --ar AWS_REPO_NAME
                        Name of ECS repository. (default: None)
  --repo_name REPO_NAME, --rn REPO_NAME
                        Name of local repository. (default: None)
  --image_tag IMAGE_TAG
                        Image tag. (default: latest)
  --docker_file_path_or_content DOCKER_FILE_PATH_OR_CONTENT, --df DOCKER_FILE_PATH_OR_CONTENT
                        Path to a directory containing the DockerFile. The
                        base image should be set to `__BASE_IMAGE__` within
                        the Dockerfile, and is automatically replaced with the
                        correct base image. (default: None)
  --framework FRAMEWORK, -f FRAMEWORK
                        The framework to use, see https://github.com/aws/deep-
                        learning-containers/blob/master/available_images.md
                        (default: sklearn)
  --framework_version FRAMEWORK_VERSION, --fv FRAMEWORK_VERSION
                        The framework version (default: 0.20.0)

Running:
  --force_running       Force running the task even if its already completed.
                        (default: False)
  --tag key value       Tag to be attached to the jobs executed for this task.
                        (default: None)
  --env key value       Environment variables for the running task. (default:
                        None)
  --arguments ARGUMENTS [ARGUMENTS ...]
                        A list of string arguments to be passed to a
                        processing job. Arguments can also be provided after
                        "--" (followed by a space), which may be needed for
                        parameters with dashes (default: None)

I/O:
  --input_path INPUT_PATH [INPUT_PATH ...], -i INPUT_PATH [INPUT_PATH ...]
                        INPUT: PATH [DISTRIBUTION] [SUBDIR] Local/s3 path for
                        the input data. If a local path is given, it will be
                        synced to the task folder on the selected S3 bucket
                        before launching the task. (default: None)
  --input_s3 INPUT_S3 [INPUT_S3 ...], --iis INPUT_S3 [INPUT_S3 ...]
                        INPUT_S3: INPUT_NAME S3_URI [DISTRIBUTION] [SUBDIR]
                        Additional S3 input sources (a few can be given).
                        (default: None)
  --input_task INPUT_TASK [INPUT_TASK ...], --iit INPUT_TASK [INPUT_TASK ...]
                        INPUT_TASK: INPUT_NAME TASK_NAME TYPE [DISTRIBUTION]
                        [SUBDIR] Use an output of a completed task in the same
                        project as an input source (a few can be given). Type
                        should be one of ['state', 'model', 'source',
                        'output']. (default: None)
  --clean_state, --cs   Clear the task state before running it. The task will
                        be running again even if it was already completed
                        before. (default: False)
  --keep_state, --ks    Keep the current task state. If the task is already
                        completed, its current output will be taken without
                        running it again. (default: True)

Download:
  --output_path OUTPUT_PATH, -o OUTPUT_PATH
                        Local path to download the outputs to. (default: None)
  --download_state      Download the state once task is finished (default:
                        False)
  --download_model      Download the model once task is finished (default:
                        False)
  --download_output     Download the output once task is finished (default:
                        False)
```

Running a shell based task is very similar, except for `source_dir` and `entry_point` which are replaced by
`dir_files` and `cmd_line`, respectively. Run `ssm shell -h` for more details.

To manage the data of an existing command:
```bash
$ ssm data -h 

usage: ssm data [-h] --project_name PROJECT_NAME --task_name TASK_NAME
                [--bucket_name BUCKET_NAME] [--output_path OUTPUT_PATH]
                [--download_state] [--download_model] [--download_output]

optional arguments:
  -h, --help            show this help message and exit
  --project_name PROJECT_NAME, -p PROJECT_NAME
                        Project name.
  --task_name TASK_NAME, -t TASK_NAME
                        Task name.
  --bucket_name BUCKET_NAME, -b BUCKET_NAME
                        S3 bucket name (a default one is used if not given).
  --output_path OUTPUT_PATH, -o OUTPUT_PATH
                        Local path to download the outputs to.
  --download_state      Download the state once task is finished
  --download_model      Download the model once task is finished
  --download_output     Download the output once task is finished
```

# A fully featured advanced example
And now to a real advanced and fully featured version, yet simple to implement.
In order to exemplify most of the possible features, the following files are used in [CLI Example 6_1](https://github.com/shiftan/simple_sagemaker/tree/master/examples/readme_examples/example6):
```
.
|-- Dockerfile
|-- code
|   |-- internal_dependency
|   |   `-- lib2.py
|   |-- requirements.txt
|   `-- worker6.py
|-- data
|   |-- sample_data1.txt
|   `-- sample_data2.txt
`-- external_dependency
    `-- lib1.py
```

- Dockerfile - the dockerfile specifying how to extend the pre-built image
    ```bash
    # __BASE_IMAGE__ is automatically replaced with the correct base image
    FROM __BASE_IMAGE__ 
    RUN pip3 install pandas==0.25.3 scikit-learn==0.21.3
    ```
- code - the source code folder
    - internal_dependency - a dependency that is part of the source code folder
    - requirements.txt - pip requirements file lists needed packages to be installed before running the worker
        ```bash
        transformers==3.0.2
        ```
    - worker6.py - the worker code
- data - input data files
- external_dependency - additional code dependency

The code is then launched a few time by [run.sh](https://github.com/shiftan/simple_sagemaker/tree/master/examples/readme_examples/run.sh), to demonstrate different features:
```bash

# Example 6_1 - a complete example part 1. 
#   - Uses local data folder as input, that is distributed among instances (--i, ShardedByS3Key)
#   - Uses a public s3 bucket as an additional input (--iis)
#   - Builds a custom docker image (--df, --repo_name, --aws_repo_name)
#   - Hyperparameter task_type
#   - 2 instance (--ic)
#   - Use an on-demand instance (--no_spot)
ssm run -p simple-sagemaker-example-cli -t task6-1 -s $BASEDIR/example6/code -e worker6.py \
    -i $BASEDIR/example6/data ShardedByS3Key \
    --iis persons s3://awsglue-datasets/examples/us-legislators/all/persons.json \
    --df $BASEDIR/example6 --repo_name "task6_repo" --aws_repo_name "task6_repo" --no_spot \
    --ic 2 --task_type 1 -o $1/example6_1

# Example 6_2 - a complete example part 2.
#   - Uses outputs from part 1 (--iit)
#   - Uses additional local code dependencies (-d)
#   - Uses the tensorflow framework as pre-built image (-f)
#   - Tags the jobs (--tag)
#   - Defines sagemaker metrics (-m, --md)
ssm run -p simple-sagemaker-example-cli -t task6-2 -s $BASEDIR/example6/code -e worker6.py \
    -d $BASEDIR/example6/external_dependency --iit task_6_1_model task6-1 model \
    --iit task_6_1_state task6-1 state ShardedByS3Key \
    -f tensorflow -m --md "Score" "Score=(.*?);" --tag "MyTag" "MyValue" \
    --ic 2 --task_type 2 -o $1/example6_2 &

# Running task6_1 again
#   - A completed task isn't executed again, but the current output is used instead. 
#       --ks (keep state, the default) is used to keep the current state
ssm run -p simple-sagemaker-example-cli -t task6-1 -s $BASEDIR/example6/code -e worker6.py \
    -i $BASEDIR/example6/data ShardedByS3Key \
    --iis persons s3://awsglue-datasets/examples/us-legislators/all/persons.json \
    --df $BASEDIR/example6 --repo_name "task6_repo" --aws_repo_name "task6_repo" \
    --ic 2 --task_type 1 -o $1/example6_1 > $1/example6_1_2_stdout --ks &


wait # wait for all processes
```

[`worker6.py`](https://github.com/shiftan/simple_sagemaker/blob/master/examples/readme_examples/example6/code/worker6.py) contains the following:
```python
    logger.info("Score=10;")
    time.sleep(60)  # sleep to be able to see the two scores
    logger.info("Score=20;")
```
This get captured by the `"Score=(.*?);"` regular expression in the `ssm` command above, then the metrics graphs can be viewed on the AWS console:

![Metrics example](https://github.com/shiftan/simple_sagemaker/blob/master/docs/metric_example.jpg?raw=true "Metric Example")

More information can be found [here](https://docs.aws.amazon.com/sagemaker/latest/dg/training-metrics.html).

Feel free to dive more into the [files of this example](https://github.com/shiftan/simple_sagemaker/tree/master/examples/readme_examples/example6). Specifically, note how the [same worker code](https://github.com/shiftan/simple_sagemaker/tree/master/examples/readme_examples/example6/code/worker6.py) is used for the two parts, and the `task_type` hyperparameter is used to distinguish between the two. 

# More examples
CLI based examples:
- [A fully featured advanced example](#A-fully-featured-advanced-example)
- [Passing command line arguments](#Passing-command-line-arguments)
- [Task state and output](#Task-state-and-output)
- [Providing input data](#Providing-channel-data)
- [Chaining tasks](#Chaining-tasks)
- [Configuring the docker image](#Configuring-the-docker-image)
- [Defining code dependencies](#Defining-code-dependencies)

API based example:
- [Single file example](#Single-file-example)

## Passing command line arguments
Any extra argument passed to the command line in the form of --[KEY_NAME] [VALUE] is passed as an hyperparameter, and anything after "--" (followed by a space) in passed as-is to the executed script command line. hyperparameters are accessible for the **worker** by the `hps` dictionary within the environment configuration or just by parsing the command time argument of the running script (e.g. sys.argv).
For example, see the following worker code `worker2.py`:
```python
from worker_toolkit import worker_lib

worker_config = worker_lib.WorkerConfig(False)
print("-***-", worker_config.hps["msg"])
```
Runner CLI:
```bash
ssm run -p simple-sagemaker-example-cli -t task2 -e worker2.py --msg "Hello, world!" -o ./output/example2
```
Output from the log file
```
Invoking script with the following command:

/opt/conda/bin/python worker2.py --msg Hello, world!

Hello, world!
```
## Task state and output


A complete example can be seen in `worker3.py`:
```python
import os

from worker_toolkit import worker_lib

worker_config = worker_lib.WorkerConfig(False)

open(os.path.join(worker_config.output_data_dir, "output_data_dir"), "wt").write(
    "output_data_dir file"
)
open(os.path.join(worker_config.model_dir, "model_dir"), "wt").write("model_dir file")
open(os.path.join(worker_config.state, "state_dir"), "wt").write("state_dir file")

# 0 retcode - marks the tasks as completed, to allow other tasks to use its output, 
# and to avoid re-running it (unless enforced)
```
Runner CLI:
```bash
ssm run -p simple-sagemaker-example-cli -t task3 -e worker3.py -o ./output/example3
```
Output from the log file
```
Invoking script with the following command:

/opt/conda/bin/python worker2.py --msg Hello, world!

Hello, world!
```

## Providing input data
A **Job** can be configured to get a few data channels:
* A single local path can be used with the `-i/--input_path` argument. This path is synchronized to the **task** directory on the S3 bucket before running the **task**. On the **worker** side the data is accessible in `worker_config.channel_data`
* Additional S3 paths (many) can be set as well. Each input source is provided with `--iis [name] [S3 URI]`, and is accessible by the worker with `worker_config.channel_[name]` when [name] is the same one as was provided on the command line.
* Setting an output of a another **task** on the same **project**, see below ["Chaining tasks"](#Chaining-tasks)

Assuming a local `data` folder containing a single `sample_data.txt` file, a complete example can be seen in `worker4.py`:
```python
import logging
import subprocess
import sys

from worker_toolkit import worker_lib

logger = logging.getLogger(__name__)


def listDir(path):
    logger.info(f"*** START listing files in {path}")
    logger.info(
        subprocess.run(
            ["ls", "-la", "-R", path], stdout=subprocess.PIPE, universal_newlines=True
        ).stdout
    )
    logger.info(f"*** END file listing {path}")


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout)
    worker_config = worker_lib.WorkerConfig(False)
    listDir(worker_config.channel_data)
    listDir(worker_config.channel_bucket)
```
Running command:
```bash
ssm run -p simple-sagemaker-example-cli -t task4 -e worker4.py -i ./data \
    --iis bucket s3://awsglue-datasets/examples/us-legislators/all/persons.json -o ./output/example4
```
Output from the log file
```
...
INFO:__main__:*** START listing files in /opt/ml/input/data/data
INFO:__main__:/opt/ml/input/data/data:
total 12
drwxr-xr-x 2 root root 4096 Sep 14 21:51 .
drwxr-xr-x 4 root root 4096 Sep 14 21:51 ..
-rw-r--r-- 1 root root   19 Sep 14 21:51 sample_data.txt

INFO:__main__:*** END file listing /opt/ml/input/data/data
INFO:__main__:*** START listing files in /opt/ml/input/data/bucket
INFO:__main__:/opt/ml/input/data/bucket:
total 7796
drwxr-xr-x 2 root root    4096 Sep 14 21:51 .
drwxr-xr-x 4 root root    4096 Sep 14 21:51 ..
-rw-r--r-- 1 root root 7973806 Sep 14 21:51 persons.json

INFO:__main__:*** END file listing /opt/ml/input/data/bucket
...
```

## Chaining tasks
The output of a completed **task** on the same **project** can be used as an input to another **task**, by using the `--iit [name] [task name] [output type]` command line parameter, where:
- [name] - is the name of the input source, accessible by the worker with `worker_config.channel_[name]`
- [task name] - the name of the **task** whose output is used as input 
- [output type] - the **task** output type, one of "model", "output", "state"

Using the model output of *task3* and the same `worker4.py` code, we can now run:
```bash
ssm run -p simple-sagemaker-example-cli -t task5 -e worker4.py --iit bucket task3 model -o ./output/example5
```

And get the following output from in the log file:
```
INFO:__main__:*** START listing files in 
INFO:__main__:
INFO:__main__:*** END file listing 
INFO:__main__:*** START listing files in /opt/ml/input/data/bucket
INFO:__main__:/opt/ml/input/data/bucket:
total 12
drwxr-xr-x 2 root root 4096 Sep 14 21:55 .
drwxr-xr-x 3 root root 4096 Sep 14 21:55 ..
-rw-r--r-- 1 root root  128 Sep 14 21:55 model.tar.gz

INFO:__main__:*** END file listing /opt/ml/input/data/bucket
```

## Configuring the docker image
The image used to run a task can either be selected from a [pre-built ones](https://github.com/aws/deep-learning-containers/blob/master/available_images.md) 
or extended with additional Dockerfile commands.
The `framework`, `framework_version` and `py_version` CLI parameters are used to define the pre-built image, then if a path to a directory containing the Dockerfile is given by `docker_file_path_or_content`, it used along with `aws_repo_name`, `repo_name` and `image_tag` to build and push an image to ECS, and then set it as the used image.
The base image should be set to `__BASE_IMAGE__` within the Dockerfile, and is automatically replaced with the correct base image (according to the provided parameters above) before building it.
The API parameter for the Dockerfile path is named `docker_file_path_or_content` and allows to provide the content of the Dockerfile, e.g. 
```python
dockerFileContent = """
# __BASE_IMAGE__ is automatically replaced with the correct base image
FROM __BASE_IMAGE__
RUN pip3 install pandas==1.1 scikit-learn==0.21.3
"""
```
Sample usages:
1. [CLI Example 6_1](https://github.com/shiftan/simple_sagemaker/tree/master/examples/readme_examples/example6)- a CLI example launched by [run.sh](https://github.com/shiftan/simple_sagemaker/tree/master/examples/readme_examples/run.sh)
2. [single file example](https://github.com/shiftan/simple_sagemaker/tree/master/examples/single_file/example.py) - API with Dockerfile content
2. [single task example](https://github.com/shiftan/simple_sagemaker/tree/master/examples/single_task/example.py) - API with Dockerfile path

## Defining code dependencies
Additional local code dependencies can be specified with the `dependencies` CLI/API parameters. These dependencies are packed along with
the source code, and are extracted on the root code folder in run time.

Sample usages:
1. [CLI Example 6_2](https://github.com/shiftan/simple_sagemaker/tree/master/examples/readme_examples/example6)- a CLI example launched by [run.sh](https://github.com/shiftan/simple_sagemaker/tree/master/examples/readme_examples/run.sh)
2. [single task example](https://github.com/shiftan/simple_sagemaker/tree/master/examples/single_task/example.py) - API

## Single file example
A [single file example](https://github.com/shiftan/simple_sagemaker/tree/master/examples/single_file/example.py) can be found in the [examples directory](https://github.com/shiftan/simple_sagemaker/tree/master/examples).
First, define the **runner**:
```python
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
    task1_name = "task1"+postfix
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
```
An additional **task** that depends on the previous one can now be scheduled as well:
```python
    # *** Task 2 - process the results of Task 1
    task2_name = "task2"+postfix
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
```

Then, the worker code (note: the same function is used for the two different **tasks**, depending on the `task` hyperparameter):
```python
def worker():
    from worker_toolkit import worker_lib

    logger.info("Starting worker...")
    # parse the arguments
    worker_config = worker_lib.WorkerConfig()

    logger.info(f"Hyperparams: {worker_config.hps}")
    logger.info(f"Input data files: {list(Path(worker_config.channel_data).rglob('*'))}")
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
```

To pack everything in a single file, we use the command line argument `--worker` (as defined in the `runner` function) to distinguish between **runner** and worker runs
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
Running the file, with a sibling directory named `data` with a sample file [as on the example](https://github.com/shiftan/simple_sagemaker/tree/master/examples/single_file/data), prduces the following outputs for Task 1:
```
INFO:__main__:Hyperparams: {'arg': 'hello world!', 'task': 1, 'worker': 1}
INFO:__main__:Input data files: [PosixPath('/opt/ml/input/data/data/sample_data1.txt')]
INFO:__main__:State files: [PosixPath('/state/algo-1')]
INFO:worker_toolkit.worker_lib:Creating instance specific state dir
INFO:__main__:finished!
```

```
INFO:__main__:Hyperparams: {'arg': 'hello world!', 'task': 1, 'worker': 1}
INFO:__main__:Input data files: [PosixPath('/opt/ml/input/data/data/sample_data2.txt')]
INFO:__main__:State files: [PosixPath('/state/algo-2')]
INFO:worker_toolkit.worker_lib:Creating instance specific state dir
INFO:__main__:finished!
```

And the following for Task 2:
```
INFO:__main__:Hyperparams: {'arg': 'hello world!', 'task': 2, 'worker': 1}
INFO:__main__:Input data files: [PosixPath('worker_toolkit'), PosixPath('example.py'), PosixPath('worker_toolkit/worker_lib.py'), PosixPath('worker_toolkit/__pycache__'), PosixPath('worker_toolkit/__init__.py'), PosixPath('worker_toolkit/__pycache__/__init__.cpython-38.pyc'), PosixPath('worker_toolkit/__pycache__/worker_lib.cpython-38.pyc')]
INFO:__main__:State files: [PosixPath('/state/algo-1')]
INFO:__main__:Input task2_data: [PosixPath('/opt/ml/input/data/task2_data/model.tar.gz')]
INFO:__main__:Input task2_data_dist: [PosixPath('/opt/ml/input/data/task2_data_dist/model.tar.gz')]
INFO:worker_toolkit.worker_lib:Creating instance specific state dir
```

```
INFO:__main__:Hyperparams: {'arg': 'hello world!', 'task': 1, 'worker': 1}
INFO:__main__:Input data files: [PosixPath('/opt/ml/input/data/data/sample_data2.txt')]
INFO:__main__:State files: [PosixPath('/state/algo-2')]
INFO:worker_toolkit.worker_lib:Creating instance specific state dir
INFO:__main__:finished!

```

As mentioned, the complete code can be found in [this directory](https://github.com/shiftan/simple_sagemaker/tree/master/examples/single_file), 


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
7. If the tests succeed, the new version is published to [PyPi](https://pypi.org/project/simple-sagemaker/)


# Open issues
1. S3_sync doesn't delete remote files if deleted locally. Optimization may be needed as well.
2. Bug: If arguments after "--" (followed by a space) are used, please initialize `WorkerConfig` object with `update_argv=True` (the default) before parsing the command line arguments, e.g before calling `parser.parse_args()`.
3. Known issues: State directory can't have too many files (how many? seems to be a SageMaker bug. TBD: open a bug)
5. Handling spot instance / timeout termination / signals (seems to be opened bug [a bug in deep-learning-containers](https://github.com/aws/deep-learning-containers/issues/632))
6. Full documentation of the APIs (Readme / Read the docs + CLI?)
7. Add support for additional SageMaker features:
    - Processing job (partial implementation exists)
    - [Built in algorithms](https://docs.aws.amazon.com/sagemaker/latest/dg/algos.html)
    - More [frameworks](https://sagemaker.readthedocs.io/en/stable/frameworks/index.html)
    - [Experiments](https://docs.aws.amazon.com/sagemaker/latest/dg/experiments.html)
    - [Debugger](https://docs.aws.amazon.com/sagemaker/latest/dg/train-debugger.html)
    - [Automatic Tuning](https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning.html)
