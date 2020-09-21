import argparse
import collections
import logging
import os
import sys

import configargparse
import sagemaker
from sagemaker.inputs import TrainingInput

from . import constants

logger = logging.getLogger(__name__)


def fileValidation(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    return arg


def fileOrS3Validation(parser, arg):
    if not arg.startswith("s3://") and not os.path.exists(arg):
        parser.error("%s has to be either a file or a s3 path!" % arg)
    return arg


InputTuple = collections.namedtuple("Input", ("path", "distribution"))
S3InputTuple = collections.namedtuple(
    "S3Input", ("input_name", "s3_uri", "distribution")
)
TaskInputTuple = collections.namedtuple(
    "TaskInput", ("input_name", "task_name", "type", "distribution")
)


def help_for_input_type(tuple, additional_text=""):
    res = f"{tuple.__name__}: {', '.join(tuple._fields[:-1])} [distribution]"
    if additional_text:
        res += "\n" + additional_text
    return res


class InputActionBase(argparse.Action):
    def __init__(self, option_strings, dest, tuple, nargs=None, **kwargs):
        self.__nargs = len(tuple._fields)
        self.__tuple = tuple
        super(InputActionBase, self).__init__(option_strings, dest, "+", **kwargs)

    def __append__(self, args, values):
        dist_options = ["FullyReplicated", "ShardedByS3Key"]
        default_dist = "FullyReplicated"
        if len(values) == self.__nargs - 1:
            values.append(default_dist)
        elif len(values) == self.__nargs:
            if values[-1] not in dist_options:
                raise argparse.ArgumentTypeError(
                    f"distribution has to be one of {dist_options}"
                )
        else:
            raise argparse.ArgumentTypeError(
                f"{self.dest} has to contain {self.__nargs}/{self.__nargs-1} arguments, got {values}"
            )
        value = self.__tuple(*values)
        if not args.__getattribute__(self.dest):
            setattr(args, self.dest, [value])
        else:
            args.__getattribute__(self.dest).append(value)


class InputAction(InputActionBase):
    def __call__(self, parser, args, values, option_string=None):
        if not os.path.exists(values[0]) and not values[0].startswith("s3://"):
            raise ValueError(f"{values[1]} has to be a local/s3 path!")

        self.__append__(args, values)


class S3InputAction(InputActionBase):
    def __call__(self, parser, args, values, option_string=None):
        if not values[1].startswith("s3://"):
            raise ValueError(f"{values[1]} has to be a s3 path!")

        self.__append__(args, values)


class TaskInputAction(InputActionBase):
    def __call__(self, parser, args, values, option_string=None):
        # print (f'{args} {values} {option_string}')
        # print("****", values, hasattr(args, self.dest))
        taskInputTypes = ["state", "model", "source", "output"]
        if values[2] not in taskInputTypes:
            raise ValueError(f"{values[2]} has to be one of {taskInputTypes}!")
        self.__append__(args, values)


def parseArgs():
    parser = configargparse.ArgParser(
        config_file_parser_class=configargparse.DefaultConfigFileParser,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # general
    parser.add("--config-file", "-c", is_config_file=True, help="Config file path.")
    parser.add_argument("--project_name", "-p", required=True, help="Project name.")
    parser.add_argument("--task_name", "-t", required=True, help="Task name.")
    parser.add_argument(
        "--bucket_name",
        "-b",
        help="S3 bucket name (a default one is used if not given).",
    )
    # coding params
    parser.add_argument(
        "--source_dir",
        "-s",
        type=lambda x: fileValidation(parser, x),
        help="""Path (absolute, relative or an S3 URI) to a directory with any other source
        code dependencies aside from the entry point file. If source_dir is an S3 URI,
        it must point to a tar.gz file. Structure within this directory are preserved when running on Amazon SageMaker.""",
    )
    parser.add_argument(
        "--entry_point",
        "-e",
        required=True,
        # type=lambda x: fileValidation(parser, x),
        help="""Path (absolute or relative) to the local Python source file which should be executed as the entry point.
        If source_dir is specified, then entry_point must point to a file located at the root of source_dir.""",
    )
    parser.add_argument(
        "--dependencies",
        "-d",
        nargs="+",
        type=lambda x: fileValidation(parser, x),
        help="""Path (absolute, relative or an S3 URI) to a directory with any other training source code dependencies
        aside from the entry point file. If source_dir is an S3 URI, it must point to a tar.gz file.
        Structure within this directory are preserved when running on Amazon SageMaker.""",
    )
    # instance params
    parser.add_argument(
        "--instance_type",
        "--it",
        default=constants.DEFAULT_INSTANCE_TYPE,
        help="Type of EC2 instance to use.",
    )
    parser.add_argument(
        "--instance_count",
        "--ic",
        type=int,
        default=constants.DEFAULT_INSTANCE_COUNT,
        help="Number of EC2 instances to use.",
    )
    parser.add_argument(
        "--volume_size",
        "-v",
        type=int,
        default=constants.DEFAULT_VOLUME_SIZE,
        help="""Size in GB of the EBS volume to use for storing input data.
        Must be large enough to store input data.""",
    )
    parser.add_argument(
        "--no_spot",
        dest="use_spot",
        action="store_false",
        help="Use on demand instances",
    )
    parser.add_argument(
        "--use_spot",
        dest="use_spot",
        action="store_true",
        help="""Specifies whether to use SageMaker Managed Spot instances.
    If enabled then the max_wait arg should also be set""",
    )
    parser.set_defaults(use_spot=constants.DEFAULT_USE_SPOT)
    parser.add_argument(
        "--max_wait",
        type=int,
        default=constants.DEFAULT_MAX_WAIT,
        help="""Timeout in seconds waiting for spot instances.
        After this amount of time Amazon SageMaker will stop waiting for Spot instances to become available.""",
    )
    parser.add_argument(
        "--max_run",
        type=int,
        default=constants.DEFAULT_MAX_RUN,
        help="""Timeout in seconds for running.
        After this amount of time Amazon SageMaker terminates the job regardless of its current status.""",
    )
    # image params
    parser.add_argument("--aws_repo", "--ar", help="Name of ECS repository.")
    parser.add_argument("--repo_name", "--rn", help="Name of local repository.")
    parser.add_argument(
        "--image_tag", default=constants.DEFAULT_REPO_TAG, help="Image tag."
    )
    parser.add_argument(
        "--docker_file_path",
        "--df",
        help="Path to a directory containing the DockerFile",
    )
    parser.add_argument(
        "--framework",
        "-f",
        help="The framework to use, see https://github.com/aws/deep-learning-containers/blob/master/available_images.md",
        choices=["pytorch", "tensorflow"],
        default="pytorch",
    )
    parser.add_argument(
        "--framework_version",
        "--fv",
        help="The framework version",
    )
    parser.add_argument(
        "--python_version",
        "--pv",
        help="The python version",
    )
    # run params
    parser.add_argument(
        "--input_path",
        "-i",
        action=InputAction,
        help=help_for_input_type(
            InputTuple,
            """Local/s3 path for the input data. If a local path is given, it will be sync'ed to the task
            folder on the selected S3 bucket before launching the task.""",
        ),
        tuple=InputTuple,
    )
    parser.add_argument(
        "--input_s3",
        "--iis",
        action=S3InputAction,
        help=help_for_input_type(
            S3InputTuple, "Additional S3 input sources (a few can be given)."
        ),
        tuple=S3InputTuple,
    )
    parser.add_argument(
        "--input_task",
        "--iit",
        action=TaskInputAction,
        help=help_for_input_type(
            TaskInputTuple,
            "Use an output of a completed task in the same project as an input source (a few can be given).",
        ),
        tuple=TaskInputTuple,
    )
    parser.add_argument(
        "--clean_state",
        "--cs",
        default=False,
        action="store_true",
        help="Clear the task state before running it. The task will be running again even if it was already completed before.",
    )
    parser.add_argument(
        "--keep_state",
        "--ks",
        action="store_false",
        dest="clean_state",
        help="Keep the current task state. If the task is already completed, its current output will \
             be taken without running it again.",
    )
    parser.add_argument(
        "--metric_definitions",
        "--md",
        nargs=2,
        metavar=("name", "regexp"),
        action="append",
        help="Name and regexp for a metric definition, a few can be given. \
            See https://docs.aws.amazon.com/sagemaker/latest/dg/training-metrics.html.",
    )
    parser.add_argument(
        "--enable_sagemaker_metrics",
        "-m",
        default=False,
        action="store_true",
        help="Enables SageMaker Metrics Time Series. \
            See https://docs.aws.amazon.com/sagemaker/latest/dg/training-metrics.html.",
    )
    parser.add_argument(
        "--tag",
        nargs=2,
        metavar=("key", "value"),
        action="append",
        help="Tag to be attached to the jobs executed for this task.",
    )

    parser.add_argument(
        "--output_path",
        "-o",
        default=None,
        help="Local path to download the outputs to.",
    )
    parser.add_argument(
        "--download_state",
        default=False,
        action="store_true",
        help="Download the state once task is finished",
    )
    parser.add_argument(
        "--download_model",
        default=False,
        action="store_true",
        help="Download the model once task is finished",
    )
    parser.add_argument(
        "--download_output",
        default=False,
        action="store_true",
        help="Download the output once task is finished",
    )

    args, rest = parser.parse_known_args()
    return args, rest


def addParam(args, argName, paramName, params):
    if hasattr(args, argName):
        arg = args.__getattribute__(argName)
        if arg is not None:
            params[paramName] = arg


def getAllParams(args, mapping):
    params = dict()
    for k, v in mapping.items():
        addParam(args, k, v, params)
    return params


def parseInputsAndAllowAccess(args, sm_project):
    input_data_path = None
    distribution = "FullyReplicated"
    if args.input_path:
        input_data_path, distribution = args.input_path[0]

    inputs = dict()
    if args.input_task:
        for (input_name, task_name, ttype, distribution) in args.input_task:
            inputs[input_name] = sm_project.getInputConfig(
                task_name, ttype, distribution=distribution
            )
    if args.input_s3:
        for (input_name, s3_uri, distribution) in args.input_s3:
            bucket, _ = sagemaker.s3.parse_s3_url(s3_uri)
            sm_project.allowAccessToS3Bucket(bucket)
            inputs[input_name] = TrainingInput(s3_uri, distribution=distribution)

    return input_data_path, distribution, inputs


def parseHyperparams(rest):
    res = dict()
    for i in range(0, len(rest), 2):
        res[rest[i].strip("-")] = rest[i + 1]
    return res


def main():
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logger.info(f"Running ssm cli, args:{sys.argv}")
    args, rest = parseArgs()

    file_path = os.path.split(__file__)[0]
    examples_path = os.path.abspath(os.path.join(file_path, ".."))
    sys.path.append(examples_path)
    from simple_sagemaker.sm_project import SageMakerProject

    sm_project = SageMakerProject(
        **getAllParams(
            args,
            {
                "project_name": "project_name",
                "bucket_name": "bucket_name",
            },
        )
    )
    sm_project.setDefaultCodeParams(
        **getAllParams(
            args,
            {
                "source_dir": "source_dir",
                "entry_point": "entry_point",
                "dependencies": "dependencies",
            },
        )
    )
    sm_project.setDefaultInstanceParams(
        **getAllParams(
            args,
            {
                "instance_count": "instance_count",
                "instance_type": "instance_type",
                "volume_size": "volume_size",
                "use_spot": "use_spot_instances",
                "max_run": "max_run",
                "max_wait": "max_wait",
            },
        )
    )
    sm_project.setDefaultImageParams(
        **getAllParams(
            args,
            {
                "aws_repo": "aws_repo_name",
                "repo_name": "repo_name",
                "image_tag": "img_tag",
                "docker_file_path": "docker_file_path_or_content",
                "framework": "framework",
                "framework_version": "version",
                "python_version": "py_version",
            },
        )
    )

    image_uri = sm_project.buildOrGetImage(
        instance_type=sm_project.defaultInstanceParams.instance_type
    )

    running_params = getAllParams(
        args,
        {
            "clean_state": "clean_state",
            "enable_sagemaker_metrics": "enable_sagemaker_metrics",
        },
    )

    input_data_path, distribution, inputs = parseInputsAndAllowAccess(args, sm_project)
    hyperparameters = parseHyperparams(rest)
    tags = {} if args.tag is None else {k: v for (k, v) in args.tag}
    metric_definitions = (
        {}
        if args.metric_definitions is None
        else {k: v for (k, v) in args.metric_definitions}
    )

    sm_project.runTask(
        args.task_name,
        image_uri,
        hyperparameters=hyperparameters,
        input_data_path=input_data_path,
        distribution=distribution,
        additional_inputs=inputs,
        tags=tags,
        metric_definitions=metric_definitions,
        **running_params,
    )

    if args.output_path:
        sm_project.downloadResults(
            args.task_name,
            args.output_path,
            logs=True,
            state=args.download_state,
            model=args.download_model,
            output=args.download_output,
        )


if __name__ == "__main__":
    main()
