import argparse
import collections
import logging
import os
import sys

import configargparse
from sagemaker.inputs import TrainingInput

from . import constants


def fileValidation(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    return arg


def fileOrS3Validation(parser, arg):
    if not arg.startswith("s3://") and not os.path.exists(arg):
        parser.error("%s has to be either a file or a s3 path!" % arg)
    return arg


S3InputTuple = collections.namedtuple("S3Input", ("input_name", "s3_uri"))
TaskInputTuple = collections.namedtuple("S3Input", ("input_name", "task_name", "type"))


class S3InputAction(argparse.Action):
    def __call__(self, parser, args, values, option_string=None):
        # print (f'{args} {values} {option_string}')
        # print ("****", values)
        if not values[1].startswith("s3://"):
            raise ValueError(f"{values[1]} has to be a s3 path!")

        if not args.__getattribute__(self.dest):
            setattr(args, self.dest, [S3InputTuple(*values)])
        else:
            args.__getattribute__(self.dest).append(S3InputTuple(*values))


class TaskInputAction(argparse.Action):
    def __call__(self, parser, args, values, option_string=None):
        # print (f'{args} {values} {option_string}')
        # print("****", values, hasattr(args, self.dest))
        taskInputTypes = ["state", "model", "source", "output"]
        if values[2] not in taskInputTypes:
            raise ValueError(f"{values[2]} has to be one of {taskInputTypes}!")

        if not args.__getattribute__(self.dest):
            setattr(args, self.dest, [TaskInputTuple(*values)])
        else:
            args.__getattribute__(self.dest).append(TaskInputTuple(*values))


def parseArgs():
    parser = configargparse.ArgParser(
        config_file_parser_class=configargparse.DefaultConfigFileParser
    )

    # general
    parser.add("--config-file", "-c", is_config_file=True, help="config file path")
    parser.add_argument("--project_name", "-p", required=True)
    parser.add_argument("--task_name", "-t", required=True)
    parser.add_argument("--bucket_name", "-b")
    # coding params
    parser.add_argument("--source_dir", "-s", type=lambda x: fileValidation(parser, x))
    parser.add_argument(
        "--entry_point", "-e", required=True, type=lambda x: fileValidation(parser, x)
    )
    parser.add_argument(
        "--dependencies", "-d", nargs="+", type=lambda x: fileValidation(parser, x)
    )
    # instance params
    parser.add_argument(
        "--instance_type", "--it", default=constants.DEFAULT_INSTANCE_TYPE
    )
    parser.add_argument(
        "--instance_count", "--ic", type=int, default=constants.DEFAULT_INSTANCE_COUNT
    )
    parser.add_argument(
        "--volume_size", "-v", type=int, default=constants.DEFAULT_VOLUME_SIZE
    )
    parser.add_argument("--no_spot", dest="use_spot", action="store_false")
    parser.add_argument("--use_spot", dest="use_spot", action="store_true")
    parser.set_defaults(use_spot=constants.DEFAULT_USE_SPOT)
    parser.add_argument("--max_wait", type=int, default=constants.DEFAULT_MAX_WAIT)
    parser.add_argument("--max_run", type=int, default=constants.DEFAULT_MAX_RUN)
    # image params
    parser.add_argument("--aws_repo", "--ar")
    parser.add_argument("--repo_name", "--rn")
    parser.add_argument("--image_tag", "--tag", constants.DEFAULT_REPO_TAG)
    parser.add_argument("--docker_file", "--df")
    # run params
    parser.add_argument(
        "--input_path", "-i", type=lambda x: fileOrS3Validation(parser, x)
    )
    parser.add_argument(
        "--input_s3",
        "--iis",
        action=S3InputAction,
        nargs=2,
        metavar=S3InputTuple._fields,
    )
    parser.add_argument(
        "--input_task",
        "--iit",
        action=TaskInputAction,
        nargs=3,
        metavar=TaskInputTuple._fields,
    )
    parser.add_argument("--clean_state", "--cs", default=False, action="store_true")
    parser.add_argument("--output_path", "-o", default=None)

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


def parseInputs(args, sm_project):
    if not args.input_task and not args.input_s3:
        return None

    inputs = dict()
    distribution = "FullyReplicated"
    if args.input_task:
        for (input_name, task_name, ttype) in args.input_task:
            inputs[input_name] = sm_project.getInputConfig(task_name, **{ttype: True})
    if args.input_s3:
        for (input_name, s3_uri) in args.input_s3:
            inputs[input_name] = TrainingInput(s3_uri, distribution=distribution)

        # sm_project.getInputConfig(task_name, distribution="ShardedByS3Key", state=True)
        # sm_project.getInputConfig(task_name, distribution="ShardedByS3Key", output=True)
    return inputs


def parseHyperparams(rest):
    res = dict()
    for i in range(0, len(rest), 2):
        res[rest[i].strip("-")] = rest[i + 1]
    return res


def main():
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
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
                "entry_point": "entryPoint",
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
                "max_wait": "maxWait",
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
                "docker_file": "docker_file_path_or_content",
            },
        )
    )

    sm_project.createIAMRole()
    image_uri = sm_project.buildOrGetImage(
        instance_type=sm_project.defaultInstanceParams.instance_type
    )

    running_params = getAllParams(
        args,
        {
            "input_path": "input_data_path",
            "clean_state": "clean_state",
        },
    )

    inputs = parseInputs(args, sm_project)
    hyperparameters = parseHyperparams(rest)

    sm_project.runTask(
        args.task_name,
        image_uri,
        distribution="ShardedByS3Key",  # distribute the input files among the workers
        hyperparameters=hyperparameters,
        additional_inputs=inputs,
        **running_params,
    )

    if args.output_path:
        sm_project.downloadResults(args.task_name, args.output_path, source=False)


if __name__ == "__main__":
    main()
