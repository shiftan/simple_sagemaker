import logging
import os
import sys

import configargparse


def fileValidation(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return arg


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
    parser.add_argument("--instance_type", "--it", default="ml.m5.large")
    parser.add_argument("--instance_count", "--ic", type=int, default=1)
    parser.add_argument("--volume_size", "-v", type=int)
    parser.add_argument("--use_spot", default=True, type=bool)
    parser.add_argument("--max_wait", type=int)
    parser.add_argument("--max_run", type=int)
    # image params
    parser.add_argument("--aws_repo", "--ar")
    parser.add_argument("--repo_name", "--rn")
    parser.add_argument("--image_tag", "--tag")
    parser.add_argument("--docker_file", "--df")
    # run params
    parser.add_argument("--input_path", "-i", type=lambda x: fileValidation(parser, x))
    parser.add_argument("--clean_state", "--cs", default=False, action="store_true")
    parser.add_argument("--output_path", "-o", default=None)

    args = parser.parse_args()
    return args


def addParam(args, argName, paramName, params):
    if hasattr(args, argName):
        arg = args.__getattribute__(argName)
        if arg:
            params[paramName] = arg


def getAllParams(args, mapping):
    params = dict()
    for k, v in mapping.items():
        addParam(args, k, v, params)
    return params


def main():
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    args = parseArgs()

    filePath = os.path.split(__file__)[0]
    examplesPath = os.path.abspath(os.path.join(filePath, ".."))
    sys.path.append(examplesPath)
    from simple_sagemaker.sm_project import SageMakerProject

    smProject = SageMakerProject(
        **getAllParams(
            args,
            {
                "task_name": "projectName",
                "bucket_name": "bucketName",
            },
        )
    )
    smProject.setDefaultCodeParams(
        **getAllParams(
            args,
            {
                "source_dir": "sourceDir",
                "entry_point": "entryPoint",
                "dependencies": "dependencies",
            },
        )
    )
    smProject.setDefaultInstanceParams(
        **getAllParams(
            args,
            {
                "instance_count": "instanceCount",
                "instance_type": "instanceType",
                "volume_size": "volumeSize",
                "use_spot": "useSpotInstances",
                "max_run": "maxRun",
                "max_wait": "maxWait",
            },
        )
    )
    smProject.setDefaultImageParams(
        **getAllParams(
            args,
            {
                "aws_repo": "awsRepoName",
                "repo_name": "repoName",
                "image_tag": "imgTag",
                "docker_file": "dockerFilePathOrContent",
            },
        )
    )

    smProject.createIAMRole()
    imageUri = smProject.buildOrGetImage(
        instanceType=smProject.defaultInstanceParams.instanceType
    )

    runningParams = getAllParams(
        args,
        {
            "input_path": "inputDataPath",
            "clean_state": "cleanState",
        },
    )
    smProject.runTask(
        args.task_name,
        imageUri,
        distribution="ShardedByS3Key",  # distribute the input files among the workers
        hyperparameters={"worker": 1, "arg": "hello world!", "task": 1},
        **runningParams
    )

    if args.output_path:
        smProject.downloadResults(args.task_name, args.output_path, source=False)


if __name__ == "__main__":
    main()
