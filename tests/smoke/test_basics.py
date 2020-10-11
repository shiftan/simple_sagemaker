import logging
import os
import platform
import shutil
import subprocess
import sys
from time import time

import boto3

from ..system.compare_outputs import isAsExpected

file_path = os.path.split(__file__)[0]
examples_path = os.path.abspath(os.path.join(file_path, "..", "..", "examples"))


def test_project(caplog, tmp_path):
    caplog.set_level(logging.INFO)
    logging.info("test_project")

    from simple_sagemaker.sm_project import SageMakerProject

    sm_project = SageMakerProject(project_name="test")
    sm_project = sm_project


def test_task(caplog, tmp_path):
    caplog.set_level(logging.INFO)
    logging.info("test_task")

    from simple_sagemaker.sm_task import SageMakerTask

    boto3_session = boto3.Session()
    image_uri = None
    smTask = SageMakerTask(boto3_session, "taskName", image_uri, prefix="tests/smoke")
    smTask = smTask


def _testCliInternal(cmd):
    shell_cmd = subprocess.run(cmd, shell=True)
    print("**************", shell_cmd)
    assert shell_cmd.returncode == 0


def test_cli_help():
    _testCliInternal("ssm -h")


def test_cli_run_help():
    _testCliInternal("ssm run -h")


def test_cli_shell_help():
    _testCliInternal("ssm shell -h")


def test_cli_data_help():
    _testCliInternal("ssm data -h")


def _internalTestCli(test_path, caplog, tmp_path):
    caplog.set_level(logging.INFO)
    print("Temp path:", tmp_path)
    print("Running cli:", test_path)

    output_path = os.path.join(tmp_path, test_path, "output_smoke")
    # remove current local output
    shutil.rmtree(output_path, ignore_errors=True)
    # prefix/suffix for project name
    py_version_string = f"py{sys.version_info.major}{sys.version_info.minor}"
    time_string = int(time())
    postfix = f"-{os.name}-{time_string}-{py_version_string}"
    prefix = "tests_smoke/"

    if platform.system() == "Linux":
        run_shell = os.path.join(examples_path, test_path, "run_smoke.sh")
    elif platform.system() == "Windows":
        run_shell = os.path.join(examples_path, test_path, "run_smoke.bat")
    subprocess.run(
        [run_shell, output_path, prefix, postfix, "--cs --force_running"], check=True
    )

    expected_path = os.path.join(examples_path, test_path, "expected_output_smoke")
    assert isAsExpected(output_path, expected_path)


def test_readme_examples(caplog, tmp_path):
    # Windows can't currently work due to lack of support in running linux images
    # Mac can't currently work as it doesn't have a docker engine
    if platform.system() in ["Linux"]:
        _internalTestCli("readme_examples", caplog, tmp_path)
