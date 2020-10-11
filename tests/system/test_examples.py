import logging
import os
import shutil
import subprocess
import sys
from time import time

from .compare_outputs import isAsExpected

file_path = os.path.split(__file__)[0]
examples_path = os.path.abspath(os.path.join(file_path, "..", "..", "examples"))
sys.path.append(examples_path)


def _internalTestExample(caplog, tmp_path, runner):
    caplog.set_level(logging.INFO)
    # print(os.environ)
    print("Temp path:", tmp_path)
    print("Running", runner, runner.__name__, runner.__module__)

    example_path = os.path.dirname(runner.__code__.co_filename)
    output_path = os.path.join(tmp_path, os.path.split(example_path)[-1], "output")
    # remove current local output
    shutil.rmtree(output_path, ignore_errors=True)
    # prefix/suffix for project name
    py_version_string = f"py{sys.version_info.major}{sys.version_info.minor}"
    time_string = int(time())
    postfix = f"-{time_string}-{py_version_string}"
    prefix = "tests/"

    sm_project = runner(postfix=postfix, prefix=prefix, output_path=output_path)
    sm_project = sm_project
    # sm_project.cleanFolder()

    expected_path = os.path.join(example_path, "expected_output")
    # check for expected_output also one level up
    if not os.path.isdir(expected_path):
        expected_path = os.path.join(os.path.dirname(example_path), "expected_output")

    assert isAsExpected(output_path, expected_path)


def _internalTestCli(test_path, caplog, tmp_path):
    caplog.set_level(logging.INFO)
    print("Temp path:", tmp_path)
    print("Running cli:", test_path)

    output_path = os.path.join(tmp_path, test_path, "output")
    # remove current local output
    shutil.rmtree(output_path, ignore_errors=True)
    # prefix/suffix for project name
    py_version_string = f"py{sys.version_info.major}{sys.version_info.minor}"
    time_string = int(time())
    postfix = f"-{time_string}-{py_version_string}"
    prefix = "tests/"

    run_shell = os.path.join(examples_path, test_path, "run.sh")
    subprocess.run(
        [run_shell, output_path, prefix, postfix, "--cs --force_running"], check=True
    )

    expected_path = os.path.join(examples_path, test_path, "expected_output")
    assert isAsExpected(output_path, expected_path)


def skip_test_cli_multi(caplog, tmp_path):
    _internalTestCli("cli_multi", caplog, tmp_path)


def test_readme_examples(caplog, tmp_path):
    _internalTestCli("readme_examples", caplog, tmp_path)


def test_processing_cli_examples(caplog, tmp_path):
    _internalTestCli("processing_cli", caplog, tmp_path)


def test_multiple_tasks(caplog, tmp_path):
    from multiple_tasks.example import runner

    _internalTestExample(caplog, tmp_path, runner)


def test_single_file_tasks(caplog, tmp_path):
    from single_file.example import runner

    _internalTestExample(caplog, tmp_path, runner)


def test_single_task(caplog, tmp_path):
    from single_task.example import runner

    _internalTestExample(caplog, tmp_path, runner)
