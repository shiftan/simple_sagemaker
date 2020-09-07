import logging
import os
import shutil
import sys
from time import gmtime, strftime

from .compare_outputs import isAsExpected

filePath = os.path.split(__file__)[0]
examplesPath = os.path.abspath(os.path.join(filePath, "..", "..", "examples"))
sys.path.append(examplesPath)


def test_single_task(caplog, tmp_path):
    from single_task.example import runner

    _internalTestExample(caplog, tmp_path, runner)


def test_multiple_tasks(caplog, tmp_path):
    from multiple_tasks.example import runner

    _internalTestExample(caplog, tmp_path, runner)


def test_single_file_tasks(caplog, tmp_path):
    from single_file.code.example import runner

    _internalTestExample(caplog, tmp_path, runner)


def _internalTestExample(caplog, tmp_path, runner):
    caplog.set_level(logging.INFO)
    # print(os.environ)
    print("Temp path:", tmp_path)
    print("Running", runner, runner.__name__, runner.__module__)

    examplePath = os.path.dirname(runner.__code__.co_filename)
    outputPath = os.path.join(tmp_path, "output")
    # remove current local output
    shutil.rmtree(outputPath, ignore_errors=True)
    # prefix/suffix for project name
    pyVersionString = f"py{sys.version_info.major}{sys.version_info.minor}"
    timeString = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    postfix = f"_{timeString}_{pyVersionString}"
    prefix = "tests/"

    smProject = runner(postfix=postfix, prefix=prefix, outputPath=outputPath)
    smProject = smProject
    # smProject.cleanFolder()

    expectedPath = os.path.join(examplePath, "expected_output")
    # check for expected_output also one level up
    if not os.path.isdir(expectedPath):
        expectedPath = os.path.join(os.path.dirname(examplePath), "expected_output")

    assert isAsExpected(outputPath, expectedPath)
