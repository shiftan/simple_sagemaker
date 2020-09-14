import logging
import re
import sys
from functools import partial
from pathlib import Path

from simple_sagemaker import constants

logger = logging.getLogger(__name__)


def applyFilter(file, filters):
    for filter in filters:
        if filter in file:
            return False
    return True


def getSortedFileList(path, filters):
    path = Path(path)
    files = [str(x)[len(str(path)) + 1 :] for x in path.rglob("*") if x.is_file()]
    return set(filter(partial(applyFilter, filters=filters), files))


def getRelevantLogBlocks(content):
    res = list()
    consts = constants.TEST_LOG_LINE_BLOCK_PREFIX, constants.TEST_LOG_LINE_BLOCK_SUFFIX
    consts = [re.escape(x) for x in consts]
    pattern = "|".join(consts)
    consts2 = (
        constants.TEST_LOG_LINE_PREFIX,
        "INFO:__main__",
        "INFO:task_toolkit.algo_lib:",
    )
    consts2 = [re.escape(x) for x in consts2]
    pattern2 = "|".join(consts2)
    splited = re.split(f"({pattern})", content, flags=re.MULTILINE)
    i = 0
    while i < len(splited):
        if splited[i][1:] == constants.TEST_LOG_LINE_BLOCK_PREFIX:
            res.append(splited[i + 1])
            i += 3
        else:
            relevant_lines = re.findall(
                f"({pattern2})(.*)", splited[i], flags=re.MULTILINE
            )
            if relevant_lines:
                res.extend([x[1] for x in relevant_lines if x])
            i += 1
    return [x for x in res if x]


def compareLog(expected_content, output_content):
    expected_blocks = getRelevantLogBlocks(expected_content)
    output_blocks = getRelevantLogBlocks(output_content)
    if len(output_blocks) != len(expected_blocks):
        return False
    for (block_exp, block_out) in zip(expected_blocks, output_blocks):
        lines_exp = block_exp.splitlines()
        lines_out = block_exp.splitlines()
        if block_exp.startswith("listing files in "):
            if len(lines_exp) != len(lines_out) or lines_exp[0] != lines_out[0]:
                return False
            for (line_exp, line_out) in zip(lines_exp[1:], lines_out[1:]):
                # 'ls -la' output
                if re.match("[drwx\\-]{10}", line_exp):
                    if line_exp.split(" ")[-1] != line_out.split(" ")[-1]:
                        return False
                else:
                    if line_exp != line_out:
                        return False
        else:
            if lines_exp != lines_out:
                return False

    return True


def compareFileContent(expectedfile_path, outputfile_path, file_name):
    differences = []
    if expectedfile_path.is_dir():
        pass
    if "logs/" in file_name or "_stdout" in file_name:
        if not compareLog(expectedfile_path.read_text(), outputfile_path.read_text()):
            differences.append(f"{file_name} doesn't match")
    elif file_name.endswith(".tar.gz"):
        pass
    elif file_name.endswith("-manifest") or file_name.endswith("init-config.json"):
        pass
    elif "/config/" in file_name and (
        "debughookconfig.json" in file_name
        or "hyperparameters.json" in file_name
        or "trainingjobconfig.json" in file_name
    ):
        pass
    elif "__COMPLETED__" in file_name:
        if (
            expectedfile_path.read_text().split("-")[0]
            != outputfile_path.read_text().split("-")[0]
        ):
            differences.append(f"{file_name} doesn't match")
    else:
        if expectedfile_path.read_text() != outputfile_path.read_text():
            differences.append(f"{file_name} doesn't match")
    return differences


def isAsExpected(output_path, expected_path):
    logger.info(f"Comparing {output_path} and {expected_path}")
    res = []

    # compare the two list of output files, except for the source directory and tars
    filters = ["source/", ".tar.gz", ".sagemaker-uploading"]
    outputFiles = getSortedFileList(output_path, filters)
    expectedFiles = getSortedFileList(expected_path, filters)
    if expectedFiles != outputFiles:
        res.append(f"Not in output: {expectedFiles-outputFiles}")
        res.append(f"Not in expected: {outputFiles-expectedFiles}")

    # compare files content
    for file_name in expectedFiles & outputFiles:
        expectedfile_path = Path(expected_path) / file_name
        outputfile_path = Path(output_path) / file_name
        differences = compareFileContent(expectedfile_path, outputfile_path, file_name)
        res.extend(differences)

    # log differences
    for line in res:
        logger.error(line)
    if not res:
        logger.info("Output matches expected output")
    return len(res) == 0


def main():
    print("...")


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    aa = Path(__file__)
    examplesDir = aa.parent.parent.parent / "examples"
    if False:
        outs = [
            "test_single_task0",
            "test_single_file_tasks0",
            "test_multiple_tasks0",
            "test_readme_examples0",
            "test_cli_multi0",
        ]
        exps = [
            "single_task",
            "single_file",
            "multiple_tasks",
            "readme_examples",
            "cli_multi",
        ]
        for exp, out in zip(exps, outs):
            exp = examplesDir / exp / "expected_output"
            out = examplesDir / "out" / out / "output"
            print(exp, out, isAsExpected(out, exp))
    else:
        exp = examplesDir / "readme_examples" / "expected_output"
        out = examplesDir / "readme_examples" / "output"
        print(exp, out, isAsExpected(out, exp))
