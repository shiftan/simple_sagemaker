import logging
import sys
from functools import partial
from pathlib import Path

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


def compareFileContent(expectedfile_path, outputfile_path, file_name):
    differences = []
    if expectedfile_path.is_dir():
        pass
    if "logs/" in file_name:
        pass
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
    sTask = aa.parent.parent / "examples" / "single_task"
    # sTask = aa.parent.parent/"examples"/"multiple_tasks"
    exp = sTask / "expected_output"
    out = sTask / "output"
    print(isAsExpected(out, exp))
