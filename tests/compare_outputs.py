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


def compareFileContent(expectedFilePath, outputFilePath, fileName):
    differences = []
    if expectedFilePath.is_dir():
        pass
    if "logs/" in fileName:
        pass
    elif fileName.endswith(".tar.gz"):
        pass
    elif fileName.endswith("-manifest") or fileName.endswith("init-config.json"):
        pass
    elif "/config/" in fileName and (
        "debughookconfig.json" in fileName
        or "hyperparameters.json" in fileName
        or "trainingjobconfig.json" in fileName
    ):
        pass
    elif "__COMPLETED__" in fileName:
        if (
            expectedFilePath.read_text().split("-")[0]
            != outputFilePath.read_text().split("-")[0]
        ):
            differences.append(f"{fileName} doesn't match")
    else:
        if expectedFilePath.read_text() != outputFilePath.read_text():
            differences.append(f"{fileName} doesn't match")
    return differences


def isAsExpected(outputPath, expectedPath):
    logger.info(f"Comparing {outputPath} and {expectedPath}")
    res = []

    # compare the two list of output files, except for the source directory and tars
    filters = ["source/", ".tar.gz", ".sagemaker-uploading"]
    outputFiles = getSortedFileList(outputPath, filters)
    expectedFiles = getSortedFileList(expectedPath, filters)
    if expectedFiles != outputFiles:
        res.append(f"Not in output: {expectedFiles-outputFiles}")
        res.append(f"Not in expected: {outputFiles-expectedFiles}")

    # compare files content
    for fileName in expectedFiles & outputFiles:
        expectedFilePath = Path(expectedPath) / fileName
        outputFilePath = Path(outputPath) / fileName
        differences = compareFileContent(expectedFilePath, outputFilePath, fileName)
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
