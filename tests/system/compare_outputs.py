import difflib
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
        "INFO:__main__:",
        "INFO:worker_toolkit.worker_lib:",
    )
    consts2 = [re.escape(x) for x in consts2]
    pattern2 = "|".join(consts2)
    splited = re.split(f"({pattern})", content, flags=re.MULTILINE)
    i = 0
    while i < len(splited):
        if splited[i] == constants.TEST_LOG_LINE_BLOCK_PREFIX:
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


def compareLog(expectedfile_path, outputfile_path):
    res = list()
    expected_content = expectedfile_path.read_text()
    output_content = outputfile_path.read_text()
    expected_blocks = getRelevantLogBlocks(expected_content)
    output_blocks = getRelevantLogBlocks(output_content)

    if len(output_blocks) != len(expected_blocks):
        res.append(
            f"len(output_blocks) and len(expected_blocks) doesn't match, {len(output_blocks)} {len(expected_blocks)}"
        )
    for (block_exp, block_out) in zip(expected_blocks, output_blocks):
        lines_exp = block_exp.splitlines()
        lines_out = block_out.splitlines()
        if block_exp.startswith("listing files"):
            if len(lines_exp) != len(lines_out) or lines_exp[0] != lines_out[0]:
                res.append(
                    f"Output of file listing doen't match, {len(lines_exp)} {len(lines_exp)} *{lines_exp[0]}* *{lines_out[0]}*"
                )
            for (line_exp, line_out) in zip(lines_exp[1:], lines_out[1:]):
                # 'ls -la' output
                if re.match("[drwx\\-]{10}", line_exp):
                    if line_exp.split(" ")[-1] != line_out.split(" ")[-1]:
                        res.append(f"different file, *{line_exp}* *{line_out}*")
                else:
                    if line_exp != line_out:
                        res.append(f"different line, *{line_exp}* *{line_out}*")
        elif block_exp.startswith("Worker config: Namespace"):
            pass  # TBD: add RE to compare
        elif block_exp.startswith("Env: environ"):
            pass  # TBD: add RE to compare
        else:
            if lines_exp != lines_out:
                res.append(f"different lines, *{lines_exp}* *{lines_out}*")

    return res


def _getAllProcesses(ps__elf_file):
    pattern = "\\d\\d:\\d\\d:\\d\\d"
    return set(
        [
            re.split(pattern, line)[-1].strip()
            for line in ps__elf_file.read_text().splitlines()[1:]
        ]
    )


def comparePsElfOutput(expectedfile_path, outputfile_path):
    processes_exp = _getAllProcesses(expectedfile_path)
    processes_out = _getAllProcesses(outputfile_path)
    if processes_exp != processes_out:
        print(processes_exp - processes_out)
        print(processes_out - processes_exp)
        return False

    return True


def compareFileContent(expectedfile_path, outputfile_path, file_name):
    differences = list()
    differences_info = list()
    if expectedfile_path.is_dir():
        pass
    if "logs/" in file_name or "_stdout" in file_name:
        compare_logs_res = compareLog(expectedfile_path, outputfile_path)
        if compare_logs_res:
            differences.append(f"{file_name} doesn't match")
            differences_info.extend(compare_logs_res)
    elif file_name.endswith(".tar.gz"):
        pass
    elif file_name.endswith("-manifest") or file_name.endswith("init-config.json"):
        pass
    elif "/config/" in file_name and (
        "debughookconfig.json" in file_name
        or "hyperparameters.json" in file_name
        or "trainingjobconfig.json" in file_name
        or "tensorboardoutputconfig.json" in file_name
    ):
        pass  # TBD: check these as well
    elif "ps__elf" in file_name:
        if not comparePsElfOutput(expectedfile_path, outputfile_path):
            differences.append(f"{file_name} doesn't match")

    else:
        if expectedfile_path.read_text() != outputfile_path.read_text():
            differences.append(f"{file_name} doesn't match")
    return differences, differences_info


def _isAsExpected(output_path, expected_path):
    logger.info(f"Comparing {output_path} and {expected_path}")
    res = []
    logs_diff_info = dict()

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
        differences, diff_info = compareFileContent(
            expectedfile_path, outputfile_path, file_name
        )
        res.extend(differences)
        if diff_info:
            logs_diff_info[file_name] = diff_info

    # log differences
    for line in res:
        logger.error(line)
    for key, lines in logs_diff_info.items():
        logger.warning(f"**** Difference for {key}")
        for line in lines:
            logger.warning(line)
    if not res:
        logger.info("Output matches expected output")
    return len(res) == 0


class OutputComparison:
    def __init__(self):
        self.extractors = list()

    @staticmethod
    def getSortedFileList(path, filters):
        path = Path(path)
        files = [str(x.relative_to(path)) for x in path.rglob("*") if x.is_file()]
        return set(filter(partial(applyFilter, filters=filters), files))

    def registerExtractor(self, extractor):
        self.extractors.append(extractor)

    def compare(self, root_path1, root_path2, ignore_filters=[]):
        res = []
        logs_diff_info = {}

        filters = ["source/", ".tar.gz", ".sagemaker-uploading", ".extracted"]
        files1 = OutputComparison.getSortedFileList(root_path1, filters)
        files2 = OutputComparison.getSortedFileList(root_path2, filters)
        if files1 != files2:
            if files2 - files1:
                res.append(f"**** Not in first: {files2-files1}")
            if files1 - files2:
                res.append(f"**** Not in second: {files1-files2}")

        for file_name in files1 & files2:
            file_name1 = root_path1 / file_name
            file_name2 = root_path2 / file_name

            for extractor in self.extractors:
                if extractor.match(file_name):
                    extracteds = list()
                    for file_to_extract in (file_name1, file_name2):
                        extracted = extractor.extract(file_to_extract.read_text())
                        extracteds.append(extracted)
                        if extracted:
                            extracted_file_name = Path(
                                str(file_to_extract) + ".extracted"
                            )
                            extracted_file_name.write_text(extracted)
                    if extracteds[0] != extracteds[1]:
                        res.append(f"**** {file_name} doesn't match")
                        html_diff = difflib.HtmlDiff()
                        diff = html_diff.make_file(
                            extracteds[0].splitlines(),
                            extracteds[1].splitlines(),
                            fromdesc=file_name1,
                            todesc=file_name2,
                        )
                        Path(str(file_name2) + ".extracted.diff.html").write_text(diff)
                        delta = difflib.unified_diff(
                            extracteds[0].splitlines(),
                            extracteds[1].splitlines(),
                            fromfile=str(file_name1),
                            tofile=str(file_name2),
                        )
                        logs_diff_info[file_name] = "\n" + "\n".join(iter(delta))

                    break
            else:
                res.append(f"**** No extractor for {file_name}")
        # log differences
        for line in res:
            logger.error(line)
        for key, lines in logs_diff_info.items():
            logger.warning(f"**** Difference for {key}")
            logger.warning(lines)
        if not res:
            logger.info("**** Output matches expected output")
        return len(res) == 0


class Extractor:
    def __init__(self):
        pass

    def extract(self, content):
        return content

    def match(self, filename):
        return True


class PsElfExtractor(Extractor):
    def extract(self, content):
        pattern = "\\d\\d:\\d\\d:\\d\\d"
        procs = [
            re.split(pattern, line)[-1].strip() for line in content.splitlines()[1:]
        ]
        return "\n".join(sorted(procs))

    def match(self, filename):
        return "ps__elf" in filename


class IgnoreExtractor(Extractor):
    def extract(self, content):
        return ""

    def match(self, filename):
        if filename.endswith(".tar.gz"):
            return True
        if filename.endswith("-manifest") or filename.endswith("init-config.json"):
            return True
        if "/config/" in filename and (
            "debughookconfig.json" in filename
            or "hyperparameters.json" in filename
            or "trainingjobconfig.json" in filename
            or "tensorboardoutputconfig.json" in filename
            or "processingjobconfig.json" in filename
        ):
            return True
        return False


class LogExtractor(Extractor):
    def __init__(self):
        self.block_markers = (
            constants.TEST_LOG_LINE_BLOCK_PREFIX,
            constants.TEST_LOG_LINE_BLOCK_SUFFIX,
        )
        self.line_markers = (
            constants.TEST_LOG_LINE_PREFIX,
            "INFO:__main__:",
            "INFO:worker_toolkit.worker_lib:",
        )

    def getRelevantBlocks(self, content):
        res = list()
        block_markers = [re.escape(x) for x in self.block_markers]
        pattern_blocks = "|".join(block_markers)

        line_markers = [re.escape(x) for x in self.line_markers]
        pattern_lines = "|".join(line_markers)

        splited = re.split(f"({pattern_blocks})", content, flags=re.MULTILINE)
        i = 0
        while i < len(splited):
            if splited[i] == self.block_markers[0]:
                res.append(splited[i + 1])
                i += 3
            else:
                relevant_lines = re.findall(
                    f"({pattern_lines})(.*)", splited[i], flags=re.MULTILINE
                )
                if relevant_lines:
                    res.extend([x[1] for x in relevant_lines if x])
                i += 1
        return [x for x in res if x]

    def extract(self, content):
        relevant_blocks = self.getRelevantBlocks(content)
        output = list()

        for block in relevant_blocks:
            if block.startswith("listing files"):
                dir_list_output = list()
                for line in block.splitlines():
                    # 'ls -la' output
                    if re.match("[drwx\\-]{10}", line):
                        if "sagemaker-uploading" not in line:
                            dir_list_output.append(line.split(" ")[-1])
                    else:
                        dir_list_output.append(line)
                output.append("\n".join(dir_list_output))
            elif block.startswith("Worker config: Namespace"):
                pass  # TBD: add RE to compare
            elif block.startswith("Env: environ"):
                pass  # TBD: add RE to compare
            else:
                output.append(block)

        return "\n****\n".join(output)

    def match(self, filename):
        return "logs/" in filename or "_stdout" in filename


def isAsExpected(output_path, expected_path):
    oc = OutputComparison()
    oc.registerExtractor(LogExtractor())
    oc.registerExtractor(PsElfExtractor())
    oc.registerExtractor(IgnoreExtractor())
    oc.registerExtractor(Extractor())
    # oc.compare("/home/user/proj/simple_sagemaker/examples/out2/1", "/home/user/proj/simple_sagemaker/examples/out2/2")
    return oc.compare(Path(expected_path).resolve(), Path(output_path).resolve())


def main():
    print("...")


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    aa = Path(__file__)
    examplesDir = aa.parent.parent.parent / "examples"
    if False:
        paths = [x.name for x in (examplesDir / "out").glob("*") if x.is_dir()]
        for path in paths:
            exp = examplesDir / path / "expected_output"
            out = examplesDir / "out" / path / "output"
            print(exp, out, isAsExpected(out, exp))
    else:
        exp = examplesDir / "single_task" / "expected_output"
        out = Path(
            "/home/user/proj/simple_sagemaker/.tox/single_proc/tmp/test_single_task0/single_task/output"
        )
        print(exp, out, isAsExpected(out, exp))
