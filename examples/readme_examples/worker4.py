import subprocess

from task_toolkit import algo_lib


def listDir(path):
    print(
        subprocess.run(
            ["ls", "-la", "-R", path], stdout=subprocess.PIPE, universal_newlines=True
        ).stdout
    )


args = algo_lib.parseArgs()
listDir(args.input_data)
listDir(args.input_bucket)
