from worker_toolkit import worker_lib

args = worker_lib.parseArgs()
print("-***-", args.hps["msg"])
