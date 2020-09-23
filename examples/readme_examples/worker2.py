from worker_toolkit import worker_lib

worker_config = worker_lib.WorkerConfig(False)
print("-***-", worker_config.hps["msg"])
