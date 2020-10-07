import os

from worker_toolkit import worker_lib

worker_config = worker_lib.WorkerConfig(False)

open(os.path.join(worker_config.output_data_dir, "output_data_dir"), "wt").write(
    "output_data_dir file"
)
open(os.path.join(worker_config.model_dir, "model_dir"), "wt").write("model_dir file")
open(os.path.join(worker_config.state, "state_dir"), "wt").write("state_dir file")

# The task is marked as completed, to allow other tasks to use its output,
# and to avoid re-running it (unless enforced)
