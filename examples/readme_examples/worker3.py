import os

from worker_toolkit import worker_lib

args = worker_lib.parseArgs()

open(os.path.join(args.output_data_dir, "output_data_dir"), "wt").write(
    "output_data_dir file"
)
open(os.path.join(args.model_dir, "model_dir"), "wt").write("model_dir file")
open(os.path.join(args.state, "state_dir"), "wt").write("state_dir file")

# Mark the tasks as completed, to allow other tasks using its output, and to avoid re-running it (unless enforced)
worker_lib.markCompleted(args)
