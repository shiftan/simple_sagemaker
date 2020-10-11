#! /bin/bash
set -ex # fail if any test fails

cd `dirname "$0"`

echo "Running $0 with ", -- $1 -- $2 -- $3 -- $4 -- $5
OUTPUT=${1:-.}

ssm run --prefix ${2} -p ssm-debugging -t metrics${3} -e ./metrics.py -o $OUTPUT/output1 ${@:4} \
    --no_spot `#temporarily to accelerate iterations` &

ssm run --prefix ${2} -p ssm-debugging -t tensorboard${3} -s ./tensorboard -e lightning.py -o $OUTPUT/output2 ${@:4} \
    --no_spot `#temporarily to accelerate iterations` --force_running &


wait # wait for all processes
