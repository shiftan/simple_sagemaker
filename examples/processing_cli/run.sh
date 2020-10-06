#! /bin/bash
set -e # fail if any test fails

# Params: [output] [prefix] [suffix] [additional ssm params...]
cd `dirname "$0"`
echo "Running with", $@

# Example 1 - processing script + dependencies
ssm process -p ssm-example-processing -t cli-code -o ./output/output1 \
    --download_state --download_output --max_run_mins 15 \
    --code ex1.py --dependencies ./dep --cs \
    -- arg1 -arg2 --arg3 "argument 4" &

# Example 2 - raw entrypoint with arguments
ssm process -p ssm-example-processing -t cli-shell -o ./output/output2 \
    --download_state --download_output --max_run_mins 15 \
    --entrypoint "/bin/bash" \
    -- -c 'pwd && ls -laR /opt && env && cp -r /opt/ml/config $SSM_OUTPUT/config && echo "output" > $SSM_OUTPUT/output && echo "state" > $SSM_STATE/state' &


#  --it ml.t3.medium

wait # wait for all processes
