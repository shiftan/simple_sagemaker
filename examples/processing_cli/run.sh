#! /bin/bash
set -e # fail if any test fails

# Params: [output] [prefix] [suffix] [additional ssm params...]
cd `dirname "$0"`
echo "Running with", $@

# Example 1 - a processing script + dependencies
ssm process -p ssm-example-processing -t cli-code -o ./output/output1 \
    --download_state --download_output --max_run_mins 15 \
    --code ex1.py --dependencies ./dep \
    -- arg1 -arg2 --arg3 "argument 4" &
pid1=$!

# Example 2 - a raw entrypoint with arguments
ssm process -p ssm-example-processing -t cli-shell -o ./output/output2 \
    --download_state --download_output --max_run_mins 15 \
    --entrypoint "/bin/bash" --dependencies ./dep \
    -- -c "echo '======= Bash script ...' && \
        echo 'Args:' $@ && echo Env: \`env\` && pwd && ls -laR /opt && \
        cp -r /opt/ml/config \$SSM_OUTPUT/config && \
        echo 'output' > \$SSM_OUTPUT/output && \
        echo 'state' > \$SSM_STATE/state" &

# Example 3 - a bash ecript that gets the output and state of cli-code as input
wait $pid1
ssm process -p ssm-example-processing -t cli-bash -o ./output/output3 \
    --download_state --command bash --download_output --max_run_mins 15 \
    -i ./data --iit cli_code_output cli-code output --iit cli_code_state cli-code state \
    --code ex3.sh --dependencies ./dep \
    -- arg1 -arg2 --arg3 "argument 4" &

# Example 3 - a shell training ecript that gets the output and state of cli-code as input
ssm shell -p ssm-example-processing -t shell-task -o ./output/output4 \
    --iit cli_code_output cli-code output --iit cli_code_state cli-code state \
    --cmd_line "ls -la /opt" \
    --max_run_mins 15 &

#  --it ml.t3.medium

wait # wait for all processes
