#! /bin/bash
set -ex # fail if any test fails

# Params: [output] [prefix] [suffix] [additional ssm params...]
cd `dirname "$0"`
echo "Running with", -- $1 -- $2 -- $3 -- $4 -- $5

# Example 1 - a processing script + dependencies
ssm process --prefix ${2} -p ssm-example-processing -t cli-code${3} -o $1/output1 \
    --download_state --download_output --max_run_mins 15 \
    --code ex1.py --dependencies ./dep ${@:4} \
    -- arg1 -arg2 --arg3 "argument 4" &
pid1=$!

# Example 2 - a raw entrypoint with arguments
ssm process --prefix ${2} -p ssm-example-processing -t cli-shell${3} -o $1/output2 \
    --download_state --download_output --max_run_mins 15 \
    --entrypoint "/bin/bash" --dependencies ./dep --force_running \
    -- -c "echo ==Bash && \
echo \"-***- Args:\"\$@ &&echo \"-- Env:\"\`env\`&& \
echo \"*** START listing files\"&&ls -laR /opt&&echo \"*** END \"&& \
cp -r /opt/ml/config \$SSM_OUTPUT/config&& \
echo output>\$SSM_OUTPUT/output&& \
echo state>\$SSM_STATE/state" &


# Example 3 - a bash script that gets the output and state of cli-code as input
wait $pid1
ssm process --prefix ${2} -p ssm-example-processing -t cli-bash${3} -o $1/output3 \
    --download_state --command bash --download_output --max_run_mins 15 \
    -i ./data --iit cli_code_output cli-code${3} output --iit cli_code_state cli-code${3} state \
    --code ex3.sh --dependencies ./dep ${@:4} \
    -- arg1 -arg2 --arg3 "argument 4" &

# Example 3 - a shell training ecript that gets the output and state of cli-code as input
ssm shell --prefix ${2} -p ssm-example-processing -t shell-task${3} -o $1/output4 \
    --iit cli_code_output cli-code${3} output --iit cli_code_state cli-code${3} state \
    --cmd_line "echo '*** START listing files in /opt/ml' && ls -laR /opt/ml && echo '*** END file listing /opt/ml'" \
    --max_run_mins 15 ${@:4} &

#  --it ml.t3.medium

wait # wait for all processes

# Run: 
# tox -e bash -- ./run.sh ./output " " " " --cs