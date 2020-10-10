#! /bin/bash
set -e # fail if any test fails

# Params: [output] [prefix] [suffix] [additional ssm params...]
BASEDIR=$(dirname "$0")
echo "Running with", $@

# Example 1 - hello world
ssm run -p ${2}simple-sagemaker-example-cli${3} -t cli-task1 -e $BASEDIR/worker1.py -o $1/example1 --it ml.p3.2xlarge --no_spot --ic 2 ${@:4} --max_run_mins 15 &

# Example 6_1 - a complete example part 1. 
#   - Uses local data folder as input, that is distributed among instances (--i, ShardedByS3Key)
#   - Uses a public s3 bucket as an additional input (--iis)
#   - Builds a custom docker image (--df, --repo_name, --aws_repo_name)
#   - Hyperparameter task_type
#   - 2 instance (--ic)
#   - Use an on-demand instance (--no_spot)
ssm run -p ${2}simple-sagemaker-example-cli${3} -t cli-task6-1 -s $BASEDIR/example6/code -e worker6.py \
    -i $BASEDIR/example6/data ShardedByS3Key --iis persons s3://awsglue-datasets/examples/us-legislators/all/persons.json \
    --df $BASEDIR/example6 --repo_name "task6_repo" --aws_repo_name "task6_repo" --no_spot \
    --download_state --download_model --download_output --max_run_mins 15 \
    --ic 2 --task_type 1 -o $1/example6_1 ${@:4} &

wait # wait for all processes

# Shell example
ssm shell -p ${2}simple-sagemaker-example-cli${3} -t shell-task --cmd_line "cat /proc/cpuinfo && nvidia-smi" -o $1/example_cmd --it ml.p3.2xlarge ${@:4} --max_run_mins 15 &

# Example 6_2 - a complete example part 2.
#   - Uses outputs from part 1 (--iit)
#   - Uses additional local code dependencies (-d)
#   - Uses the tensorflow framework as pre-built image (-f)
#   - Tags the jobs (--tag)
#   - Defines sagemaker metrics (-m, --md)
ssm run -p ${2}simple-sagemaker-example-cli${3} -t cli-task6-2 -s $BASEDIR/example6/code -e worker6.py \
    -d $BASEDIR/example6/external_dependency --iit task_6_1_model cli-task6-1 model --iit task_6_1_state cli-task6-1 state ShardedByS3Key \
    -f tensorflow -m --md "Score" "Score=(.*?);" --tag "MyTag" "MyValue" \
    --download_state --download_model --download_output --max_run_mins 15 \
    --ic 2 --task_type 2 -o $1/example6_2 ${@:4} &

wait # wait for all processes

# Run task6_1 again
#   The rest of arguments ${@:4} (specifying --force_running) aren't passed here, to demonstrate that existing output is used, without running the task again
ssm run -p ${2}simple-sagemaker-example-cli${3} -t cli-task6-1 -s $BASEDIR/example6/code -e worker6.py \
    -i $BASEDIR/example6/data ShardedByS3Key --iis persons s3://awsglue-datasets/examples/us-legislators/all/persons.json \
    --df $BASEDIR/example6 --repo_name "task6_repo" --aws_repo_name "task6_repo" \
    --download_state --download_model --download_output --max_run_mins 15 \
    --ic 2 --task_type 1 -o $1/example6_1 > $1/example6_1_2_stdout &

wait # wait for all processes
