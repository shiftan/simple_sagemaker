#! /bin/bash
set -e # fail if any test fails

# Params: [output] [prefix] [suffix] [additional ssm params...]
BASEDIR=$(dirname "$0")
echo "Running with", $@

# Example 1 - hello world
ssm -p ${2}simple-sagemaker-example-cli${3} -t task1 -e $BASEDIR/worker1.py -o $1/example1 --it ml.p3.2xlarge ${@:4} &

# Example 6_1 - a complete example part 1
ssm -p ${2}simple-sagemaker-example-cli${3} -t task6-1 -s $BASEDIR/example6/code -e worker6.py \
    -i $BASEDIR/example6/data ShardedByS3Key --iis persons s3://awsglue-datasets/examples/us-legislators/all/persons.json \
    --df $BASEDIR/example6 --repo_name "task6_repo" --aws_repo "task6_repo" \
    --ic 2 --task_type 1 -o $1/example6_1 ${@:4} &

wait # wait for all processes

# Example 6_2 - a complete example part 2 - uses outputs from part 1
ssm -p ${2}simple-sagemaker-example-cli${3} -t task6-2 -s $BASEDIR/example6/code -e worker6.py \
    -d $BASEDIR/example6/external_dependency --iit task_6_1_model task6-1 model --iit task_6_1_state task6-1 state ShardedByS3Key \
    --ic 2 --task_type 2 -o $1/example6_2 ${@:4} &


# running task6_1 again with --ks (keep state) to demonstrate tgat existing output is used, without running the task again
ssm -p ${2}simple-sagemaker-example-cli${3} -t task6-1 -s $BASEDIR/example6/code -e worker6.py \
    -i $BASEDIR/example6/data ShardedByS3Key --iis persons s3://awsglue-datasets/examples/us-legislators/all/persons.json \
    --df $BASEDIR/example6 --repo_name "task6_repo" --aws_repo "task6_repo" \
    --ic 2 --task_type 1 -o $1/example6_1 ${@:4} > $1/example6_1_2_stdout --ks &


wait # wait for all processes
