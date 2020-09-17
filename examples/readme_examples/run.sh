#! /bin/bash
#set -e # fail if any test fails

# Params: [output] [prefix] [suffix] [additional ssm params...]
BASEDIR=$(dirname "$0")

# Example 1 - hello world
ssm -p ${2}simple-sagemaker-example-cli${3} -t task1 -e $BASEDIR/worker1.py -o $1/example1 --it ml.p3.2xlarge ${@:4} &

# Example 2 - passing hyperparams as command line arguments
ssm -p ${2}simple-sagemaker-example-cli${3} -t task2 -e $BASEDIR/worker2.py --msg "Hello, world!" -o $1/example2 ${@:4} &

wait # wait for all processes, to avoid AWS resource limits... :(

# Example 3 - outputs
ssm -p ${2}simple-sagemaker-example-cli${3} -t task3 -e $BASEDIR/worker3.py -o $1/example3 ${@:4} &

# Example 4 - Inputs, using a local file + s3 bucket
ssm -p ${2}simple-sagemaker-example-cli${3} -t task4 -e $BASEDIR/worker4.py -i $BASEDIR/data --iis bucket s3://awsglue-datasets/examples/us-legislators/all/persons.json -o $1/example4 ${@:4} &

wait # wait for all processes

# running task3 again
ssm -p ${2}simple-sagemaker-example-cli${3} -t task3 -e $BASEDIR/worker3.py -o $1/example3_2 ${@:4} --ks > $1/example3_2_stdout &

# Example 5 - chaining data, using task3's output
ssm -p ${2}simple-sagemaker-example-cli${3} -t task5 -e $BASEDIR/worker4.py --iit bucket task3 model -o $1/example5 ${@:4} &

wait # wait for all processes

# Example 6_1 - a complete example
#--df $BASEDIR/example6/Dockerfile \
ssm -p ${2}simple-sagemaker-example-cli${3} -t task6-1 -s $BASEDIR/example6/code -e $BASEDIR/example6/code/worker6.py \
    -i $BASEDIR/example6/data ShardedByS3Key --iis persons s3://awsglue-datasets/examples/us-legislators/all/persons.json \
    --df $BASEDIR/example6/Dockerfile --repo_name "task6_repo", --aws_repo "task6_repo" \
    --task_type 1 -o $1/example6_1 ${@:4}
# Example 6_2 - a complete example
ssm -p ${2}simple-sagemaker-example-cli${3} -t task6-2 -s $BASEDIR/example6/code -e $BASEDIR/example6/code/worker6.py \
    -d $BASEDIR/example6/external_dependency --iit task_6_1_model task6-1 model --iit task_6_1_state task6-1 state ShardedByS3Key \
     --task_type 2 -o $1/example5 ${@:4} --ks &


wait # wait for all processes