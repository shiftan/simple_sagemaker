#! /bin/bash
#set -e # fail if any test fails

# Params: [output] [prefix] [suffix] [additional ssm params...]
BASEDIR=$(dirname "$0")

# Example 2 - passing hyperparams as command line arguments
ssm run -p ${2}simple-sagemaker-example-cli${3} -t task2 -e $BASEDIR/worker2.py --msg "Hello, world!" -o $1/example2 ${@:4} --max_run_mins 15 &

# Example 3 - outputs
ssm run -p ${2}simple-sagemaker-example-cli${3} -t task3 -e $BASEDIR/worker3.py -o $1/example3 ${@:4} --max_run_mins 15 &

wait # wait for all processes, to avoid AWS resource limits... :(

# Example 4 - Inputs, using a local data directory + s3 bucket
ssm run -p ${2}simple-sagemaker-example-cli${3} -t task4 -e $BASEDIR/worker4.py \
    -i $BASEDIR/data --iis bucket s3://awsglue-datasets/examples/us-legislators/all/persons.json \
    --max_run_mins 15 -o $1/example4 ${@:4} &

# running task3 again
ssm run -p ${2}simple-sagemaker-example-cli${3} -t task3 -e $BASEDIR/worker3.py -o $1/example3_2 ${@:4} --ks > $1/example3_2_stdout --max_run_mins 15 &

wait # wait for all processes

# Example 5 - chaining data, using task3's output
ssm run -p ${2}simple-sagemaker-example-cli${3} -t task5 -e $BASEDIR/worker4.py --iit bucket task3 model -o $1/example5 ${@:4} --max_run_mins 15 &

wait # wait for all processes
