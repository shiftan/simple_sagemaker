#! /bin/bash

# Params: [output] [prefix] [suffix] [additional ssm params...]
BASEDIR=$(dirname "$0")
ssm run -p ${2}simple-sagemaker-example-cli-multi${3} -t task1 -e $BASEDIR/worker.py -o $1/output1 --task_type 1 -i $BASEDIR/../single_file/data ${@:4}
ssm run -p ${2}simple-sagemaker-example-cli-multi${3} -t task2 -e $BASEDIR/worker.py -o $1/output2 --task_type 2 --iit task2_data task1 model ${@:4}
