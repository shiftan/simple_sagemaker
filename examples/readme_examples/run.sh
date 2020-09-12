#! /bin/bash

# Params: [output] [prefix] [suffix] [additional ssm params...]
BASEDIR=$(dirname "$0")

# Example 1
ssm -p ${2}simple-sagemaker-example-cli${3} -t task1 -e $BASEDIR/worker1.py -o $1 ${@:4}
# Example 2
ssm -p ${2}simple-sagemaker-example-cli${3} -t task1 -e $BASEDIR/worker2.py -o $1 ${@:4}
