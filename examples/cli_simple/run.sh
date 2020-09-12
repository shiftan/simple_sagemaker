#! /bin/bash

# Params: [output] [prefix] [suffix] [additional ssm params...]
BASEDIR=$(dirname "$0")
ssm -p ${2}simple-sagemaker-example-cli${3} -t task1 -e $BASEDIR/worker.py -o $1 ${@:4}
