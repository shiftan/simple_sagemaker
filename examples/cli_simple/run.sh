#! /bin/bash
BASEDIR=$(dirname "$0")
ssm -p ${1}simple-sagemaker-example-cli${2} -t task1 -e $BASEDIR/worker.py --cs -o $1
