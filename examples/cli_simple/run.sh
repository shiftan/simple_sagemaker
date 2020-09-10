#! /bin/bash
BASEDIR=$(dirname "$0")
ssm -p simple-sagemaker-example-cli -t task1 -e $BASEDIR/worker.py --cs -o $BASEDIR/output
