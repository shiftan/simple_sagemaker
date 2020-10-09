#! /bin/bash

echo "======= Starting Bash script ..."
echo "-***- Args:" $@
echo "-- Env:", `env`
echo "-***- Pwd:", `pwd`
echo "*** START listing files in /opt"
ls -laR /opt
echo "*** END file listing /opt"
cp -r /opt/ml/config $SSM_OUTPUT/config 
echo "output" > $SSM_OUTPUT/output_sh 
echo "state" > $SSM_STATE/state_sh