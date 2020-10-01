BASEDIR=$(dirname "$0")
pushd .
cd $BASEDIR

ssm run -p ssm-ex -t ex2 -e ssm_ex2.py -o ./out2 --it ml.p3.2xlarge --ic 2 --cs

cat ./out2/logs/logs0

popd