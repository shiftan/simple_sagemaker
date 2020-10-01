BASEDIR=$(dirname "$0")
pushd .
cd $BASEDIR

# Download the data
ssm run -p ex-cifar10 -t download -e cifar10.py --no_spot -- \
    --download_only

# Train on a single node
ssm run -p ex-cifar10 -t train-single -e cifar10.py \
    --iit cifar_data download state --it ml.p3.2xlarge

# Train distibuted
ssm run -p ex-cifar10 -t train-dist -e cifar10.py \
    --iit cifar_data download state --it ml.p3.2xlarge --ic 2 -- \
    --distributed --backend nccl

popd