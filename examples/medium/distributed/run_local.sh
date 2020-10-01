BASEDIR=$(dirname "$0")
pushd .
cd $BASEDIR

# Download the data
python cifar10.py --download_only --data_path ./data
# Train on a single node
python cifar10.py --data_path ./data
# Train distibuted
python cifar10.py --distributed --backend nccl --data_path ./data

popd