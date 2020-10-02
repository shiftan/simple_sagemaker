set -e # stop and fail if anything stops
BASEDIR=$(dirname "$0")
pushd .
cd $BASEDIR

# Download the data
python cifar10.py --download_only --data_path ./data
# Train on a single node
python cifar10.py --data_path ./data \
    --test_batch_size 100 --train_batch_size 256 --num_workers 2
# Train distibuted
python cifar10.py --data_path ./data \
    --test_batch_size 100 --train_batch_size 256 --num_workers 2 \
    --distributed --backend nccl 

popd