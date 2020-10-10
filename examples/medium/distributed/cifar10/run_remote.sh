set -e # stop and fail if anything stops
BASEDIR=$(dirname "$0")
pushd .
cd $BASEDIR

# Download the data
ssm run -p ex-cifar10 -t download -e cifar10.py --no_spot --\
    --download_only

# Train on a single node
ssm run -p ex-cifar10 -t train-single -e cifar10.py \
     -m --md "Loss" "loss: ([0-9\\.]*)" --md "Accuracy" "Accuracy: ([0-9\\.]*)" \
    --no_spot `#temporarily to accelerate iterations` \
    --iit cifar_data download state --it ml.p3.2xlarge \
    `# Beginning of training script params` -- \
    --test_batch_size 100 --train_batch_size 256 --epochs 10 --num_workers 2 

# Train distibuted
ssm run -p ex-cifar10 -t train-dist -e cifar10.py \
    -m --md "Loss" "loss: ([0-9\\.]*)" --md "Accuracy" "Accuracy: ([0-9\\.]*)" \
    --no_spot `#temporarily to accelerate iterations` \
    --iit cifar_data download state --it ml.p3.2xlarge \
    --ic 2 \
    `# Beginning of training script params` -- \
    --test_batch_size 100 --train_batch_size 256 --epochs 10 --num_workers 2 \
    --distributed --backend nccl 

wait # wait for all processes

popd