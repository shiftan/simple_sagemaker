#! /bin/bash

set -e # stop and fail if anything stops
BASEDIR=$(dirname "$0")
DATA_DIR=${1:-~/proj/data/cv/imagenet}
cd $BASEDIR

# Download the code from PyTorch's examples repository
[ -f code/main.py ] || wget -O code/main.py https://raw.githubusercontent.com/pytorch/examples/master/imagenet/main.py

# Download and extract the data
./code/download.sh $DATA_DIR
./code/extract.sh $DATA_DIR

# Train on a single GPU, 7 epochs
echo ===== Training 7 epochs, a single GPU...
python ./code/main.py --epochs 7 $DATA_DIR 

# "Distributed training" on 1 GPU, 7 epochs
echo ===== Training 7 epochs, distributed, a single GPU...
export MASTER_PORT=8888
export MASTER_ADDR=localhost
python ./code/main.py --multiprocessing-distributed  --dist-url env:// --world-size 1 --rank 0 --seed 123 --epochs 1 $DATA_DIR 
