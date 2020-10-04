#! /bin/bash

set -e # stop and fail if anything stops
BASEDIR=$(dirname "$0")
cd $BASEDIR

# Download the code from PyTorch's examples repository
[ -f code/main.py ] || wget -O code/main.py https://raw.githubusercontent.com/pytorch/examples/master/imagenet/main.py

# Download the data
ssm shell -p ex-imagenet -t download \
    --dir_files ./code -o ./output/download \
    --no_spot \
    --cmd_line './download.sh $SSM_STATE/data'
    
# Train on a single GPU, 7 epochs
echo ===== Training 7 epochs, a single GPU...
ssm shell -p ex-imagenet -t train-1gpu \
    --dir_files ./code -o ./output/train-1gpu \
    --iit train download state FullyReplicated data/train \
    --iit val download state FullyReplicated data/val \
    --download_model --download_output \
    --no_spot --it ml.p3.2xlarge \
    --cmd_line './extract.sh $SM_CHANNEL_TRAIN/.. && \
                CODE_DIR=`pwd` && cd $SSM_INSTANCE_STATE && \
                python $CODE_DIR/main.py --epochs 7 $SM_CHANNEL_TRAIN/..' &
    
# "Distributed training" on 1 GPU, 7 epochs
echo ===== Training 7 epochs, distributed, a single GPU...
ssm shell -p ex-imagenet -t train-dist-1gpu \
    --dir_files ./code -o ./output/train-dist-1gpu \
    --iit train download state FullyReplicated data/train \
    --iit val download state FullyReplicated data/val \
    --download_model --download_output \
    --no_spot --it ml.p3.2xlarge \
    --cmd_line './extract.sh $SM_CHANNEL_TRAIN/.. && \
                CODE_DIR=`pwd` && cd $SSM_INSTANCE_STATE && \
                python $CODE_DIR/main.py --multiprocessing-distributed --dist-url env:// --world-size 1 --rank 0 --seed 123 --epochs 7 $SM_CHANNEL_TRAIN/..' &


# 7 epochs "Distributed training" on a single node with 4 GPUs
echo ===== Training 7 epochs, distributed, 4 GPUs...
ssm shell -p ex-imagenet -t train-dist-4gpus \
    --dir_files ./code -o ./output/train-dist-4gpus \
    --iit train download state FullyReplicated data/train \
    --iit val download state FullyReplicated data/val \
    --download_model --download_output \
    --no_spot --it ml.g4dn.12xlarge	 \
    --cmd_line './extract.sh $SM_CHANNEL_TRAIN/.. && \
                CODE_DIR=`pwd` && cd $SSM_INSTANCE_STATE && \
                python $CODE_DIR/main.py --multiprocessing-distributed --dist-url env:// --world-size 1 --rank 0 --seed 123 --epochs 7 $SM_CHANNEL_TRAIN/..' &

wait

echo "FINISHED!