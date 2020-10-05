#! /bin/bash

set -e # stop and fail if anything stops
BASEDIR=$(dirname "$0")
cd $BASEDIR

# Download the code from PyTorch's examples repository
[ -f code/main.py ] || wget -O code/main.py https://raw.githubusercontent.com/pytorch/examples/master/imagenet/main.py

EPOCHS=20
ADDITIONAL_ARGS="--no_spot --force_running"

# Download the data
ssm shell -p ex-imagenet -t download \
    --dir_files ./code -o ./output/download \
    --no_spot \
    --cmd_line './download.sh $SSM_STATE/data'

SETUP='./extract.sh $SM_CHANNEL_TRAIN/.. && \ 
        CODE_DIR=`pwd` && cd $SSM_INSTANCE_STATE'

# Train on a single GPU, $EPOCHS epochs
echo ===== Training $EPOCHS epochs, a single GPU...
ssm shell -p ex-imagenet -t train-1gpu \
    --dir_files ./code -o ./output/train-1gpu \
    --iit train download state FullyReplicated data/train \
    --iit val download state FullyReplicated data/val \
    --download_model --download_output \
    --it ml.p3.2xlarge $ADDITIONAL_ARGS \
    --cmd_line  $SETUP'python $CODE_DIR/main.py --epochs '$EPOCHS' $SM_CHANNEL_TRAIN/..' &
    
# "Distributed training" on 1 GPU, $EPOCHS epochs
echo ===== Training $EPOCHS epochs, distributed, a single GPU...
ssm shell -p ex-imagenet -t train-dist-1gpu \
    --dir_files ./code -o ./output/train-dist-1gpu \
    --iit train download state FullyReplicated data/train \
    --iit val download state FullyReplicated data/val \
    --download_model --download_output \
    --it ml.p3.2xlarge $ADDITIONAL_ARGS \
    --cmd_line $SETUP'python $CODE_DIR/main.py --multiprocessing-distributed --dist-url env:// --world-size 1 --rank 0 --seed 123 --epochs '$EPOCHS' $SM_CHANNEL_TRAIN/..' &


# $EPOCHS epochs "Distributed training" on a single node with 4 GPUs
echo ===== Training $EPOCHS epochs, distributed, 8 GPUs...
ssm shell -p ex-imagenet -t train-dist-8gpus \
    --dir_files ./code -o ./output/train-dist-8gpus \
    --iit train download state FullyReplicated data/train \
    --iit val download state FullyReplicated data/val \
    --download_model --download_output \
    --it ml.p2.8xlarge $ADDITIONAL_ARGS \
    --cmd_line $SETUP'python $CODE_DIR/main.py --multiprocessing-distributed --dist-url env:// --world-size 1 --rank 0 --seed 123 --epochs '$EPOCHS' $SM_CHANNEL_TRAIN/..' &

wait

echo "FINISHED!"