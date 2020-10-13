#! /bin/bash

set -ex # stop and fail if anything stops
cd `dirname "$0"`

# Download the code from PyTorch's examples repository
[ -f code/main.py ] || wget -O main.py https://raw.githubusercontent.com/pytorch/examples/master/imagenet/main.py

# The dogs vs cats DB can be downloaded from
## Kaggle - https://www.kaggle.com/c/dogs-vs-cats
## Microsoft - https://www.microsoft.com/en-us/download/details.aspx?id=54765
## Floyhub - https://www.floydhub.com/fastai/datasets/cats-vs-dogs

# For simplicity, we currently just download a few sample images out of the full DB
if [ ! -d ./data ]; then
    mkdir -p data && cd data
    wget -O sample_data.tar "https://www.floydhub.com/api/v1/download/artifacts/data/VbpRSQnFkQmYaBUtwt3aca?is_dir=true&path=sample"
    tar xf sample_data.tar && mv valid val && cd ..
fi

# Train on a single node
#   We're as the data set is small (sample data) -i switch makes sense here, other approaches may be better for larger sets.
ssm shell -p cat-vs-dogs -t 1-node -o ./output/output_1node --download_state \
  -i ./data --it ml.p3.2xlarge -d main.py \
  --cmd_line "CODE_DIR=\`pwd\` && cd \$SSM_INSTANCE_STATE && \
python \$CODE_DIR/main.py --epochs 40 \$SM_CHANNEL_DATA --dist-url env:// --world-size \$SSM_NUM_NODES --rank \$SSM_HOST_RANK --seed 123" &

# Train on 3 nodes
ssm shell -p cat-vs-dogs -t 3-nodes -o ./output/output_3nodes --download_state \
  -i ./data --it ml.p3.2xlarge -d main.py --ic 3 \
  --cmd_line "CODE_DIR=\`pwd\` && cd \$SSM_INSTANCE_STATE && \
python \$CODE_DIR/main.py --epochs 40 \$SM_CHANNEL_DATA --dist-url env:// --world-size \$SSM_NUM_NODES --rank \$SSM_HOST_RANK --seed 123" &

wait

echo "FINISHED!"

