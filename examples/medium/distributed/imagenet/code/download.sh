#! /bin/bash

set -e # stop and fail if anything stops

DATA_DIR=$1
mkdir -p $1
cd $1

[ -d ./train ] && rm -r ./train

### Adapted from https://github.com/pytorch/examples/blob/master/run_python_examples.sh
mkdir -p val/n
mkdir -p train/n
if [ -f "train/n/Socks-clinton.jpg" ]; then
    echo "Alrady exists."
else 
    wget "https://upload.wikimedia.org/wikipedia/commons/5/5a/Socks-clinton.jpg" || { error "couldn't download sample image for imagenet"; return; }
    mv Socks-clinton.jpg train/n
    cp train/n/* val/n/
fi

### Imagenet 16
#wget http://www.image-net.org/image/downsample/Imagenet16_train.zip
#wget http://www.image-net.org/image/downsample/Imagenet16_val.zip

### From https://cloud.google.com/tpu/docs/imagenet-setup:
# nohup wget http://image-net.org/challenges/LSVRC/2012/dd31405981ef5f776aa17412e1f0c112/ILSVRC2012_img_train.tar
# wget http://www.image-net.org/challenges/LSVRC/2012/dd31405981ef5f776aa17412e1f0c112/ILSVRC2012_img_val.tar
echo Downloading training set to `pwd`
wget -N http://www.image-net.org/challenges/LSVRC/2012/dd31405981ef5f776aa17412e1f0c112/ILSVRC2012_img_train_t3.tar
echo "Extracting first level..."
tar -xf ILSVRC2012_img_train_t3.tar --xform="s|^|train/|S"
exit
