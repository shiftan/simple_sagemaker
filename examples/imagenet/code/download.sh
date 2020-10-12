#! /bin/bash
# Expected to be launched with DATA_DIR as first argument

set -ex # stop and fail if anything stops

mkdir -p $1
cd $1

[ -d ./train ] && rm -r ./train
[ -d ./val ] && rm -r ./val

apt-get update
apt-get -y --allow-unauthenticated install aria2
download () {
    aria2c --summary-interval=30 --conditional-get=true -x 16 -s 16 $1
}

### From https://cloud.google.com/tpu/docs/imagenet-setup, please make sure you have the permission to download the files from [Imagenet](http://image-net.org)
echo Downloading to `pwd` 
for FILENAME in ILSVRC2012_img_val.tar ILSVRC2012_img_train_t3.tar
do
    download http://image-net.org/challenges/LSVRC/2012/dd31405981ef5f776aa17412e1f0c112/${FILENAME} 2>&1 && echo finished downloading $FILENAME  &
done
wait
echo "Download finished!"

echo "Extracting first level..."
tar -xf ILSVRC2012_img_train_t3.tar --xform="s|^|train/|S" &
wait
mv ILSVRC2012_img_val.tar val/
echo "Done!"
