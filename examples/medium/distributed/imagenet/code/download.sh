#! /bin/bash
# Expected to be launched with DATA_DIR as first argument

set -e # stop and fail if anything stops

mkdir -p $1
cd $1

[ -d ./train ] && rm -r ./train
[ -d ./val ] && rm -r ./val

apt-get update
apt-get -y --allow-unauthenticated install aria2
download () {
    aria2c --summary-interval=30 --conditional-get=true -x 16 -s 16 $1
}

### Adapted from https://github.com/pytorch/examples/blob/master/run_python_examples.sh
mkdir -p val/n
mkdir -p train/n
wget "https://upload.wikimedia.org/wikipedia/commons/5/5a/Socks-clinton.jpg" || { error "couldn't download sample image for imagenet"; return; }
mv Socks-clinton.jpg train/n
cp train/n/* val/n/

echo Downloading to `pwd` 
for FILENAME in ILSVRC2012_img_val.tar ILSVRC2012_img_train_t3.tar
do
    download http://image-net.org/challenges/LSVRC/2012/dd31405981ef5f776aa17412e1f0c112/${FILENAME} 2>&1 && echo finished downloading $FILENAME  &
done
wait
echo "Download finished!"

echo "Extracting first level..."
tar -xf ILSVRC2012_img_train_t3.tar --xform="s|^|train/|S" &
tar -xf ILSVRC2012_img_val.tar --xform="s|^|val/|S" &
wait
echo "Done!"
