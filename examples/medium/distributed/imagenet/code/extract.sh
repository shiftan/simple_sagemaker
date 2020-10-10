#! /bin/bash
# Expected to be launched with DATA_DIR as first argument

set -e # stop and fail if anything stops

echo "Extracting all..."
cd $1

extract_and_delete() {
    filename=$1
    OUTDIR=${filename%.tar}
    tar -xf $filename --xform="s|^|$OUTDIR/|S"
    rm $filename    
}

for filename in train/*.tar; do
    extract_and_delete $filename &
done

wait

cd val
# https://github.com/facebookarchive/fb.resnet.torch/blob/master/INSTALL.md
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
