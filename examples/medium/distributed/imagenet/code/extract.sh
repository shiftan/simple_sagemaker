#! /bin/bash

set -e # stop and fail if anything stops

echo "Extracting all..."
cd $1

for filename in train/*.tar; do
    OUTDIR=${filename%.tar}
    tar -xf $filename --xform="s|^|$OUTDIR/|S"
    rm $filename
done
