#! /bin/bash

BASEDIR=$(dirname "$0")
cd $BASEDIR

rm -rf output
unzip $1 -d ./output
cd output
mv popen*/*0/* .
rm -r popen*
find . | grep "\.extracted" | xargs rm

for file in *; do
    echo updating $file ...
    rm -rf ../../$file/expected_output/*
    cp -r $file/output/* ../../$file/expected_output
done