BASEDIR=$(dirname "$0")
pushd .
cd $BASEDIR/example3

ssm run -p ssm-ex -t ex3-1 -s ./code -e ssm_ex3_worker.py \
    -i ./data ShardedByS3Key \
    --iis persons s3://awsglue-datasets/examples/us-legislators/all/persons.json \
    --df "RUN pip3 install pandas==0.25.3 scikit-learn==0.21.3" \
    --repo_name "ex3_repo" --aws_repo_name "ex3_repo" --no_spot \
    --ic 2 --task_type 1 -o ./out3/ex3_1 --cs

ssm run -p ssm-ex -t ex3-2 -s ./code -e ssm_ex3_worker.py \
    -d ./external_dependency --iit ex3_1_model ex3-1 model \
    --iit ex3_1_state ex3-1 state ShardedByS3Key \
    -f tensorflow -m --md "Score" "Score=(.*?);" --tag "MyTag" "MyValue" \
    --ic 2 --task_type 2 -o ./out3/ex3_2 --cs

popd