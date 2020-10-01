#! /bin/bash
set -e # fail if any test fails

# Params: [output] [prefix] [suffix] [additional ssm params...]
BASEDIR=$(dirname "$0")
echo "Running with", $@

# Example 7 - local mode
#   --ks is used to avoid messing with state (not supported in local mode)
ssm shell -p ${2}simple-sagemaker-example-cli${3} -t shell-cli-local \
    --cmd_line "ps -elf >> \$SM_OUTPUT_DATA_DIR/ps__elf" \
    -o $1/example7 --it 'local' --no_spot --download_output ${@:4} --ks