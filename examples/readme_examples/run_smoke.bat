:: Params: [output] [prefix] [suffix] [additional ssm params...]
echo %0
echo off
for %%F in (%0) do set dirname=%%~dpF
echo "Running with"
echo %*
echo %dirname%

:: Example 7 - local mode
::   --ks is used to avoid messing with state (not supported in local mode)
ssm shell -p %2simple-sagemaker-example-cli%3 -t shell-cli-local ^
    --cmd_line "ps -elf >> \$SM_OUTPUT_DATA_DIR/ps__elf" ^
    -o %1/example7 --it 'local' --no_spot --download_output %4 %5 %6 %7 %8 %9 --ks