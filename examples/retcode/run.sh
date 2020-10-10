#! /bin/bash
set -ex # fail if any test fails

cd `dirname "$0"`

# Args: expected actual msg
assert_eq() {
  local expected="$1"
  local actual="$2"
  local msg

  if [ "$#" -ge 3 ]; then
    msg="$3"
  fi

  if [ "$expected" == "$actual" ]; then
    return 0
  else
    [ "${#msg}" -gt 0 ] && echo "$expected == $actual :: $msg" || true
    return 1
  fi
}

pids=()
expected=()

# Args: expected command arg1 arg2 ...
run_and_append() {
    "${@:2}" &
    pids+=($!)
    expected+=($1)
}


run_and_append 0 ssm process -p exit-tests -t proc-cli-ret-0 --max_run_mins 15 \
    --entrypoint "/bin/bash" -- -c "exit 0"

run_and_append 1 ssm process -p exit-tests -t proc-cli-ret-1 --max_run_mins 15 \
    --entrypoint "/bin/bash" -- -c "exit 1"

run_and_append 1 ssm process -p exit-tests -t proc-cli-ret-0-msg --max_run_mins 15 \
    --entrypoint "/bin/bash" -- -c "echo Message >> /opt/ml/output/message && exit 0" &

run_and_append 1 ssm process -p exit-tests -t proc-cli-ret-1-msg --max_run_mins 15 \
    --entrypoint "/bin/bash" -- -c "echo Message >> /opt/ml/output/message && exit 1"


run_and_append 0 ssm shell -p exit-tests -t shel-cli-ret-0-0 \
    --cmd_line "echo \$SSM_HOST_RANK && exit 0" --ic 2 --force_running

run_and_append 1 ssm shell -p exit-tests -t shel-cli-ret-0-1 \
    --cmd_line "echo \$SSM_HOST_RANK && exit \$SSM_HOST_RANK" --ic 2 --force_running

run_and_append 1 ssm shell -p exit-tests -t shel-cli-ret-0-0-msg \
    --cmd_line "echo \$SSM_HOST_RANK && echo Message >> /opt/ml/output/failure && exit 0" --ic 2 --force_running

echo "PIDs" ${pids[@]}

for i in ${!pids[@]} ;do
    wait ${pids[$i]} && true
    assert_eq $? ${expected[$i]} "Retcode should be ${expected[$i]} for $i"
done

echo "PASSED"