#!/bin/bash

usage() {
  echo "Usage: $0 <python_version> <training_script>.py"
  exit 1
}

# Function to monitor log handler process
monitor_log_handler() {
  local log_handler_pid=$1
  local training_script_pid=$2

  while :; do
    if ! kill -0 "$log_handler_pid" 2>/dev/null; then
      kill -9 "$training_script_pid"
      echo -e "\e[31mError:\e[0m The log handler has terminated unexpectedly. The training script has been stopped to prevent data loss. Please check the log file or the environment variables provided.\e[0m"
      exit 1
    fi
    sleep 1
  done
}

# Validate input arguments
if [ $# -ne 2 ]; then
  echo "Error: Exactly two arguments are required."
  usage
fi

python_version=$1
script_file=$2

if [[ "$python_version" != python3.* ]]; then
  echo "Error: The first argument must be a valid Python version (e.g., python3.8)."
  usage
fi

if [ "${script_file: -3}" != ".py" ]; then
  echo "Error: The second argument must be a Python file (.py)."
  usage
fi

if [ ! -f "$script_file" ]; then
  echo "Error: The file $script_file does not exist."
  usage
fi

log_file_path="/experiment/training.log"
python_cmd=$python_version

# 1. Start the log handler script in the background
$python_cmd logs/handler.py --log_file_path "$log_file_path" &
log_handler_pid=$!

# 2. Start the training script in the background and redirect output to log file
$python_cmd "$script_file" > "$log_file_path" 2>&1 &
training_script_pid=$!

# 3. Start the monitor in the background
monitor_log_handler "$log_handler_pid" "$training_script_pid" &
monitor_pid=$!

# Wait for the training script to finish
wait "$training_script_pid" 2>/dev/null
training_exit_code=$?

echo "--ec-- $training_exit_code"
echo "--ec-- $training_exit_code" >> "$log_file_path"
wait "$log_handler_pid"
# Training script has finished, now we can kill the log handler monitor
kill "$monitor_pid" 2>/dev/null

# Exit with the training script's same exit code
exit "$training_exit_code"
