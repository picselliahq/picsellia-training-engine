#!/bin/bash

usage() {
  echo "Usage: $0 <training_script>.py"
  exit 1
}

# Depending on the model to train, choose to either use python3.8 or python3.10
get_python_command() {
  for version in 3.12 3.11 3.10 3.9 3.8; do
    if command -v python$version &> /dev/null; then
      echo "python$version"
      return 0
    fi
  done

  echo "Error: No supported Python version (3.8 to 3.12) is available on the system."
  exit 1
}

# Function to monitor log handler process
monitor_log_handler() {
  local log_handler_pid=$1
  local training_script_pid=$2

  while :; do
    if ! kill -0 "$log_handler_pid" 2>/dev/null; then
      kill -9 "$training_script_pid"
      echo -e "\e[31mError:\e[0m The log handler has terminated unexpectedly. The training script has been stopped to prevent sata loss. Please check the log file or the environment variables provided.\e[0m"
      exit 1
    fi
    sleep 1
  done
}

# Validate input arguments
if [ $# -ne 1 ]; then
  echo "Error: Exactly one argument is required."
  usage
fi

if [ "${1: -3}" != ".py" ]; then
  echo "Error: The argument must be a Python file (.py)."
  usage
fi

if [ ! -f "$1" ]; then
  echo "Error: The file $1 does not exist."
  usage
fi

log_file_path="/experiment/training.log"
python_cmd=$(get_python_command)

# 1. Start the log handler script in the background
$python_cmd logs/handler.py --log_file_path "$log_file_path" &
log_handler_pid=$!

# 2. Start the training script in the background and redirect output to log file
$python_cmd "$1" > "$log_file_path" 2>&1 &
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
