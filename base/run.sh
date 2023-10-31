#!/bin/bash

# Function to print the usage information for the script
usage() {
  echo "Usage: $0 filename"
  exit 1
}

# Function to find the appropriate Python version to use, depending on the chosen model version to train
get_python() {
  if command -v python3.10 &> /dev/null; then
    echo "python3.10"
  elif command -v python3.8 &> /dev/null; then
    echo "python3.8"
  else
    echo "Neither Python 3.10 nor Python 3.8 is available on the system."
    exit 1
  fi
}

# Function to monitor the log_handler process
monitor_log_handler() {
  while :
  do
    # If log_handler has stopped, stop the training script and exit
    if ! kill -0 "$log_handler_pid" 2> /dev/null; then
      echo -e "\e[31mError:\e[0m Log handler has terminated unexpectedly. The training script has been stopped to prevent data loss. Please check the provided environment variables."
      kill "$training_script_pid" 2> /dev/null
      wait "$training_script_pid" 2> /dev/null
      exit 1
    fi
    sleep 1
  done
}

# Validate the input arguments
if [ $# -eq 0 ]; then
  echo "Error: A filename must be provided."
  usage
fi

if [ ! -e "$1" ]; then
  echo "Error: The provided filename $1 does not exist."
  usage
fi

if [ "${1: -3}" != ".py" ]; then
  echo "Error: $1 must be a Python file."
  usage
fi

log_file_path="/experiment/training.log"
python_cmd=$(get_python)

# Start the log_handler
$python_cmd logs/handler.py --log_file_path "$log_file_path" &
log_handler_pid=$!

# Start the log_handler monitoring in the background
monitor_log_handler &
log_handler_monitor_pid=$!

# Start the training script
$python_cmd "$1" > "$log_file_path" 2>&1 &
training_script_pid=$!

wait "$training_script_pid"
training_script_exit_code=$?

# Stop the monitor as the training script has finished and wait for the log_handler
kill "$log_handler_monitor_pid" 2> /dev/null
wait "$log_handler_pid" 2> /dev/null

# Append the exit code of the training script to the log file
echo "--ec-- $training_script_exit_code" >> "$log_file_path"

# Exit with the exit code of the training script
exit $training_script_exit_code
