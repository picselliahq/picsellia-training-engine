#!/bin/bash

usage() {
  echo "Usage: $0 filename"
  exit 1
}

get_python() {
  if command -v python3.10 &> /dev/null; then
    echo "python3.10"
    return 0
  elif command -v python3.8 &> /dev/null; then
    echo "python3.8"
    return 0
  else
    echo "Neither Python 3.10 nor Python 3.8 is available on the system."
    exit 1
  fi
}

if [ $# -eq 0 ]; then
  echo "Error: a filename must be provided."
  usage
fi

if [ ! -e "$1" ]; then
  echo "Error: The provided filename $1 does not exist inside the current directory."
  usage
fi

if [ "${1: -3}" != ".py" ]; then
  echo "Error: $1 must be a Python file."
  usage
fi

log_file_path="/experiment/training.log"
python_cmd=$(get_python)

$python_cmd logs/handler.py --log_file_path "$log_file_path" &
LOG_HANDLER_PID=$!

$python_cmd "$1" > "$log_file_path" 2>&1
RETURN_CODE=$?

echo "--ec-- $RETURN_CODE" >> "$log_file_path"

wait "$LOG_HANDLER_PID"

exit $RETURN_CODE