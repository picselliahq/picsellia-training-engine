#!/bin/bash

usage() {
  echo "Usage: $0 filename"
  exit 1
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

log_file_path="/picsellia/training.log"

python3.10 logs_handler.py --log_file_path "$log_file_path" &
LOG_HANDLER_PID=$!

python3.10 "$1" > "$log_file_path" 2>&1
RETURN_CODE=$?

echo "--ec-- $RETURN_CODE" >> "$log_file_path"

wait "$LOG_HANDLER_PID"

exit $RETURN_CODE