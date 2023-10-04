import argparse
import logging
import os

from base.common.utils.log_monitor import LogMonitor

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["PICSELLIA_SDK_CUSTOM_LOGGING"] = "True"
os.environ["PICSELLIA_SDK_DOWNLOAD_BAR_MODE"] = "2"
os.environ["PICSELLIA_SDK_SECTION_HANDLER"] = "1"

logging.getLogger("picsellia").setLevel(logging.INFO)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log_file_path",
        help="Path to the log file where the training logs are written",
    )
    args = parser.parse_args()

    log_monitor = LogMonitor(args.log_file_path)
    log_monitor.start_monitoring()