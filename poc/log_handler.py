import logging


class CustomLogHandler(logging.StreamHandler):
    def emit(self, record):
        if record.levelno == logging.ERROR:
            # ANSI escape sequence for red text
            self.stream.write("\x1b[31m")
        super().emit(record)
        if record.levelno == logging.ERROR:
            # Reset ANSI color to default after the message
            self.stream.write("\x1b[0m")
        self.flush()


class StreamToLogger:
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ""

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass
