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
