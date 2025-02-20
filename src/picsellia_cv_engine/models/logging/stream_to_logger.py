class StreamToLogger:
    """
    Logger that duplicates output to stdout and a log file.
    """

    def __init__(self, filepath, original_stream, mode="a"):
        self.original_stream = original_stream
        self.log = open(filepath, mode)

    def write(self, message):
        self.original_stream.write(message)
        self.log.write(message)

    def flush(self):
        self.original_stream.flush()
        self.log.flush()
