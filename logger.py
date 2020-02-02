import logging.handlers
import os
import sys

handler = logging.handlers.WatchedFileHandler(
    os.environ.get("LOGFILE", "logs/gan-logs.log"))
formatter = logging.Formatter("%(asctime)s; \t %(levelname)s; %(message)s",
                              "%Y-%m-%d %H:%M:%S")
handler.setFormatter(formatter)
root = logging.getLogger()
root.setLevel(os.environ.get("LOGLEVEL", "DEBUG"))
root.addHandler(logging.StreamHandler(sys.stdout))
root.addHandler(handler)
