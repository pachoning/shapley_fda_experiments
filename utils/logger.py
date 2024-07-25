import logging
import sys

# Set the logging
log_file = "shapley_fda.log"
logger = logging.getLogger(__name__)
file_handler = logging.FileHandler(
    log_file,
    mode="a",
    encoding="utf-8"
)
logger.addHandler(file_handler)
formatter = logging.Formatter(
    "{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M:%S",
)
file_handler.setFormatter(formatter)
logger.setLevel("INFO")
log = open(log_file, "a")
sys.stdout = log
sys.stderr = log
