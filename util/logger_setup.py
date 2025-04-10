import logging
import os
import sys

import config

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def logger_setup():
    if logger.handlers:  # Prevent adding handlers multiple times if this script is re-imported
        return
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(config.FOLDER, "log.txt"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)