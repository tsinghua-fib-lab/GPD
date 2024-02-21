import sys
import logging
import sys

def setup_logger(expIndex):
    # Create a custom logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Create handlers
    c_handler = logging.StreamHandler(sys.stdout)
    f_handler = logging.FileHandler("Logs/{}_logs.log".format(expIndex))

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger,"Logs/{}_logs.log".format(expIndex)