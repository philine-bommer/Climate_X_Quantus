import sys
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)

def handle_unhandled_exception(exc_type, exc_value, exc_traceback):
    '''
    Log unhandled exceptions.
    '''
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logger.critical("Unhandled exception", exc_info=(exc_type, exc_value, exc_traceback))

def get_logger(dirLog):
    log_file = dirLog + os.sep + "logging_{date_time}.log".format(date_time=datetime.now())
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=log_file, filemode='w', level=logging.DEBUG,
                        format='%(asctime)s %(message)s')

    sys.excepthook = handle_unhandled_exception
    return logger
