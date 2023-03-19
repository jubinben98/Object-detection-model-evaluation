import sys
import logging
from object_detection_config import Config

GLOBAL_LOGGER = None


class CustomFilter(logging.Filter):
    '''
    Custom filter set for name and line number
    '''
    def filter(self, record):
        '''
        custom filter
        '''
        record.filename_lineno = "%s:%d" % (record.filename, record.lineno)
        return True


class Logger:
    """
    Logging class for defining logger and using it in the program
    """
    def __init__(self):
        self.config = Config.get_config()
        self.log_level = self.config.log_level
        self.name = "PalletDetection"

    def logger(self):
        '''
        Logger function to define the logger
        '''

        global GLOBAL_LOGGER

        if not GLOBAL_LOGGER:

            if self.log_level == 'debug':
                level = logging.DEBUG
            elif self.log_level == 'info':
                level = logging.INFO
            elif self.log_level == 'warning':
                level = logging.WARNING
            elif self.log_level == 'error':
                level = logging.ERROR
            elif self.log_level == 'critical':
                level = logging.CRITICAL
            else:
                raise ValueError("Please provide a correct logging level in the config file")

            log_object = logging.getLogger(self.name)
            log_object.setLevel(level)
            log_object.addFilter(CustomFilter())
            streamhandler = logging.StreamHandler(stream=sys.stdout)
            formatter = logging.Formatter(
                '%(asctime)s %(levelname)-7s %(filename_lineno)-27s : %(message)s')
            streamhandler.setFormatter(formatter)

            log_object.addHandler(streamhandler)

            GLOBAL_LOGGER = log_object

        return GLOBAL_LOGGER