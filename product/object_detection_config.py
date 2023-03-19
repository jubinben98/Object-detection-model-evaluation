import ast
import configparser
import os

import numpy as np


class Config:
    """
    Config classes for getting the config values
    """

    GLOBAL_CONFIG = None

    @classmethod
    def get_config(self, config_file="etc/config.ini"):
        """
        Class method to initialise the config object
        """
        if Config.GLOBAL_CONFIG is None:
            Config.GLOBAL_CONFIG = __Config__(config_file)
        return Config.GLOBAL_CONFIG

    @classmethod
    def __clear_config__(self):
        """
        Class method to clear any config objects that
        might be in memory
        """
        if Config.GLOBAL_CONFIG is not None:
            del Config.GLOBAL_CONFIG
            Config.GLOBAL_CONFIG = None


class __Config__(object):

    """
    Class for singleton config
    """

    def __init__(self, config_file):
        """
        constructor
        """
        self.path = os.path.normpath(config_file)
        self._config = configparser.ConfigParser()
        self._config.read(self.path)
        self.current_dir = os.getcwd()

    '''
    SYSTEM FUNCTIONS
    '''
    @property
    def log_level(self):
        """
        This property function defines the log level
        or log mode for the system logger.

        :return: str
        """
        return str(self._config['SYSTEM']['log_level']).lower()

    '''
    OBJECT DETECTION MODEL
    '''

    @property
    def get_model_path(self):
        """
        Defines the tensorflow model path for object detection

        :return: model_path -> str
        """
        try:
            para = self._config["OBJECT_DETECTION_MODEL"]["model_path"]
            if os.path.isdir(para):
                return para
            else:
                raise ValueError("Add a model directory for the object detection model.")
        except Exception as e:
            raise e

    @property
    def get_num_image_process(self):
        """
        Number of images to process
        :options: if "all" -> process all images present
                  if int() -> process only the n number of images for testing
        :return: int
        """
        try:
            para = self._config["OBJECT_DETECTION_MODEL"]["process_images"]
            if para.lower()=="all":
                return para

            para = int(para)
            return para

        except Exception as e:
            raise e

    @property
    def get_cars_conf_th(self):
        """
        Defines the cars_conf_th for the prediction post-processing
        :return: [float, float, ...]
        """
        para = self._config["OBJECT_DETECTION_MODEL"]["cars_conf_th"]
        para = para.split(", ")
        para = list(map(float, para))
        return para

    @property
    def get_cars_iou_th(self):
        """
        Defines the cars_iou_th for the prediction post-processing
        :return: [float, float, ...]
        """
        para = self._config["OBJECT_DETECTION_MODEL"]["cars_iou_th"]
        para = para.split(", ")
        para = list(map(float, para))
        return para

    @property
    def get_pedestrians_conf_th(self):
        """
        Defines the pedestrians_conf_th for the prediction post-processing
        :return: [float, float, ...]
        """
        para = self._config["OBJECT_DETECTION_MODEL"]["pedestrians_conf_th"]
        para = para.split(", ")
        para = list(map(float, para))
        return para

    @property
    def get_pedestrians_iou_th(self):
        """
        Defines the pedestrians_iou_th for the prediction post-processing
        :return: [float, float, ...]
        """
        para = self._config["OBJECT_DETECTION_MODEL"]["pedestrians_iou_th"]
        para = para.split(", ")
        para = list(map(float, para))
        return para



    '''
    DATA CONFIGURATIONS
    '''

    @property
    def get_data_img_path(self):
        """
        Defines the path where all the test images are saved

        :return: model_path -> str
        """
        try:
            para = self._config["DATA"]["test_images"]
            if os.path.isdir(para):
                return para
            else:
                raise ValueError("Add a valid test-image directory for the object detection model.")
        except Exception as e:
            raise e

    @property
    def get_data_label_path(self):
        """
        Defines the path where all the labels are saved

        :return: model_path -> str
        """
        try:
            para = self._config["DATA"]["test_labels"]
            if os.path.isdir(para):
                return para
            else:
                raise ValueError("Add a valid test-labels directory for the object detection model.")
        except Exception as e:
            raise e

    @property
    def get_img_width(self):
        """
        Defines the Image width from the test dataset
        :return: int
        """
        para = int(self._config["DATA"]["img_width"])
        return para

    @property
    def get_img_height(self):
        """
        Defines the Image height from the test dataset
        :return: int
        """
        para = int(self._config["DATA"]["img_height"])
        return para