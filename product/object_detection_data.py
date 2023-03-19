import numpy as np
import cv2 as cv
import json

from glob import glob
from object_detection_config import Config
from object_detection_logs import Logger
from object_detection_utils import get_all_bb


class DATA:
    def __init__(self,config_file):
        self.config = Config.get_config(config_file)
        self.logger = Logger().logger()
        self.test_img_dir   = self.config.get_data_img_path
        self.test_label_dir = self.config.get_data_label_path
        self.num_images_to_process = self.config.get_num_image_process

    def load_data(self):
        test_images_path = sorted(glob(self.test_img_dir + "/*jp*"))
        test_label_paths = sorted(glob(self.test_label_dir + "/*json"))

        if isinstance(self.num_images_to_process, int):
            test_images_path = test_images_path[:self.num_images_to_process]
            test_label_paths = test_label_paths[:self.num_images_to_process]

        assert (len(test_images_path) == len(test_label_paths))

        self.logger.info("Number of test images found: %d" %len(test_images_path))

        test_labels = []
        test_images = []

        self.logger.debug("Reading the images and their respective JSONs")
        for i in range(len(test_images_path)):
            # Loading the label-JSON file
            with open(test_label_paths[i], "rb") as f:
                test_labels.append(get_all_bb(json.load(f)))
            # Loading the image
            test_images.append(cv.cvtColor(cv.imread(test_images_path[i]), cv.COLOR_BGR2RGB))
        test_images = np.array(test_images)

        return test_images, test_labels

    