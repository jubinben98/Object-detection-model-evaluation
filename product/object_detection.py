import os
import numpy as np

from object_detection_config import Config
from object_detection_logs import Logger
from object_detection_load_model import LOAD_MODEL
from object_detection_data import DATA
from object_detection_utils import cal_recall_precision

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

class ObjectDetect:
    """
    6D pallet pose detection main class
    """
    def __init__(self, config_path='etc/config.ini'):
        """
        Constructor
        :param config_path: Path of the config file.
        """
        self.config = Config.get_config(config_path)
        self.logger = Logger().logger()

        self.logger.info("Loading the model")
        self.model  = LOAD_MODEL(config_path)
        self.model.show_model_ip_op()

        self.data   = DATA(config_path)

        self.cars_conf_th = self.config.get_cars_conf_th
        self.cars_iou_th = self.config.get_cars_iou_th
        self.pedestrians_conf_th = self.config.get_pedestrians_conf_th
        self.pedestrians_iou_th = self.config.get_pedestrians_iou_th

    def predict(self):
        self.logger.info("Loading the dataset")
        test_images, test_labels = self.data.load_data()

        self.logger.info("Model prediction started")
        cars_recall, cars_precision, pedestrians_recall, pedestrians_precision, iteration_configuration = self.model.predict(test_images,
                                                                                                                             test_labels,
                                                                                                                             cars_conf_th=self.cars_conf_th,
                                                                                                                             cars_iou_th=self.cars_iou_th,
                                                                                                                             pedestrians_conf_th=self.pedestrians_conf_th,
                                                                                                                             pedestrians_iou_th=self.pedestrians_iou_th)

        self.logger.debug("Result lengths: %d, %d, %d, %d, %d" %(len(cars_recall),
                                                                 len(cars_precision),
                                                                 len(pedestrians_recall),
                                                                 len(pedestrians_precision),
                                                                 len(iteration_configuration)))

        # Collecting all the TPs, FPs and FNs for each class. Then calculating it's respective recall and precision for each configuration

        precision_cars = []
        recall_cars    = []

        precision_pedestrians = []
        recall_pedestrians    = []

        iter_tp_c = []
        iter_fp_c = []
        iter_fn_c = []

        iter_tp_p = []
        iter_fp_p = []
        iter_fn_p = []

        for itr_ in range(len(iteration_configuration)):
            tp_cars = []
            fp_cars = []
            fn_cars = []

            tp_pedestrians = []
            fp_pedestrians = []
            fn_pedestrians = []

            for i in range(len(iteration_configuration[itr_]["all_tp_cars"])):
                tp_cars.append(iteration_configuration[itr_]["all_tp_cars"][i])
                fp_cars.append(iteration_configuration[itr_]["all_fp_cars"][i])
                fn_cars.append(iteration_configuration[itr_]["all_fn_cars"][i])
                tp_pedestrians.append(iteration_configuration[itr_]["all_tp_pedestrians"][i])
                fp_pedestrians.append(iteration_configuration[itr_]["all_fp_pedestrians"][i])
                fn_pedestrians.append(iteration_configuration[itr_]["all_fn_pedestrians"][i])

            prec_c, recall_c = cal_recall_precision(sum(tp_cars), sum(fp_cars), sum(fn_cars))
            prec_p, recall_p = cal_recall_precision(sum(tp_pedestrians), sum(fp_pedestrians), sum(fn_pedestrians))

            precision_cars.append(prec_c)
            recall_cars.append(recall_c)
            precision_pedestrians.append(prec_p)
            recall_pedestrians.append(recall_p)

            iter_tp_c.append(sum(tp_cars))
            iter_fp_c.append(sum(fp_cars))
            iter_fn_c.append(sum(fn_cars))

            iter_tp_p.append(sum(tp_pedestrians))
            iter_fp_p.append(sum(fp_pedestrians))
            iter_fn_p.append(sum(fn_pedestrians))


        AP_CARS = np.sum((np.array(recall_cars)[:-1] - np.array(recall_cars)[1:]) * np.array(precision_cars)[:-1])
        AP_PEDESTRIANS = np.sum((np.array(recall_pedestrians)[:-1] - np.array(recall_pedestrians)[1:]) * np.array(precision_pedestrians)[:-1])

        self.logger.info("Average precision cars: %.3f" %AP_CARS)
        self.logger.info("Average precision pedestrians: %.3f" %AP_PEDESTRIANS)
        self.logger.info("Mean-average-precision: %.3f" %np.mean([AP_CARS, AP_PEDESTRIANS]))

        return {
            "cars-recall": recall_cars,
            "cars-precision": precision_cars,
            "cars-avg-precision": AP_CARS,
            "pedestrians-avg-precision": AP_PEDESTRIANS,
            "map": np.mean([AP_CARS, AP_PEDESTRIANS]),
            "pedestrians-recall": recall_pedestrians,
            "pedestrians-precision": precision_pedestrians,
            "cars-conf-threshold": self.cars_conf_th,
            "cars-iou-threshold": self.cars_iou_th,
            "pedestrians-conf-threshold": self.pedestrians_conf_th,
            "pedestrians-iou-threshold": self.pedestrians_iou_th,
            "model_inference_time": np.mean(self.model.model_inference_time),
            "over_all_inference_time": np.mean(self.model.overall_inference_time),
            "tp_cars": iter_tp_c,
            "fp_cars": iter_fp_c,
            "fn_cars": iter_fn_c,
            "tp_pedestrians": iter_tp_p,
            "fp_pedestrians": iter_fp_p,
            "fn_pedestrians": iter_fn_p
        }

if __name__ == "__main__":
    ObjectDetect().predict()
