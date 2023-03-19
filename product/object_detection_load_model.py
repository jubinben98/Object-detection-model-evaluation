import numpy as np
import tensorflow as tf

from time import time

from object_detection_config import Config
from object_detection_logs import Logger
from object_detection_utils import *

class LOAD_MODEL:
    def __init__(self,config_file):
        self.config = Config.get_config(config_file)
        self.logger = Logger().logger()
        self.model  = tf.saved_model.load(self.config.get_model_path)
        self.img_height = self.config.get_img_height
        self.img_width  = self.config.get_img_width
        self.model_inference_time   = list()
        self.overall_inference_time = list()

    def show_model_ip_op(self):
        self.logger.debug(self.model.signatures['serving_default'].inputs)
        self.logger.debug(self.model.signatures['serving_default'].output_dtypes)
        self.logger.debug(self.model.signatures['serving_default'].output_shapes)

    def model_predict(self, image):
        # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
        input_tensor = tf.convert_to_tensor(image)
        # Run inference
        model_fn = self.model.signatures['serving_default']
        output_dict = model_fn(input_tensor)

        return output_dict

    def evaluate_model(self, test_images, test_labels, CARS_CONF_THRESHOLD=0.5, CARS_IOU_THRESHOLD=0.5, PEDESTRIANS_CONF_THRESHOLD=0.5, PEDESTRIANS_IOU_THRESHOLD=0.5):

        # Model prediction
        start  = time()
        _start = start
        model_predictions = self.model_predict(test_images)
        self.model_inference_time.append(len(test_images)/(time() - start))
        self.logger.debug("Time taken for prediction: %.2f" % self.model_inference_time[-1])
        self.logger.debug("Model output shape: %s" %str(model_predictions["detection_boxes"].shape))

        # Categorizing output of predictions
        self.logger.debug("Categorizing the predictions")
        categorized_predictions = []
        for i in range(model_predictions["detection_boxes"].shape[0]):
            categorized_predictions.append(categorize_predictions(model_predictions, i))

        self.logger.debug("Number of categorized predictions: %d" %len(categorized_predictions))

        # Filtering the categorized bounding boxes based on their respective confidence value
        self.logger.debug("Filtering the predictions based on their confidence threshold")
        conf_filtered_bb = []
        for i in categorized_predictions:
            temp_dict = {"pedestrians": filtering_bb_on_conf(i["pedestrians"]["bbox_conf"], i["pedestrians"]["bbox"],
                                                             conf_threshold=PEDESTRIANS_CONF_THRESHOLD),
                         "cars": filtering_bb_on_conf(i["cars"]["bbox_conf"], i["cars"]["bbox"],
                                                      conf_threshold=CARS_CONF_THRESHOLD)}

            # Normalizing the coordinates based on actual image dimension
            temp_dict["pedestrians"] *= self.img_height
            temp_dict["cars"] *= self.img_height
            conf_filtered_bb.append(temp_dict)
        self.logger.debug("Number of predictions after filtration process: %d" %len(conf_filtered_bb))

        # Filtering out the TP, FP and FN based on their IOU scores with the gt_boxes
        all_precision_cars = []
        all_recall_cars = []
        all_precision_pedestrians = []
        all_recall_pedestrians = []

        all_tp_cars = []
        all_fp_cars = []
        all_fn_cars = []

        all_tp_pedestrians = []
        all_fp_pedestrians = []
        all_fn_pedestrians = []

        self.overall_inference_time.append(len(test_images)/(time()-_start))
        self.logger.info("Calculating the TP, FP and FN of the predictions")
        for b_idx in range(len(test_labels)):
            tp_cars, fp_cars, fn_cars = calculate_tp_fp_fn(test_labels[b_idx]["cars"], conf_filtered_bb[b_idx]["cars"],
                                                           iou_threshold=CARS_IOU_THRESHOLD)
            tp_pedestrians, fp_pedestrians, fn_pedestrians = calculate_tp_fp_fn(test_labels[b_idx]["pedestrians"],
                                                                                conf_filtered_bb[b_idx]["pedestrians"],
                                                                                iou_threshold=PEDESTRIANS_IOU_THRESHOLD)


            precision_cars, recall_cars = cal_recall_precision(len(tp_cars), len(fp_cars), len(fn_cars))
            precision_pedestrians, recall_pedestrians = cal_recall_precision(len(tp_pedestrians), len(fp_pedestrians),
                                                                             len(fn_pedestrians))

            all_precision_cars.append(precision_cars)
            all_recall_cars.append(recall_cars)
            all_precision_pedestrians.append(precision_pedestrians)
            all_recall_pedestrians.append(recall_pedestrians)

            all_tp_cars.append(len(tp_cars))
            all_fp_cars.append(len(fp_cars))
            all_fn_cars.append(len(fn_cars))

            all_tp_pedestrians.append(len(tp_pedestrians))
            all_fp_pedestrians.append(len(fp_pedestrians))
            all_fn_pedestrians.append(len(fn_pedestrians))

        # Car's avg-precision and avg-recall
        self.logger.debug("Car's evaluation: %.2f, %.2f" %(np.mean(all_precision_cars), np.mean(all_recall_cars)))

        # Pedestrians's avg-precision and avg-recall
        self.logger.debug("Pedestrians's evaluation: %.2f, %.2f" %(np.mean(all_precision_pedestrians),
                                                                   np.mean(all_recall_pedestrians)))

        return {
            "cars_avg_precision": np.mean(all_precision_cars),
            "cars_avg_recall": np.mean(all_recall_cars),
            "pedestrians_avg_precision": np.mean(all_precision_pedestrians),
            "pedestrians_avg_recall": np.mean(all_recall_pedestrians),
            "all_tp_cars": all_tp_cars,
            "all_fp_cars": all_fp_cars,
            "all_fn_cars": all_fn_cars,
            "all_tp_pedestrians": all_tp_pedestrians,
            "all_fp_pedestrians": all_fp_pedestrians,
            "all_fn_pedestrians": all_fn_pedestrians
        }

    def predict(self, test_images, test_labels, cars_conf_th=[], cars_iou_th=[], pedestrians_conf_th=[], pedestrians_iou_th=[]):

        cars_recall = []
        cars_precision = []
        pedestrians_recall = []
        pedestrians_precision = []
        iteration_configuration = []

        for i in range(len(cars_conf_th)):
            self.logger.info("Running model evaluation for with following configurations")
            self.logger.info("Cars-confidence threshold:  %.3f,  Cars-IOU threshold : %.3f"%(cars_conf_th[i],
                                                                                             cars_iou_th[i]))
            self.logger.info("Pedst-confidence threshold: %.3f,  Pedst-IOU threshold: %.3f"%(pedestrians_conf_th[i],
                                                                                             pedestrians_iou_th[i]))
            result = self.evaluate_model(test_images,
                                         test_labels,
                                         CARS_CONF_THRESHOLD=cars_conf_th[i],
                                         CARS_IOU_THRESHOLD=cars_iou_th[i],
                                         PEDESTRIANS_CONF_THRESHOLD=pedestrians_conf_th[i],
                                         PEDESTRIANS_IOU_THRESHOLD=pedestrians_iou_th[i])

            cars_precision.append(result["cars_avg_precision"])
            cars_recall.append(result["cars_avg_recall"])
            pedestrians_precision.append(result["pedestrians_avg_precision"])
            pedestrians_recall.append(result["pedestrians_avg_recall"])

            iteration_configuration.append({
                "car_conf_th": cars_conf_th[i],
                "car_iou_th": cars_iou_th[i],
                "pedestrians_conf_th": pedestrians_conf_th[i],
                "pedestrians_iou_th": pedestrians_iou_th[i],
                "all_tp_cars": result["all_tp_cars"],
                "all_fp_cars": result["all_fp_cars"],
                "all_fn_cars": result["all_fn_cars"],
                "all_tp_pedestrians": result["all_tp_pedestrians"],
                "all_fp_pedestrians": result["all_fp_pedestrians"],
                "all_fn_pedestrians": result["all_fn_pedestrians"]
            })

        self.logger.debug("All configuration predictions done")
        return cars_recall, cars_precision, pedestrians_recall, pedestrians_precision, iteration_configuration

