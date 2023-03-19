import numpy as np

from collections import defaultdict

def filtering_bb_on_conf(bb_confs, bb, conf_threshold=0.5):
    selected_idx = np.where(bb_confs > conf_threshold)
    selected_bb = bb[selected_idx]
    selected_bb = np.array(list(map(rearrange_predicted_bb_coordinates, selected_bb)))
    return selected_bb

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

def calculate_tp_fp_fn(gt_boxes, pr_boxes, iou_threshold=0.5):
    tp_predictions = list()
    fp_predictions = list()
    fn_predictions = list()
    gt_selected = np.full((len(gt_boxes)), False)

    # Comparing IOU of all the pr boxes with the gt-boxes
    for i in pr_boxes:
        tp_check = False
        for j in gt_boxes:
            if bb_intersection_over_union(i, j) > iou_threshold:
                gt_selected[gt_boxes.index(j)] = True
                tp_check = True
                tp_predictions.append(i)
                break
        if not tp_check:
            fp_predictions.append(i)

    fn_predictions = np.array(gt_boxes)[np.where(gt_selected == False)[0]]
    return tp_predictions, fp_predictions, fn_predictions

def cal_recall_precision(TP, FP, FN):
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    return precision, recall


def categorize_predictions(model_predictions, batch_idx):
    # Filtering out predctions with pedestrians
    pedestrians_idx = np.where(model_predictions["detection_classes"][batch_idx] == 1)
    pedestrains_bb = np.array(model_predictions["detection_boxes"][batch_idx])[pedestrians_idx]
    pedestrains_bb_conf = np.array(model_predictions["detection_scores"][batch_idx])[pedestrians_idx]

    # Filtering out predctions with cars
    cars_idx = np.where(model_predictions["detection_classes"][batch_idx] == 3)
    cars_bb = np.array(model_predictions["detection_boxes"][batch_idx])[cars_idx]
    cars_bb_conf = np.array(model_predictions["detection_scores"][batch_idx])[cars_idx]

    return {
        "pedestrians": {
            "bbox": pedestrains_bb,
            "bbox_conf": pedestrains_bb_conf
        },
        "cars": {
            "bbox": cars_bb,
            "bbox_conf": cars_bb_conf
        }
    }

def get_all_bb(label):
    annotations = label["annotations"]
    categories = defaultdict(lambda: list())
    for i in annotations:
        upper_left_x = i["bbox"][0]
        upper_left_y = i["bbox"][1]
        lower_right_x = i["bbox"][0] + i["bbox"][2]
        lower_right_y = i["bbox"][1] + i["bbox"][3]

        if i["category_id"] == 3:
            categories["pedestrians"].append([upper_left_x, upper_left_y, lower_right_x, lower_right_y])
        elif i["category_id"] == 2:
            categories["cars"].append([upper_left_x, upper_left_y, lower_right_x, lower_right_y])

    return dict(categories)

def rearrange_predicted_bb_coordinates(bb):
    return [bb[1], bb[0], bb[3], bb[2]]
