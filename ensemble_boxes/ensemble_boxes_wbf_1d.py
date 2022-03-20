# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


import warnings
import numpy as np


def prefilter_line_segments(boxes, scores, labels, weights, thr):
    # Create dict with boxes stored by its label
    new_boxes = dict()

    for t in range(len(boxes)):

        if len(boxes[t]) != len(scores[t]):
            print('Error. Length of boxes arrays not equal to length of scores array: {} != {}'.format(len(boxes[t]), len(scores[t])))
            exit()

        if len(boxes[t]) != len(labels[t]):
            print('Error. Length of boxes arrays not equal to length of labels array: {} != {}'.format(len(boxes[t]), len(labels[t])))
            exit()

        for j in range(len(boxes[t])):
            score = scores[t][j]
            if score < thr:
                continue
            label = int(labels[t][j])
            box_part = boxes[t][j]
            x1 = float(box_part[0])
            x2 = float(box_part[1])

             # Box data checks
            if x2 < x1:
                warnings.warn('X2 < X1 value in box. Swap them.')
                x1, x2 = x2, x1
            if x1 < 0:
                warnings.warn('X1 < 0 in box. Set it to 0.')
                x1 = 0
            if x1 > 1:
                warnings.warn('X1 > 1 in box. Set it to 1. Check that you normalize boxes in [0, 1] range.')
                x1 = 1
            if x2 < 0:
                warnings.warn('X2 < 0 in box. Set it to 0.')
                x2 = 0
            if x2 > 1:
                warnings.warn('X2 > 1 in box. Set it to 1. Check that you normalize boxes in [0, 1] range.')
                x2 = 1
            if (x2 - x1) == 0.0:
                warnings.warn("Zero length line segment skipped: {}.".format(box_part))
                continue

            # [label, score, weight, model index, x1, y1, x2, y2]
            b = [int(label), float(score) * weights[t], weights[t], t, x1, x2]
            if label not in new_boxes:
                new_boxes[label] = []
            new_boxes[label].append(b)

    # Sort each list in dict by score and transform it to numpy array
    for k in new_boxes:
        current_boxes = np.array(new_boxes[k])
        new_boxes[k] = current_boxes[current_boxes[:, 1].argsort()[::-1]]

    return new_boxes


def get_weighted_box(boxes, conf_type='avg'):
    """
    Create weighted box for set of boxes
    :param boxes: set of boxes to fuse
    :param conf_type: type of confidence one of 'avg' or 'max'
    :return: weighted box (label, score, weight, index, x1, y1, x2, y2)
    """

    box = np.zeros(6, dtype=np.float32)
    conf = 0
    conf_list = []
    w = 0
    for b in boxes:
        box[4:] += (b[1] * b[4:])
        conf += b[1]
        conf_list.append(b[1])
        w += b[2]
    box[0] = boxes[0][0]
    if conf_type == 'avg':
        box[1] = conf / len(boxes)
    elif conf_type == 'max':
        box[1] = np.array(conf_list).max()
    elif conf_type in ['box_and_model_avg', 'absent_model_aware_avg']:
        box[1] = conf / len(boxes)
    box[2] = w
    box[3] = -1 # model index field is retained for consistensy but is not used.
    box[4:] /= conf
    return box


def find_matching_line_segment_quickly(
        boxes_list,
        new_box,
        match_iou
):
    """
        Reimplementation of find_matching_box with numpy instead of loops. Gives significant speed up for larger arrays
        (~100x). This was previously the bottleneck since the function is called for every entry in the array.
    """
    def bb_iou_array(boxes, new_box):
        # bb interesection over union
        xA = np.maximum(boxes[:, 0], new_box[0])
        xB = np.minimum(boxes[:, 1], new_box[1])

        interSeg = np.maximum(xB - xA, 0)

        # compute the area of both the prediction and ground-truth rectangles
        lsAArea = (boxes[:, 1] - boxes[:, 0])
        lsBArea = (new_box[1] - new_box[0])

        iou = interSeg / (lsAArea + lsBArea - interSeg)

        return iou

    if boxes_list.shape[0] == 0:
        return -1, match_iou

    boxes = boxes_list

    ious = bb_iou_array(boxes[:, 4:], new_box[4:])

    ious[boxes[:, 0] != new_box[0]] = -1

    best_idx = np.argmax(ious)
    best_iou = ious[best_idx]

    if best_iou <= match_iou:
        best_iou = match_iou
        best_idx = -1

    return best_idx, best_iou


def weighted_boxes_fusion_1d(
        boxes_list,
        scores_list,
        labels_list,
        weights=None,
        iou_thr=0.55,
        skip_box_thr=0.0,
        conf_type='avg',
        allows_overflow=False
):
    '''
    :param boxes_list: list of line segments predictions from each model, each box is 2 numbers.
    It has 3 dimensions (models_number, model_preds, 2)
    Order of line segments: x1, x2. We expect float normalized coordinates [0; 1]
    :param scores_list: list of scores for each model
    :param labels_list: list of labels for each model
    :param weights: list of weights for each model. Default: None, which means weight == 1 for each model
    :param iou_thr: IoU value for line segments to be a match
    :param skip_box_thr: exclude line segments with score lower than this variable
    :param conf_type: how to calculate confidence in weighted line segments. 'avg': average value, 'max': maximum value, 'box_and_model_avg': box and model wise hybrid weighted average, 'absent_model_aware_avg': weighted average that takes into account the absent model.
    :param allows_overflow: false if we want confidence score not exceed 1.0

    :return: boxes: line segments coordinates (Order of boxes: x1, x2).
    :return: scores: confidence scores
    :return: labels: boxes labels
    '''

    if weights is None:
        weights = np.ones(len(boxes_list))
    if len(weights) != len(boxes_list):
        print('Warning: incorrect number of weights {}. Must be: {}. Set weights equal to 1.'.format(len(weights), len(boxes_list)))
        weights = np.ones(len(boxes_list))
    weights = np.array(weights)

    if conf_type not in ['avg', 'max', 'box_and_model_avg', 'absent_model_aware_avg']:
        print('Unknown conf_type: {}. Must be "avg", "max" or "box_and_model_avg", or "absent_model_aware_avg"'.format(conf_type))
        exit()

    filtered_boxes = prefilter_line_segments(
        boxes_list,
        scores_list,
        labels_list,
        weights,
        skip_box_thr
    )
    if len(filtered_boxes) == 0:
        return np.zeros((0, 2)), np.zeros((0,)), np.zeros((0,))

    overall_boxes = []
    for label in filtered_boxes:
        boxes = filtered_boxes[label]
        new_boxes = []
        weighted_boxes = np.empty((0, 6))

        # Clusterize boxes
        for j in range(0, len(boxes)):
            index, best_iou = find_matching_line_segment_quickly(weighted_boxes, boxes[j], iou_thr)

            if index != -1:
                new_boxes[index].append(boxes[j])
                weighted_boxes[index] = get_weighted_box(new_boxes[index], conf_type)
            else:
                new_boxes.append([boxes[j].copy()])
                weighted_boxes = np.vstack((weighted_boxes, boxes[j].copy()))

        # Rescale confidence based on number of models and boxes
        for i in range(len(new_boxes)):
            clustered_boxes = np.array(new_boxes[i])
            if conf_type == 'box_and_model_avg':
                # weighted average for boxes
                weighted_boxes[i, 1] = weighted_boxes[i, 1] * len(clustered_boxes) / weighted_boxes[i, 2]
                # identify unique model index by model index column
                _, idx = np.unique(clustered_boxes[:, 3], return_index=True)
                # rescale by unique model weights
                weighted_boxes[i, 1] = weighted_boxes[i, 1] *  clustered_boxes[idx, 2].sum() / weights.sum()
            elif conf_type == 'absent_model_aware_avg':
                # get unique model index in the cluster
                models = np.unique(clustered_boxes[:, 3]).astype(int)
                # create a mask to get unused model weights
                mask = np.ones(len(weights), dtype=bool)
                mask[models] = False
                # absent model aware weighted average
                weighted_boxes[i, 1] = weighted_boxes[i, 1] * len(clustered_boxes) / (weighted_boxes[i, 2] + weights[mask].sum())
            elif conf_type == 'max':
                weighted_boxes[i, 1] = weighted_boxes[i, 1] / weights.max()
            elif not allows_overflow:
                weighted_boxes[i, 1] = weighted_boxes[i, 1] * min(len(weights), len(clustered_boxes)) / weights.sum()
            else:
                weighted_boxes[i, 1] = weighted_boxes[i, 1] * len(clustered_boxes) / weights.sum()
        overall_boxes.append(weighted_boxes)
    overall_boxes = np.concatenate(overall_boxes, axis=0)
    overall_boxes = overall_boxes[overall_boxes[:, 1].argsort()[::-1]]
    boxes = overall_boxes[:, 4:]
    scores = overall_boxes[:, 1]
    labels = overall_boxes[:, 0]
    return boxes, scores, labels
