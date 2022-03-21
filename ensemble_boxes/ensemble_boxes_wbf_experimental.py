# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


import numpy as np


def prefilter_boxes(
        boxes,
        scores,
        labels,
        weights,
        thr,
        skip_checks=False
):

    # Concat everything
    all_boxes = np.concatenate(boxes, axis=0)
    all_scores = np.expand_dims(np.concatenate(scores, axis=0), axis=1)
    all_labels = np.expand_dims(np.concatenate(labels, axis=0), axis=1)
    unique_labels = np.unique(all_labels, return_counts=False)

    wp = []
    tp = []
    for i in range(len(boxes)):
        w1 = np.full(len(boxes[i]), weights[i])
        t1 = np.full(len(boxes[i]), i)
        wp.append(w1)
        tp.append(t1)
    all_weights = np.expand_dims(np.concatenate(wp), axis=1)
    all_index = np.expand_dims(np.concatenate(tp), axis=1)
    all_scaled_scores = all_scores * all_weights

    data = np.concatenate([all_labels, all_scaled_scores, all_weights, all_index, all_boxes], axis=1)

    # Remove all values lower THR
    cond = (data[:, 1] >= thr)
    data = data[cond]

    if not skip_checks:
        # Sort values x and y
        data[:, [4, 6]] = np.sort(data[:, [4, 6]], axis=1)
        data[:, [5, 7]] = np.sort(data[:, [5, 7]], axis=1)

        # Checks range [0, 1]
        part = data[:, 4:]
        part[part < 0.0] = 0.0
        part[part > 1.0] = 1.0
        data[:, 4:] = part

        # Check area
        area = (data[:, 6] - data[:, 4]) * (data[:, 7] - data[:, 5])
        data = data[area > 0.0]

    # Sort array by score (desc)
    sort_condition = data[:, 1].argsort()[::-1]
    data = data[sort_condition]

    # Create dict with boxes stored by its label
    new_boxes = dict()
    for label in unique_labels:
        cond = data[:, 0] == label
        new_boxes[label] = data[cond]

    return new_boxes


def get_weighted_box(
        boxes,
        conf_type='avg'
):
    """
    Create weighted box for set of boxes
    :param boxes: set of boxes to fuse
    :param conf_type: type of confidence one of 'avg' or 'max'
    :return: weighted box (label, score, weight, model index, x1, y1, x2, y2)
    """

    box = np.zeros(8, dtype=np.float32)
    conf = boxes[:, 1].sum()
    w = boxes[:, 2].sum()
    box[4:] = (boxes[:, 1:2] * boxes[:, 4:]).sum(axis=0)
    box[0] = boxes[0, 0]
    if conf_type == 'max':
        box[1] = np.array(boxes[:, 1]).max()
    else:
        box[1] = conf / len(boxes)
    box[2] = w
    box[3] = -1 # model index field is retained for consistency but is not used.
    box[4:] /= conf
    return box


def get_each_vs_each(arr, func):
    x = len(arr)
    arr_tile = np.tile(arr, x)
    arr_repeat = np.repeat(arr, x)
    func_arr = func(arr_tile, arr_repeat)
    res = func_arr.reshape((x, x))
    return res


def get_iou_matrix(boxes):
    xA = get_each_vs_each(boxes[:, 0], np.maximum)
    yA = get_each_vs_each(boxes[:, 1], np.maximum)
    xB = get_each_vs_each(boxes[:, 2], np.minimum)
    yB = get_each_vs_each(boxes[:, 3], np.minimum)
    interArea = np.maximum(xB - xA, 0) * np.maximum(yB - yA, 0)

    # compute sum of areas each vs each
    boxArea = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    sumArea = get_each_vs_each(boxArea, np.add)

    iou_matrix = interArea / (sumArea - interArea)
    return iou_matrix


def weighted_boxes_fusion_experimental(
        boxes_list,
        scores_list,
        labels_list,
        weights=None,
        iou_thr=0.55,
        skip_box_thr=0.0,
        conf_type='avg',
        allows_overflow=False,
        skip_checks=False,
):
    '''
    :param boxes_list: list of boxes predictions from each model, each box is 4 numbers.
    It has 3 dimensions (models_number, model_preds, 4)
    Order of boxes: x1, y1, x2, y2. We expect float normalized coordinates [0; 1]
    :param scores_list: list of scores for each model
    :param labels_list: list of labels for each model
    :param weights: list of weights for each model. Default: None, which means weight == 1 for each model
    :param iou_thr: IoU value for boxes to be a match
    :param skip_box_thr: exclude boxes with score lower than this variable
    :param conf_type: how to calculate confidence in weighted boxes.
        'avg': average value,
        'max': maximum value,
        'box_and_model_avg': box and model wise hybrid weighted average,
        'absent_model_aware_avg': weighted average that takes into account the absent model.
    :param allows_overflow: false if we want confidence score not exceed 1.0
    :param skip_checks: if true then checks for varaiables values will be disabled (speed up calculations)

    :return: boxes: boxes coordinates (Order of boxes: x1, y1, x2, y2).
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

    filtered_boxes = prefilter_boxes(
        boxes_list,
        scores_list,
        labels_list,
        weights,
        skip_box_thr,
        skip_checks,
    )
    if len(filtered_boxes) == 0:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,))

    overall_boxes = []
    for label in filtered_boxes:
        boxes = filtered_boxes[label]
        new_boxes = []
        weighted_boxes = []

        # Find all IoUs
        iou_matrix = get_iou_matrix(boxes[:, 4:])

        # Clusterize boxes first
        used_locations = set()
        for j in range(0, len(boxes)):
            if j in used_locations:
                continue
            locations = np.where(iou_matrix[j] > iou_thr)[0]
            set_loc = set(locations)
            locations = list(set_loc - used_locations)
            bs = boxes[locations]
            if conf_type == 'avg':
                new_boxes.append(len(bs))
            elif conf_type in ['box_and_model_avg', 'absent_model_aware_avg']:
                new_boxes.append(bs)
            wb = get_weighted_box(bs, conf_type)
            weighted_boxes.append(wb)
            used_locations |= set_loc

        weighted_boxes = np.stack(weighted_boxes, axis=0)

        # Rescale confidence based on number of models and boxes
        for i in range(len(weighted_boxes)):
            clustered_boxes = new_boxes[i]
            if conf_type == 'box_and_model_avg':
                clustered_boxes = np.array(clustered_boxes)
                # weighted average for boxes
                weighted_boxes[i, 1] = weighted_boxes[i, 1] * len(clustered_boxes) / weighted_boxes[i, 2]
                # identify unique model index by model index column
                _, idx = np.unique(clustered_boxes[:, 3], return_index=True)
                # rescale by unique model weights
                weighted_boxes[i, 1] = weighted_boxes[i, 1] * clustered_boxes[idx, 2].sum() / weights.sum()
            elif conf_type == 'absent_model_aware_avg':
                clustered_boxes = np.array(clustered_boxes)
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
                weighted_boxes[i, 1] = weighted_boxes[i, 1] * min(len(weights), clustered_boxes) / weights.sum()
            else:
                weighted_boxes[i, 1] = weighted_boxes[i, 1] * clustered_boxes / weights.sum()
        overall_boxes.append(weighted_boxes)
    overall_boxes = np.concatenate(overall_boxes, axis=0)
    overall_boxes = overall_boxes[overall_boxes[:, 1].argsort()[::-1]]
    boxes = overall_boxes[:, 4:]
    scores = overall_boxes[:, 1]
    labels = overall_boxes[:, 0]
    return boxes, scores, labels
