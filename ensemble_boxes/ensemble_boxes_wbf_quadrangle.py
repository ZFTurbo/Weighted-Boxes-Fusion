# coding: utf-8
__author__ = 'TatiGabru: https://kaggle.com/blondinka'


import warnings
import numpy as np
from .ensemble_boxes_wbf_rotated import polygon_clip, polygon_area


def order_quadrangle(pts):
    """
    Order the 4 vertices of a quadrangle into a canonical CCW order starting from
    the top-left-most vertex. Different models may emit the four corners
    (x1, y1, x2, y2, x3, y3, x4, y4) starting from a different corner or with a
    different winding; canonicalizing lets us clip and (crucially) weighted-average
    corresponding vertices consistently.
    :param pts: array-like of shape (4, 2) or flat length-8 sequence
    :return: (4, 2) numpy array of ordered vertices
    """
    pts = np.asarray(pts, dtype=float).reshape(4, 2)
    cx = pts[:, 0].mean()
    cy = pts[:, 1].mean()
    # Sort vertices counterclockwise by their angle around the centroid.
    angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    pts = pts[np.argsort(angles)]
    # Rotate the ordering so the top-left-most vertex (min y, then min x) comes first.
    start = np.lexsort((pts[:, 0], pts[:, 1]))[0]
    pts = np.roll(pts, -start, axis=0)
    return pts


def bb_intersection_over_union_quadrangle(boxA, boxB):
    """
    IoU of two quadrangles, each given as 8 numbers (x1, y1, x2, y2, x3, y3, x4, y4).
    Vertices are canonically ordered before clipping so the winding is consistent.
    """
    cornersA = order_quadrangle(boxA)
    cornersB = order_quadrangle(boxB)
    inter_pts = polygon_clip(cornersA, cornersB)
    inter_area = polygon_area(inter_pts)
    areaA = polygon_area(cornersA)
    areaB = polygon_area(cornersB)
    union = areaA + areaB - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def prefilter_boxes(boxes, scores, labels, weights, thr):
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

            # Clamp all 8 coordinates into [0, 1]
            coords = []
            for k in range(8):
                c = float(box_part[k])
                if c < 0:
                    warnings.warn('Coordinate < 0 in box. Set it to 0.')
                    c = 0
                if c > 1:
                    warnings.warn('Coordinate > 1 in box. Set it to 1. Check that you normalize boxes in [0, 1] range.')
                    c = 1
                coords.append(c)

            # Canonically order the 4 vertices so vertices can be averaged consistently.
            ordered = order_quadrangle(coords)
            if polygon_area(ordered) == 0.0:
                warnings.warn("Zero area box skipped: {}.".format(box_part))
                continue
            ordered = ordered.reshape(-1).tolist()

            # [label, score, weight, model index, x1, y1, x2, y2, x3, y3, x4, y4]
            b = [int(label), float(score) * weights[t], weights[t], t] + ordered
            if label not in new_boxes:
                new_boxes[label] = []
            new_boxes[label].append(b)

    # Sort each list in dict by score and transform it to numpy array
    for k in new_boxes:
        current_boxes = np.array(new_boxes[k])
        new_boxes[k] = current_boxes[current_boxes[:, 1].argsort()[::-1]]

    return new_boxes


def align_quadrangle_to_reference(pts, ref):
    """
    Cyclically shift the vertices of a CCW-ordered quadrangle so that each vertex
    corresponds to the nearest vertex of the CCW-ordered reference quadrangle.
    The canonical order alone is not enough: for nearly axis-aligned boxes the
    top-most vertex flips between the top-left and top-right corner as the box
    tilts from +eps to -eps degrees, so the same corner would otherwise land at
    different positions in different predictions.
    :param pts: flat length-8 sequence or (4, 2) array, CCW order
    :param ref: flat length-8 sequence or (4, 2) array, CCW order
    :return: flat length-8 numpy array of aligned vertices
    """
    pts = np.asarray(pts, dtype=float).reshape(4, 2)
    ref = np.asarray(ref, dtype=float).reshape(4, 2)
    shift = min(range(4), key=lambda k: ((np.roll(pts, -k, axis=0) - ref) ** 2).sum())
    return np.roll(pts, -shift, axis=0).reshape(-1)


def get_weighted_box(boxes, conf_type='avg'):
    """
    Create weighted box for set of quadrangles
    :param boxes: set of boxes to fuse. boxes[0] (the highest scoring one) is the reference
        whose vertex order all other boxes are aligned to before averaging.
    :param conf_type: type of confidence one of 'avg', 'max', 'box_and_model_avg', 'absent_model_aware_avg'
    :return: weighted box (label, score, weight, model index, x1, y1, x2, y2, x3, y3, x4, y4)
    """

    box = np.zeros(12, dtype=np.float32)
    conf = 0
    conf_list = []
    w = 0
    ref = boxes[0][4:]
    for b in boxes:
        box[4:] += (b[1] * align_quadrangle_to_reference(b[4:], ref))
        conf += b[1]
        conf_list.append(b[1])
        w += b[2]
    box[0] = boxes[0][0]
    if conf_type in ('avg', 'box_and_model_avg', 'absent_model_aware_avg'):
        box[1] = conf / len(boxes)
    elif conf_type == 'max':
        box[1] = np.array(conf_list).max()
    box[2] = w
    box[3] = -1  # model index field is retained for consistency but is not used.
    box[4:] /= conf
    return box


def find_matching_box(boxes_list, new_box, match_iou):
    """
    Scalar loop matching for quadrangles (polygon IoU does not vectorize cleanly).
    """
    best_iou = match_iou
    best_index = -1
    for i in range(len(boxes_list)):
        box = boxes_list[i]
        if box[0] != new_box[0]:
            continue
        iou = bb_intersection_over_union_quadrangle(box[4:12], new_box[4:12])
        if iou > best_iou:
            best_index = i
            best_iou = iou
    return best_index, best_iou


def weighted_boxes_fusion_quadrangle(
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
    Weighted Boxes Fusion for quadrangle (4-vertex) boxes, as used by annotation
    formats like DOTA and HRSC2016.

    :param boxes_list: list of boxes predictions from each model, each box is 8 numbers:
    (x1, y1, x2, y2, x3, y3, x4, y4) - the four corner vertices of the (possibly rotated)
    box. It has 3 dimensions (models_number, model_preds, 8). All coordinates are float
    normalized values [0; 1] relative to image width/height, matching the convention of
    the axis-aligned weighted_boxes_fusion. The four vertices may be provided in any order
    or winding: they are canonically re-ordered (CCW, starting top-left) during prefiltering
    so that corresponding vertices from different models are fused together.
    :param scores_list: list of scores for each model
    :param labels_list: list of labels for each model
    :param weights: list of weights for each model. Default: None, which means weight == 1 for each model
    :param iou_thr: rotated (polygon) IoU value for boxes to be a match
    :param skip_box_thr: exclude boxes with score lower than this variable
    :param conf_type: how to calculate confidence in weighted boxes.
        'avg': average value,
        'max': maximum value,
        'box_and_model_avg': box and model wise hybrid weighted average,
        'absent_model_aware_avg': weighted average that takes into account the absent model.
    :param allows_overflow: false if we want confidence score not exceed 1.0

    :return: boxes: boxes coordinates (Order: x1, y1, x2, y2, x3, y3, x4, y4).
    :return: scores: confidence scores
    :return: labels: boxes labels

    NOTE: quadrangle IoU is computed via polygon clipping (Sutherland-Hodgman) + shoelace
    area formula on the 4 corners of each box; no shapely/opencv dependency is required.
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

    filtered_boxes = prefilter_boxes(boxes_list, scores_list, labels_list, weights, skip_box_thr)
    if len(filtered_boxes) == 0:
        return np.zeros((0, 8)), np.zeros((0,)), np.zeros((0,))

    overall_boxes = []
    for label in filtered_boxes:
        boxes = filtered_boxes[label]
        new_boxes = []
        weighted_boxes = np.empty((0, 12))

        # Clusterize boxes
        for j in range(0, len(boxes)):
            index, best_iou = find_matching_box(weighted_boxes, boxes[j], iou_thr)

            if index != -1:
                new_boxes[index].append(boxes[j])
                weighted_boxes[index] = get_weighted_box(new_boxes[index], conf_type)
            else:
                new_boxes.append([boxes[j].copy()])
                weighted_boxes = np.vstack((weighted_boxes, boxes[j].copy()))

        # Rescale confidence based on number of models and boxes
        for i in range(len(new_boxes)):
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
