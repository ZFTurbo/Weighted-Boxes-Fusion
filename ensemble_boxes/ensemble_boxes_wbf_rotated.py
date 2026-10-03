# coding: utf-8
__author__ = 'TatiGabru: https://kaggle.com/blondinka'


import warnings
import numpy as np


def rotated_box_corners(cx, cy, w, h, angle_deg):
    """
    Compute the 4 corner vertices (CCW order) of a rotated rectangle.
    :param cx, cy: box center
    :param w, h: box width (long edge) and height (short edge)
    :param angle_deg: rotation of the width-edge relative to the x-axis, in degrees
    :return: 4x2 numpy array of (x, y) corners
    """
    theta = np.radians(angle_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    dx, dy = w / 2.0, h / 2.0
    local = np.array([(-dx, -dy), (dx, -dy), (dx, dy), (-dx, dy)])
    rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    return local.dot(rot.T) + np.array([cx, cy])


def polygon_clip(subject_polygon, clip_polygon):
    """
    Sutherland-Hodgman clipping of convex polygon `subject_polygon` against
    convex polygon `clip_polygon`. Both polygons must be given in CCW order.
    :return: list of (x, y) points of the clipped polygon (possibly empty)
    """
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def line_intersection(a, b, p, q):
        # Intersection of infinite line AB with segment PQ
        a1 = b[1] - a[1]
        b1 = a[0] - b[0]
        c1 = a1 * a[0] + b1 * a[1]
        a2 = q[1] - p[1]
        b2 = p[0] - q[0]
        c2 = a2 * p[0] + b2 * p[1]
        det = a1 * b2 - a2 * b1
        if abs(det) < 1e-12:
            return p
        x = (b2 * c1 - b1 * c2) / det
        y = (a1 * c2 - a2 * c1) / det
        return (x, y)

    output = list(subject_polygon)
    n = len(clip_polygon)
    for i in range(n):
        if len(output) == 0:
            return []
        A = clip_polygon[i]
        B = clip_polygon[(i + 1) % n]
        input_list = output
        output = []
        m = len(input_list)
        for j in range(m):
            P = input_list[j]
            Q = input_list[(j + 1) % m]
            p_inside = cross(A, B, P) >= 0
            q_inside = cross(A, B, Q) >= 0
            if q_inside:
                if not p_inside:
                    output.append(line_intersection(A, B, P, Q))
                output.append(Q)
            elif p_inside:
                output.append(line_intersection(A, B, P, Q))
    return output


def polygon_area(pts):
    """ Shoelace formula. """
    if len(pts) < 3:
        return 0.0
    area = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def bb_intersection_over_union_rotated(boxA, boxB):
    """
    IoU of two rotated boxes, each given as (cx, cy, w, h, angle_deg).
    """
    cornersA = rotated_box_corners(*boxA)
    cornersB = rotated_box_corners(*boxB)
    inter_pts = polygon_clip(cornersA, cornersB)
    inter_area = polygon_area(inter_pts)
    areaA = boxA[2] * boxA[3]
    areaB = boxB[2] * boxB[3]
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
            cx = float(box_part[0])
            cy = float(box_part[1])
            w = float(box_part[2])
            h = float(box_part[3])
            angle = float(box_part[4])

            # Box data checks
            if cx < 0:
                warnings.warn('CX < 0 in box. Set it to 0.')
                cx = 0
            if cx > 1:
                warnings.warn('CX > 1 in box. Set it to 1. Check that you normalize boxes in [0, 1] range.')
                cx = 1
            if cy < 0:
                warnings.warn('CY < 0 in box. Set it to 0.')
                cy = 0
            if cy > 1:
                warnings.warn('CY > 1 in box. Set it to 1. Check that you normalize boxes in [0, 1] range.')
                cy = 1
            if w < 0:
                warnings.warn('W < 0 in box. Set it to 0.')
                w = 0
            if w > 1:
                warnings.warn('W > 1 in box. Set it to 1. Check that you normalize boxes in [0, 1] range.')
                w = 1
            if h < 0:
                warnings.warn('H < 0 in box. Set it to 0.')
                h = 0
            if h > 1:
                warnings.warn('H > 1 in box. Set it to 1. Check that you normalize boxes in [0, 1] range.')
                h = 1
            if w * h == 0.0:
                warnings.warn("Zero area box skipped: {}.".format(box_part))
                continue

            # le90 convention: W is always the longer edge. Swap and rotate if violated.
            if w < h:
                warnings.warn('W < H in box (violates le90 long-edge convention). Swapping W/H and rotating angle by 90 degrees.')
                w, h = h, w
                angle = angle + 90

            # Wrap angle into [-90, 90)
            if angle < -90 or angle >= 90:
                warnings.warn('Angle out of [-90, 90) range in box. Wrapping it.')
            angle = wrap_angle(angle)

            # [label, score, weight, model index, cx, cy, w, h, angle]
            b = [int(label), float(score) * weights[t], weights[t], t, cx, cy, w, h, angle]
            if label not in new_boxes:
                new_boxes[label] = []
            new_boxes[label].append(b)

    # Sort each list in dict by score and transform it to numpy array
    for k in new_boxes:
        current_boxes = np.array(new_boxes[k])
        new_boxes[k] = current_boxes[current_boxes[:, 1].argsort()[::-1]]

    return new_boxes


def wrap_angle(angle):
    """ Wrap angle in degrees into [-90, 90). """
    return ((angle + 90) % 180) - 90


def align_to_reference(w, h, angle, ref_w, ref_h, ref_angle):
    """
    A rotated box has two equivalent parametrizations: (w, h, angle) and
    (h, w, angle + 90). Pick the one closest to the reference box, with the angle
    unwrapped to lie within 90 degrees of ref_angle, so that boxes can be averaged
    linearly. Closeness is |log w ratio| + |log h ratio| + |angle difference in radians|:
    elongated boxes keep their own orientation, while near-square boxes predicted
    ~90 degrees apart are swapped so they fuse to the shared orientation instead of
    the meaningless midpoint between the two.
    :return: (w, h, angle) aligned to the reference
    """
    d_same = wrap_angle(angle - ref_angle)
    d_swap = wrap_angle(angle + 90 - ref_angle)
    cost_same = abs(np.log(w / ref_w)) + abs(np.log(h / ref_h)) + abs(np.radians(d_same))
    cost_swap = abs(np.log(h / ref_w)) + abs(np.log(w / ref_h)) + abs(np.radians(d_swap))
    if cost_swap < cost_same:
        return h, w, ref_angle + d_swap
    return w, h, ref_angle + d_same


def get_weighted_box(boxes, conf_type='avg'):
    """
    Create weighted box for set of rotated boxes
    :param boxes: set of boxes to fuse. boxes[0] (the highest scoring one) is the reference
        that all other boxes are aligned to before averaging.
    :param conf_type: type of confidence one of 'avg', 'max', 'box_and_model_avg', 'absent_model_aware_avg'
    :return: weighted box (label, score, weight, model index, cx, cy, w, h, angle)
    """

    box = np.zeros(9, dtype=np.float32)
    conf = 0
    conf_list = []
    w = 0
    ref_w, ref_h, ref_angle = boxes[0][6], boxes[0][7], boxes[0][8]
    sum_cx = sum_cy = sum_w = sum_h = sum_angle = 0.0
    for b in boxes:
        bw, bh, bangle = align_to_reference(b[6], b[7], b[8], ref_w, ref_h, ref_angle)
        sum_cx += b[1] * b[4]
        sum_cy += b[1] * b[5]
        sum_w += b[1] * bw
        sum_h += b[1] * bh
        sum_angle += b[1] * bangle
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
    fused_w, fused_h, fused_angle = sum_w / conf, sum_h / conf, sum_angle / conf
    # Back to le90: w is the long edge, angle in [-90, 90)
    if fused_w < fused_h:
        fused_w, fused_h, fused_angle = fused_h, fused_w, fused_angle + 90
    box[4:9] = [sum_cx / conf, sum_cy / conf, fused_w, fused_h, wrap_angle(fused_angle)]
    return box


def find_matching_box(boxes_list, new_box, match_iou):
    """
    Scalar loop matching (analogous to the 3D WBF variant): rotated-rectangle
    IoU is computed via polygon clipping and does not vectorize as cleanly as
    the axis-aligned case, so this stays a per-pair loop.
    """
    best_iou = match_iou
    best_index = -1
    for i in range(len(boxes_list)):
        box = boxes_list[i]
        if box[0] != new_box[0]:
            continue
        iou = bb_intersection_over_union_rotated(box[4:9], new_box[4:9])
        if iou > best_iou:
            best_index = i
            best_iou = iou
    return best_index, best_iou


def weighted_boxes_fusion_rotated(
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
    :param boxes_list: list of boxes predictions from each model, each box is 5 numbers:
    (cx, cy, w, h, angle). It has 3 dimensions (models_number, model_preds, 5).
    cx, cy, w, h are float normalized coordinates [0; 1] relative to image width/height,
    matching the convention of the axis-aligned weighted_boxes_fusion.
    angle is in DEGREES and uses the le90 (long-edge 90) layout: angle in [-90, 90),
    and w is always the LONGER edge of the box (w >= h), with angle being the rotation of
    that long edge from the +x axis towards the +y axis (clockwise on screen, as image y
    points down). Results do not depend on the rotation direction as long as all inputs
    use the same one. Boxes with w < h or angle outside [-90, 90) describe the same
    rectangle and are normalized during prefiltering (w/h swapped and angle rotated by
    90 degrees, angle wrapped modulo 180).
    Other libraries use different units/ranges; convert before calling:
        MMRotate 'le90' (radians): angle = np.degrees(angle)
        Ultralytics YOLO OBB xywhr (radians): angle = np.degrees(r)
        Detectron2 RotatedBoxes (degrees, counter-clockwise): pass as is
        OpenCV minAreaRect (degrees): pass as is
    and normalize cx, cy, w, h to [0; 1].
    :param scores_list: list of scores for each model
    :param labels_list: list of labels for each model
    :param weights: list of weights for each model. Default: None, which means weight == 1 for each model
    :param iou_thr: rotated IoU value for boxes to be a match
    :param skip_box_thr: exclude boxes with score lower than this variable
    :param conf_type: how to calculate confidence in weighted boxes.
        'avg': average value,
        'max': maximum value,
        'box_and_model_avg': box and model wise hybrid weighted average,
        'absent_model_aware_avg': weighted average that takes into account the absent model.
    :param allows_overflow: false if we want confidence score not exceed 1.0

    :return: boxes: boxes coordinates (Order: cx, cy, w, h, angle[degrees]).
    :return: scores: confidence scores
    :return: labels: boxes labels

    NOTE: before averaging, each box in a cluster is aligned to the highest scoring box:
    its angle is unwrapped to within 90 degrees of the reference (handles the +/-90 degree
    wraparound, e.g. -89 and 89 fuse to -90, not 0), and near-square boxes predicted ~90
    degrees apart are re-expressed with w/h swapped so they fuse to their shared orientation.
    NOTE: rotated IoU is computed via polygon clipping (Sutherland-Hodgman) + shoelace
    area formula on the 4 corners of each rotated rectangle; no shapely/opencv
    dependency is required.
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

    angles = [float(b[4]) for model_boxes in boxes_list for b in model_boxes]
    if any(a != 0 for a in angles) and max(abs(a) for a in angles) <= np.pi / 2:
        warnings.warn('All angles are within [-pi/2, pi/2]. Angles must be in degrees, convert radians with np.degrees().')

    filtered_boxes = prefilter_boxes(boxes_list, scores_list, labels_list, weights, skip_box_thr)
    if len(filtered_boxes) == 0:
        return np.zeros((0, 5)), np.zeros((0,)), np.zeros((0,))

    overall_boxes = []
    for label in filtered_boxes:
        boxes = filtered_boxes[label]
        new_boxes = []
        weighted_boxes = np.empty((0, 9))

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
