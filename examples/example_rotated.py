# coding: utf-8
__author__ = 'TatiGabru: https://kaggle.com/blondinka'


import cv2
import numpy as np
from ensemble_boxes import *
from ensemble_boxes.ensemble_boxes_wbf_rotated import rotated_box_corners
from example import show_image, gen_color_list


def show_rotated_boxes(boxes_list, scores_list, labels_list, image_size=800):
    """
    Draw rotated boxes (cx, cy, w, h, angle) on a blank canvas, one color per
    (model, label), with line thickness scaling with the box score. Corners are
    computed with the same rotated_box_corners helper the fusion/IoU code uses,
    so what you see matches what WBF operates on.
    """
    thickness = 5
    color_list = gen_color_list(len(boxes_list), len(np.unique(labels_list)))
    image = np.full((image_size, image_size, 3), 255, dtype=np.uint8)
    for i in range(len(boxes_list)):
        for j in range(len(boxes_list[i])):
            cx, cy, w, h, angle = boxes_list[i][j]
            corners = rotated_box_corners(cx, cy, w, h, angle) * image_size
            pts = corners.astype(np.int32).reshape((-1, 1, 2))
            lbl = int(labels_list[i][j])
            color = tuple(int(c) for c in color_list[i][lbl])
            cv2.polylines(image, [pts], isClosed=True, color=color, thickness=max(1, int(thickness * scores_list[i][j])))
    show_image(image)


def example_wbf_rotated_2_models(iou_thr=0.55, draw_image=True):
    """
    This example shows how to ensemble rotated boxes from 2 models using WBF method.
    Each box is (cx, cy, w, h, angle), with angle in degrees using the le90
    (long-edge 90) convention: angle in [-90, 90), w is always the longer edge.
    :return:
    """

    boxes_list = [
        [
            [0.50, 0.50, 0.20, 0.10, 30.0],
            [0.20, 0.30, 0.15, 0.08, -45.0],
            [0.70, 0.75, 0.25, 0.05, 10.0],
        ],
        [
            [0.51, 0.51, 0.22, 0.11, 32.0],
            [0.19, 0.29, 0.14, 0.09, -40.0],
            [0.705, 0.745, 0.24, 0.06, 12.0],
        ],
    ]
    scores_list = [
        [0.9, 0.8, 0.7],
        [0.85, 0.75, 0.65],
    ]
    labels_list = [
        [0, 1, 0],
        [0, 1, 0],
    ]
    weights = [2, 1]

    if draw_image:
        show_rotated_boxes(boxes_list, scores_list, labels_list)

    boxes, scores, labels = weighted_boxes_fusion_rotated(
        boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=0.0
    )

    print(len(boxes))
    print(boxes)
    print(scores)
    print(labels)

    if draw_image:
        show_rotated_boxes([boxes], [scores], [labels.astype(np.int32)])


def example_wbf_rotated_1_model(iou_thr=0.55, draw_image=True):
    """
    This example shows how to ensemble rotated boxes from a single model using WBF method.
    Each box is (cx, cy, w, h, angle), with angle in degrees using the le90
    (long-edge 90) convention: angle in [-90, 90), w is always the longer edge.
    :return:
    """

    boxes_list = [
        [0.50, 0.50, 0.20, 0.10, 30.0],
        [0.20, 0.30, 0.15, 0.08, -45.0],
        [0.70, 0.75, 0.25, 0.05, 10.0],
    ]
    scores_list = [0.9, 0.8, 0.7]
    labels_list = [0, 1, 0]

    if draw_image:
        show_rotated_boxes([boxes_list], [scores_list], [labels_list])

    boxes, scores, labels = weighted_boxes_fusion_rotated(
        [boxes_list], [scores_list], [labels_list], weights=None, iou_thr=iou_thr, skip_box_thr=0.0
    )

    print(len(boxes))
    print(boxes)
    print(scores)
    print(labels)
    if draw_image:
        show_rotated_boxes([boxes], [scores], [labels.astype(np.int32)])


if __name__ == '__main__':
    draw_image = True
    example_wbf_rotated_2_models()
    example_wbf_rotated_1_model()
