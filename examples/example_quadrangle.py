# coding: utf-8
__author__ = 'TatiGabru: https://kaggle.com/blondinka'


import cv2
import numpy as np
from ensemble_boxes import *
from example import show_image, gen_color_list


def show_quadrangle_boxes(boxes_list, scores_list, labels_list, image_size=800):
    """
    Draw quadrangle boxes (x1, y1, x2, y2, x3, y3, x4, y4) - the four corners
    directly (DOTA / HRSC2016 declaration format) - on a blank canvas, one color
    per (model, label), with line thickness scaling with the box score.
    """
    thickness = 5
    color_list = gen_color_list(len(boxes_list), len(np.unique(labels_list)))
    image = np.full((image_size, image_size, 3), 255, dtype=np.uint8)
    for i in range(len(boxes_list)):
        for j in range(len(boxes_list[i])):
            corners = np.asarray(boxes_list[i][j], dtype=float).reshape(4, 2) * image_size
            pts = corners.astype(np.int32).reshape((-1, 1, 2))
            lbl = int(labels_list[i][j])
            color = tuple(int(c) for c in color_list[i][lbl])
            cv2.polylines(image, [pts], isClosed=True, color=color, thickness=max(1, int(thickness * scores_list[i][j])))
    show_image(image)


def example_wbf_quadrangle_2_models(iou_thr=0.55, draw_image=True):
    """
    This example shows how to ensemble quadrangle (4-vertex) boxes from 2 models
    using WBF method. Each box is (x1, y1, x2, y2, x3, y3, x4, y4) - the four
    corners, normalized to [0; 1]. This is the DOTA / HRSC2016 declaration format.
    :return:
    """

    boxes_list = [
        [
            [0.40, 0.45, 0.60, 0.45, 0.60, 0.55, 0.40, 0.55],
            [0.10, 0.10, 0.25, 0.12, 0.23, 0.27, 0.08, 0.25],
        ],
        [
            [0.42, 0.47, 0.62, 0.47, 0.62, 0.57, 0.42, 0.57],
            [0.11, 0.11, 0.26, 0.13, 0.24, 0.28, 0.09, 0.26],
        ],
    ]
    scores_list = [
        [0.9, 0.8],
        [0.85, 0.75],
    ]
    labels_list = [
        [0, 1],
        [0, 1],
    ]
    weights = [2, 1]

    if draw_image:
        show_quadrangle_boxes(boxes_list, scores_list, labels_list)

    boxes, scores, labels = weighted_boxes_fusion_quadrangle(
        boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=0.0
    )

    print(len(boxes))
    print(boxes)
    print(scores)
    print(labels)

    if draw_image:
        show_quadrangle_boxes([boxes], [scores], [labels.astype(np.int32)])


if __name__ == '__main__':
    draw_image = True
    example_wbf_quadrangle_2_models()
