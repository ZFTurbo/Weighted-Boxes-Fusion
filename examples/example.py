# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


import cv2
import numpy as np
from ensemble_boxes import *


def show_image(im, name='image'):
    cv2.imshow(name, im.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def gen_color_list(model_num, labels_num):
    color_list = np.zeros((model_num, labels_num, 3))
    colors_to_use = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 255, 0), (0, 0, 0)]
    total = 0
    for i in range(model_num):
        for j in range(labels_num):
            color_list[i, j, :] = colors_to_use[total]
            total = (total + 1) % len(colors_to_use)
    return color_list


def show_boxes(boxes_list, scores_list, labels_list, image_size=800):
    thickness = 5
    color_list = gen_color_list(len(boxes_list), len(np.unique(labels_list)))
    image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    image[...] = 255
    for i in range(len(boxes_list)):
        for j in range(len(boxes_list[i])):
            x1 = int(image_size * boxes_list[i][j][0])
            y1 = int(image_size * boxes_list[i][j][1])
            x2 = int(image_size * boxes_list[i][j][2])
            y2 = int(image_size * boxes_list[i][j][3])
            lbl = labels_list[i][j]
            cv2.rectangle(image, (x1, y1), (x2, y2), color_list[i][lbl], int(thickness * scores_list[i][j]))
    show_image(image)


def example_wbf_2_models(iou_thr=0.55, draw_image=True):
    """
    This example shows how to ensemble boxes from 2 models using WBF method    
    :return: 
    """

    boxes_list = [
        [
            [0.00, 0.51, 0.81, 0.91],
            [0.10, 0.31, 0.71, 0.61],
            [0.01, 0.32, 0.83, 0.93],
            [0.02, 0.53, 0.11, 0.94],
            [0.03, 0.24, 0.12, 0.35],
        ],
        [
            [0.04, 0.56, 0.84, 0.92],
            [0.12, 0.33, 0.72, 0.64],
            [0.38, 0.66, 0.79, 0.95],
            [0.08, 0.49, 0.21, 0.89],
        ],
    ]
    scores_list = [
        [
            0.9,
            0.8,
            0.2,
            0.4,
            0.7,
        ],
        [
            0.5,
            0.8,
            0.7,
            0.3,
        ]
    ]
    labels_list = [
        [
            0,
            1,
            0,
            1,
            1,
        ],
        [
            1,
            1,
            1,
            0,
        ]
    ]
    weights = [2, 1]
    if draw_image:
        show_boxes(boxes_list, scores_list, labels_list)

    boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=0.0)

    if draw_image:
        show_boxes([boxes], [scores], [labels.astype(np.int32)])

    print(len(boxes))
    print(boxes)


def example_wbf_1_model(iou_thr=0.55, draw_image=True):
    """
    This example shows how to ensemble boxes from single model using WBF method    
    :return: 
    """

    boxes_list = [
        [0.00, 0.51, 0.81, 0.91],
        [0.10, 0.31, 0.71, 0.61],
        [0.01, 0.32, 0.83, 0.93],
        [0.02, 0.53, 0.11, 0.94],
        [0.03, 0.24, 0.12, 0.35],
        [0.04, 0.56, 0.84, 0.92],
        [0.12, 0.33, 0.72, 0.64],
        [0.38, 0.66, 0.79, 0.95],
        [0.08, 0.49, 0.21, 0.89],
    ]
    scores_list = [0.9, 0.8, 0.2, 0.4, 0.7, 0.5, 0.8, 0.7, 0.3]
    labels_list = [0, 1, 0, 1, 1, 1, 1, 1, 0]

    if draw_image:
        show_boxes([boxes_list], [scores_list], [labels_list])

    boxes, scores, labels = weighted_boxes_fusion([boxes_list], [scores_list], [labels_list], weights=None, iou_thr=iou_thr, skip_box_thr=0.0)

    if draw_image:
        show_boxes([boxes], [scores], [labels.astype(np.int32)])

    print(len(boxes))
    print(boxes)


def example_nms_2_models(method, iou_thr=0.5, sigma=0.5, thresh=0.001, draw_image=True):
    """
    This example shows how to ensemble boxes from 2 models using NMS method    
    :return: 
    """

    boxes_list = [
        [
            [0.00, 0.51, 0.81, 0.91],
            [0.10, 0.31, 0.71, 0.61],
            [0.01, 0.32, 0.83, 0.93],
            [0.02, 0.53, 0.11, 0.94],
            [0.03, 0.24, 0.12, 0.35],
        ],
        [
            [0.04, 0.56, 0.84, 0.92],
            [0.12, 0.33, 0.72, 0.64],
            [0.38, 0.66, 0.79, 0.95],
            [0.08, 0.49, 0.21, 0.89],
        ],
    ]
    scores_list = [
        [
            0.9,
            0.8,
            0.2,
            0.4,
            0.7,
        ],
        [
            0.5,
            0.8,
            0.7,
            0.3,
        ]
    ]
    labels_list = [
        [
            0,
            1,
            0,
            1,
            1,
        ],
        [
            1,
            1,
            1,
            0,
        ]
    ]
    weights = [2, 1]

    if draw_image:
        show_boxes(boxes_list, scores_list, labels_list)

    boxes, scores, labels = nms_method(boxes_list, scores_list, labels_list, method=method, weights=weights, iou_thr=iou_thr, sigma=sigma, thresh=thresh)

    if draw_image:
        show_boxes([boxes], [scores], [labels.astype(np.int32)])

    print(len(boxes))
    print(boxes)


if __name__ == '__main__':
    draw_image = True
    example_wbf_2_models(draw_image=draw_image)
    example_wbf_1_model(draw_image=draw_image)
    example_nms_2_models(draw_image=draw_image, method=3, iou_thr=0.5, thresh=0.0)
    example_nms_2_models(draw_image=draw_image, method=2, iou_thr=0.3, sigma=0.05, thresh=0.001)
