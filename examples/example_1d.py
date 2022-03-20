# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


import numpy as np
from ensemble_boxes import *
import cv2


def show_image(im, name='image'):
    cv2.imshow(name, im.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_boxes(boxes_list, scores_list, labels_list, image_size=800):
    image = np.zeros((image_size // 2, image_size, 3), dtype=np.uint8)
    image[...] = 255

    y_pos = 30
    for i in range(len(boxes_list)):
        for j in range(len(boxes_list[i])):
            x1 = int(image_size * boxes_list[i][j][0])
            x2 = int(image_size * boxes_list[i][j][1])
            lbl = labels_list[i][j]
            print(x1, x2, lbl, scores_list[i][j])
            thickness = int(20 * scores_list[i][j])
            image[y_pos:y_pos + thickness, x1:x2, lbl] = 0
            y_pos += 30

    show_image(image)


def example_wbf_1d_2_models(
        iou_thr=0.55,
        draw_image=True
):
    """
    This example shows how to ensemble boxes from 2 models using WBF_1D method
    :return: 
    """

    boxes_list = [
        [
            [0.00, 0.21],
            [0.21, 0.51],
            [0.52, 0.67],
            [0.66, 0.80],
            [0.80, 0.85],
        ],
        [
            [0.05, 0.18],
            [0.22, 0.45],
            [0.52, 0.80],
            [0.84, 0.99],
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

    boxes, scores, labels = weighted_boxes_fusion_1d(
        boxes_list,
        scores_list,
        labels_list,
        weights=weights,
        iou_thr=iou_thr,
        skip_box_thr=0.01
    )

    if draw_image:
        show_boxes([boxes], [scores], [labels.astype(np.int32)])

    print(len(boxes))
    print(boxes)


if __name__ == '__main__':
    draw_image = True
    example_wbf_1d_2_models(
        iou_thr=0.2,
        draw_image=draw_image
    )

