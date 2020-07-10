# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from ensemble_boxes import *


def plot_cube(ax, cube_definition, lbl, thickness):
    cube_definition_array = [
        np.array(list(item))
        for item in cube_definition
    ]

    points = []
    points += cube_definition_array
    vectors = [
        cube_definition_array[1] - cube_definition_array[0],
        cube_definition_array[2] - cube_definition_array[0],
        cube_definition_array[3] - cube_definition_array[0]
    ]

    points += [cube_definition_array[0] + vectors[0] + vectors[1]]
    points += [cube_definition_array[0] + vectors[0] + vectors[2]]
    points += [cube_definition_array[0] + vectors[1] + vectors[2]]
    points += [cube_definition_array[0] + vectors[0] + vectors[1] + vectors[2]]

    points = np.array(points)

    edges = [
        [points[0], points[3], points[5], points[1]],
        [points[1], points[5], points[7], points[4]],
        [points[4], points[2], points[6], points[7]],
        [points[2], points[6], points[3], points[0]],
        [points[0], points[2], points[4], points[1]],
        [points[3], points[6], points[7], points[5]]
    ]

    faces = Poly3DCollection(edges, linewidths=thickness + 1)
    if lbl == 0:
        faces.set_edgecolor((1, 0, 0))
    else:
        faces.set_edgecolor((0, 0, 1))
    faces.set_facecolor((0, 0, 1, 0.1))

    ax.add_collection3d(faces)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=0)


def show_boxes(boxes_list, scores_list, labels_list, image_size=800):
    image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    image[...] = 255
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(len(boxes_list)):
        for j in range(len(boxes_list[i])):
            x1 = int(image_size * boxes_list[i][j][0])
            y1 = int(image_size * boxes_list[i][j][1])
            z1 = int(image_size * boxes_list[i][j][2])
            x2 = int(image_size * boxes_list[i][j][3])
            y2 = int(image_size * boxes_list[i][j][4])
            z2 = int(image_size * boxes_list[i][j][5])
            lbl = labels_list[i][j]
            cube_definition = [
                (x1, y1, z1), (x1, y2, z1), (x2, y1, z1), (x1, y1, z2)
            ]
            plot_cube(ax, cube_definition, lbl, int(4 * scores_list[i][j]))

    plt.show()


def example_wbf_3d_2_models(iou_thr=0.55, draw_image=True):
    """
    This example shows how to ensemble boxes from 2 models using WBF_3D method
    :return: 
    """

    boxes_list = [
        [
            [0.00, 0.51, 0.41, 0.81, 0.91, 0.78],
            [0.10, 0.31, 0.45, 0.71, 0.61, 0.85],
            [0.01, 0.32, 0.55, 0.83, 0.93, 0.95],
            [0.02, 0.53, 0.11, 0.11, 0.94, 0.55],
            [0.03, 0.24, 0.34, 0.12, 0.35, 0.67],
        ],
        [
            [0.04, 0.56, 0.36, 0.84, 0.92, 0.82],
            [0.12, 0.33, 0.46, 0.72, 0.64, 0.75],
            [0.38, 0.66, 0.55, 0.79, 0.95, 0.90],
            [0.08, 0.49, 0.15, 0.21, 0.89, 0.67],
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

    boxes, scores, labels = weighted_boxes_fusion_3d(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=0.0)

    if draw_image:
        show_boxes([boxes], [scores], [labels.astype(np.int32)])

    print(len(boxes))
    print(boxes)


if __name__ == '__main__':
    draw_image = True
    example_wbf_3d_2_models(iou_thr=0.2, draw_image=draw_image)

