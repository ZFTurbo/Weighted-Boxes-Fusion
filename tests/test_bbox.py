import unittest
import numpy as np
from ensemble_boxes import *


class TestWBF(unittest.TestCase):
    def test_box_and_model_avg(self):
        boxes_list = [
            [
                [0.10, 0.10, 0.50, 0.50], # cluster 2
                [0.11, 0.11, 0.51, 0.51], # cluster 2
                [0.60, 0.60, 0.80, 0.80], # cluster 1

            ],
            [
                [0.59, 0.59, 0.79, 0.79], # cluster 1
                [0.61, 0.61, 0.81, 0.81], # cluster 1
                [0.80, 0.80, 0.90, 0.90], # cluster 3
            ],
        ]

        scores_list = [[0.9, 0.8, 0.7], [0.85, 0.75, 0.65]]
        labels_list = [[1, 1, 1], [1, 1, 0]]
        weights = [2, 1]
        iou_thr = 0.5
        skip_box_thr = 0.0001

        boxes, scores, labels = weighted_boxes_fusion(
            boxes_list,
            scores_list,
            labels_list,
            weights=weights,
            iou_thr=iou_thr,
            skip_box_thr=skip_box_thr,
            conf_type='box_and_model_avg'
        )

        print("box_and_model_avg")
        print(boxes)
        print(scores)

        ## test for bbox

        # cluster 1
        np.testing.assert_allclose(boxes[0][0],
                                   (0.60 * 0.7 * 2 + 0.59 * 0.85 * 1 + 0.61 * 0.75 * 1) / (0.7 * 2 + 0.85 * 1 + 0.75 * 1))
        np.testing.assert_allclose(boxes[0][1],
                                   (0.60 * 0.7 * 2 + 0.59 * 0.85 * 1 + 0.61 * 0.75 * 1) / (0.7 * 2 + 0.85 * 1 + 0.75 * 1))
        np.testing.assert_allclose(boxes[0][2],
                                   (0.80 * 0.7 * 2 + 0.79 * 0.85 * 1 + 0.81 * 0.75 * 1) / (0.7 * 2 + 0.85 * 1 + 0.75 * 1))
        np.testing.assert_allclose(boxes[0][3],
                                   (0.80 * 0.7 * 2 + 0.79 * 0.85 * 1 + 0.81 * 0.75 * 1) / (0.7 * 2 + 0.85 * 1 + 0.75 * 1))

        # cluster 2
        np.testing.assert_allclose(boxes[1][0], (0.1 * 0.9 * 2 + 0.11 * 0.8 * 2) / (0.9 * 2 + 0.8 * 2))
        np.testing.assert_allclose(boxes[1][1], (0.1 * 0.9 * 2 + 0.11 * 0.8 * 2) / (0.9 * 2 + 0.8 * 2))
        np.testing.assert_allclose(boxes[1][2], (0.5 * 0.9 * 2 + 0.51 * 0.8 * 2) / (0.9 * 2 + 0.8 * 2))
        np.testing.assert_allclose(boxes[1][3], (0.5 * 0.9 * 2 + 0.51 * 0.8 * 2) / (0.9 * 2 + 0.8 * 2))

        # cluster 3
        np.testing.assert_allclose(boxes[2][0], (0.8 * 0.65 * 1) / (0.65 * 1))
        np.testing.assert_allclose(boxes[2][1], (0.8 * 0.65 * 1) / (0.65 * 1))
        np.testing.assert_allclose(boxes[2][2], (0.9 * 0.65 * 1) / (0.65 * 1))
        np.testing.assert_allclose(boxes[2][3], (0.9 * 0.65 * 1) / (0.65 * 1))

        ## test for scores

        # cluster 11c`
        box_avg = (0.7 * 2 + 0.85 * 1 + 0.75 * 1) / (2 + 1 + 1)
        model_avg = (2 + 1) / (2 + 1)
        np.testing.assert_allclose(scores[0],  box_avg * model_avg)

        # cluster 2
        box_avg = (0.9 * 2 + 0.8 * 2) / (2 + 2)
        model_avg = 2 / (2 + 1)
        np.testing.assert_allclose(scores[1],  box_avg * model_avg)

        # cluster 3
        box_avg = 0.65 * 1 / 1
        model_avg = 1 / (2 + 1)
        np.testing.assert_allclose(scores[2],  box_avg * model_avg)

        ## test for labels
        np.testing.assert_array_equal(labels, [1, 1, 0])

    def test_absent_model_aware_avg(self):
        boxes_list = [
            [
                [0.10, 0.10, 0.50, 0.50], # cluster 2
                [0.11, 0.11, 0.51, 0.51], # cluster 2
                [0.60, 0.60, 0.80, 0.80], # cluster 1

            ],
            [
                [0.59, 0.59, 0.79, 0.79], # cluster 1
                [0.61, 0.61, 0.81, 0.81], # cluster 1
                [0.80, 0.80, 0.90, 0.90], # cluster 3
            ],
        ]

        scores_list = [[0.9, 0.8, 0.7], [0.85, 0.75, 0.65]]
        labels_list = [[1, 1, 1], [1, 1, 0]]
        weights = [2, 1]
        iou_thr = 0.5
        skip_box_thr = 0.0001

        boxes, scores, labels = weighted_boxes_fusion(
            boxes_list,
            scores_list,
            labels_list,
            weights=weights,
            iou_thr=iou_thr,
            skip_box_thr=skip_box_thr,
            conf_type='absent_model_aware_avg'
        )
        print("absent_model_aware_avg")
        print(boxes)
        print(scores)

        ## test for bbox

        # cluster 1
        np.testing.assert_allclose(boxes[0][0],
                                   (0.60 * 0.7 * 2 + 0.59 * 0.85 * 1 + 0.61 * 0.75 * 1) / (0.7 * 2 + 0.85 * 1 + 0.75 * 1))
        np.testing.assert_allclose(boxes[0][1],
                                   (0.60 * 0.7 * 2 + 0.59 * 0.85 * 1 + 0.61 * 0.75 * 1) / (0.7 * 2 + 0.85 * 1 + 0.75 * 1))
        np.testing.assert_allclose(boxes[0][2],
                                   (0.80 * 0.7 * 2 + 0.79 * 0.85 * 1 + 0.81 * 0.75 * 1) / (0.7 * 2 + 0.85 * 1 + 0.75 * 1))
        np.testing.assert_allclose(boxes[0][3],
                                   (0.80 * 0.7 * 2 + 0.79 * 0.85 * 1 + 0.81 * 0.75 * 1) / (0.7 * 2 + 0.85 * 1 + 0.75 * 1))

        # cluster 2
        np.testing.assert_allclose(boxes[1][0], (0.1 * 0.9 * 2 + 0.11 * 0.8 * 2) / (0.9 * 2 + 0.8 * 2))
        np.testing.assert_allclose(boxes[1][1], (0.1 * 0.9 * 2 + 0.11 * 0.8 * 2) / (0.9 * 2 + 0.8 * 2))
        np.testing.assert_allclose(boxes[1][2], (0.5 * 0.9 * 2 + 0.51 * 0.8 * 2) / (0.9 * 2 + 0.8 * 2))
        np.testing.assert_allclose(boxes[1][3], (0.5 * 0.9 * 2 + 0.51 * 0.8 * 2) / (0.9 * 2 + 0.8 * 2))

        # cluster 3
        np.testing.assert_allclose(boxes[2][0], (0.8 * 0.65 * 1) / (0.65 * 1))
        np.testing.assert_allclose(boxes[2][1], (0.8 * 0.65 * 1) / (0.65 * 1))
        np.testing.assert_allclose(boxes[2][2], (0.9 * 0.65 * 1) / (0.65 * 1))
        np.testing.assert_allclose(boxes[2][3], (0.9 * 0.65 * 1) / (0.65 * 1))

        ## test for scores

        # cluster 1
        absent_weights = 0
        avg = (0.7 * 2 + 0.85 * 1 + 0.75 * 1) / (2 + 1 + 1 + absent_weights)
        np.testing.assert_allclose(scores[0],  avg)

        # cluster 2
        absent_weights = 1
        avg = (0.9 * 2 + 0.8 * 2) / (2 + 2 + absent_weights)
        np.testing.assert_allclose(scores[1],  avg)

        # cluster 3
        absent_weights = 2
        avg = 0.65 * 1 / (1 + absent_weights)
        np.testing.assert_allclose(scores[2], avg)

        ## test for labels
        np.testing.assert_array_equal(labels, [1, 1, 0])


    def test_avg(self):
        boxes_list = [
            [
                [0.10, 0.10, 0.50, 0.50], # cluster 2
                [0.11, 0.11, 0.51, 0.51], # cluster 2
                [0.60, 0.60, 0.80, 0.80], # cluster 1

            ],
            [
                [0.59, 0.59, 0.79, 0.79], # cluster 1
                [0.61, 0.61, 0.81, 0.81], # cluster 1
                [0.80, 0.80, 0.90, 0.90], # cluster 3
            ],
        ]

        scores_list = [[0.9, 0.8, 0.7], [0.85, 0.75, 0.65]]
        labels_list = [[1, 1, 1], [1, 1, 0]]
        weights = [2, 1]
        iou_thr = 0.5
        skip_box_thr = 0.0001

        boxes, scores, labels = weighted_boxes_fusion(
            boxes_list,
            scores_list,
            labels_list,
            weights=weights,
            iou_thr=iou_thr,
            skip_box_thr=skip_box_thr,
            conf_type='avg',
            allows_overflow=True
        )

        print("avg")
        print(boxes)
        print(scores)

        ## test for bbox

        # cluster 2
        np.testing.assert_allclose(boxes[0][0], (0.1 * 0.9 * 2 + 0.11 * 0.8 * 2) / (0.9 * 2 + 0.8 * 2))
        np.testing.assert_allclose(boxes[0][1], (0.1 * 0.9 * 2 + 0.11 * 0.8 * 2) / (0.9 * 2 + 0.8 * 2))
        np.testing.assert_allclose(boxes[0][2], (0.5 * 0.9 * 2 + 0.51 * 0.8 * 2) / (0.9 * 2 + 0.8 * 2))
        np.testing.assert_allclose(boxes[0][3], (0.5 * 0.9 * 2 + 0.51 * 0.8 * 2) / (0.9 * 2 + 0.8 * 2))

        # cluster 1
        np.testing.assert_allclose(boxes[1][0],
                                   (0.60 * 0.7 * 2 + 0.59 * 0.85 * 1 + 0.61 * 0.75 * 1) / (0.7 * 2 + 0.85 * 1 + 0.75 * 1))
        np.testing.assert_allclose(boxes[1][1],
                                   (0.60 * 0.7 * 2 + 0.59 * 0.85 * 1 + 0.61 * 0.75 * 1) / (0.7 * 2 + 0.85 * 1 + 0.75 * 1))
        np.testing.assert_allclose(boxes[1][2],
                                   (0.80 * 0.7 * 2 + 0.79 * 0.85 * 1 + 0.81 * 0.75 * 1) / (0.7 * 2 + 0.85 * 1 + 0.75 * 1))
        np.testing.assert_allclose(boxes[1][3],
                                   (0.80 * 0.7 * 2 + 0.79 * 0.85 * 1 + 0.81 * 0.75 * 1) / (0.7 * 2 + 0.85 * 1 + 0.75 * 1))

        # cluster 3
        np.testing.assert_allclose(boxes[2][0], (0.8 * 0.65 * 1) / (0.65 * 1))
        np.testing.assert_allclose(boxes[2][1], (0.8 * 0.65 * 1) / (0.65 * 1))
        np.testing.assert_allclose(boxes[2][2], (0.9 * 0.65 * 1) / (0.65 * 1))
        np.testing.assert_allclose(boxes[2][3], (0.9 * 0.65 * 1) / (0.65 * 1))

        ## test for scores

        # cluster 2
        avg = (0.9 * 2 + 0.8 * 2) / (2 + 1)
        np.testing.assert_allclose(scores[0], avg)

        # cluster 1
        avg = (0.7 * 2 + 0.85 * 1 + 0.75 * 1) / (2 + 1)
        np.testing.assert_allclose(scores[1], avg)

        # cluster 3
        avg = 0.65 * 1 / (2 + 1)
        np.testing.assert_allclose(scores[2], avg)

        ## test for labels
        np.testing.assert_array_equal(labels, [1, 1, 0])

    def test_simple_case_for_all_methods(self):
        boxes_list = []
        scores_list = []
        labels_list = []
        weigths = []
        fixed_score = 0.8
        fixed_box = [0., 0., 0.1, 0.1]
        n_models = 5
        # All models have the same result with one box
        for _ in range(n_models):
            boxes_list.append([fixed_box])
            scores_list.append([fixed_score])
            labels_list.append([0])
            weigths.append(1 / n_models)

        for conf_type in ['avg', 'max', 'box_and_model_avg', 'absent_model_aware_avg']:
            for allows_overflow in [True, False]:
                boxes, scores, labels = weighted_boxes_fusion(
                    boxes_list,
                    scores_list,
                    labels_list,
                    weights=weigths,
                    iou_thr=0.4,
                    skip_box_thr=0.,
                    conf_type=conf_type,
                    allows_overflow=allows_overflow
                )
                np.testing.assert_allclose(scores, [fixed_score])
                np.testing.assert_array_equal(labels, [0])
                np.testing.assert_allclose(boxes[0], fixed_box)

    def test_max_conf_type(self):
        boxes_list = [[
            [0.1, 0.1, 0.2, 0.2],
        ], [
            [0.1, 0.1, 0.2, 0.2],
        ]]
        scores_list = [[0.9], [0.8]]
        labels_list = [[0], [0]]
        weights = [1, 2]

        iou_thr = 0.5
        skip_box_thr = 0.0001

        boxes, scores, labels = weighted_boxes_fusion(
            boxes_list,
            scores_list,
            labels_list,
            weights=weights,
            iou_thr=iou_thr,
            skip_box_thr=skip_box_thr,
            conf_type='max'
        )
        # 0.9 * 1 < 0.8 * 2, so the result is 0.8
        np.testing.assert_allclose(scores, [0.8])

if __name__ == "__main__":
    unittest.main()
