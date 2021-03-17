import unittest
import numpy as np
from ensemble_boxes import *


class TestWBF(unittest.TestCase):
    def test_weighted_avg(self):
        boxes_list = [
            [
                [0.1, 0.1, 0.5, 0.5],
                [0.2, 0.2, 0.5, 0.5],
                [0.45, 0.45, 0.5, 0.5],
            ],
            [
                [0.3, 0.3, 0.6, 0.6],
                [0.8, 0.8, 0.9, 0.9],
            ],
        ]

        scores_list = [[0.9, 0.7, 0.2], [0.5, 0.95]]
        labels_list = [[1, 1, 1], [1, 0]]
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
            conf_type='weighted_avg'
        )

        ## test for bbox

        # bbox with score = 0.95 (no overlap)
        np.testing.assert_allclose(boxes[0], [0.8, 0.8, 0.9, 0.9])

        # bbox with score = 0.9 and 0.7 (overlapped)
        # x1 (or y1) = (0.9 * 0.1 + 0.7 * 0.2) / (0.9 + 0.7) = 0.14375
        # x2 (or y2) = (0.9 * 0.5 + 0.7 * 0.5) / (0.9 + 0.7) = 0.5
        np.testing.assert_allclose(boxes[1], [0.14375, 0.14375, 0.5, 0.5])

        # bbox with score = 0.5 (no overlap)
        np.testing.assert_allclose(boxes[2], [0.3, 0.3, 0.6, 0.6])

        # bbox with score = 0.2 (no overlap)
        np.testing.assert_allclose(boxes[3], [0.45, 0.45, 0.5, 0.5])

        ## test for scores

        # no overlap
        np.testing.assert_allclose(scores[0], 0.95)

        # overlap 0.9 and 0.7
        # (2 * 0.9 + 2 * 0.7) / (2 + 2) = 0.8
        np.testing.assert_allclose(scores[1], 0.8)

        # no overlap
        np.testing.assert_allclose(scores[2], 0.5)

        # no overlap
        np.testing.assert_allclose(scores[3], 0.2)

        ## test for labels
        np.testing.assert_array_equal(labels, [0, 1, 1, 1])

if __name__ == "__main__":
    unittest.main()
