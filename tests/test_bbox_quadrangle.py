import unittest
import numpy as np
from ensemble_boxes import *
from ensemble_boxes.ensemble_boxes_wbf_quadrangle import bb_intersection_over_union_quadrangle
from ensemble_boxes.ensemble_boxes_wbf_rotated import bb_intersection_over_union_rotated, rotated_box_corners, polygon_area


class TestWBFQuadrangle(unittest.TestCase):
    def test_same_box_fusion(self):
        # Two axis-aligned quads (given as 4 corners), highly overlapping.
        # corners: top-left, top-right, bottom-right, bottom-left
        boxes_list = [
            [[0.40, 0.45, 0.60, 0.45, 0.60, 0.55, 0.40, 0.55]],
            [[0.42, 0.47, 0.62, 0.47, 0.62, 0.57, 0.42, 0.57]],
        ]
        scores_list = [[0.9], [0.8]]
        labels_list = [[0], [0]]
        weights = [1, 1]

        boxes, scores, labels = weighted_boxes_fusion_quadrangle(
            boxes_list,
            scores_list,
            labels_list,
            weights=weights,
            iou_thr=0.5,
            skip_box_thr=0.0001,
            conf_type='avg',
            allows_overflow=True,
        )

        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes.shape[1], 8)
        # Each of the 8 coordinates is the score-weighted average of the two inputs.
        boxA = np.array([0.40, 0.45, 0.60, 0.45, 0.60, 0.55, 0.40, 0.55])
        boxB = np.array([0.42, 0.47, 0.62, 0.47, 0.62, 0.57, 0.42, 0.57])
        expected = (boxA * 0.9 + boxB * 0.8) / (0.9 + 0.8)
        # Compare as sets of ordered vertices (both inputs share the same canonical order here).
        np.testing.assert_allclose(np.sort(boxes[0]), np.sort(expected), atol=1e-4)
        np.testing.assert_allclose(scores[0], (0.9 + 0.8) / 2)
        np.testing.assert_array_equal(labels, [0])

    def test_vertex_order_invariance(self):
        # Same physical box, but the second model lists the corners starting from a
        # different vertex and with reversed winding. Canonical ordering must make
        # them fuse into a single box with the correct averaged geometry.
        box_a = [0.40, 0.45, 0.60, 0.45, 0.60, 0.55, 0.40, 0.55]  # TL, TR, BR, BL
        box_b = [0.60, 0.55, 0.40, 0.55, 0.40, 0.45, 0.60, 0.45]  # BR, BL, TL, TR (rotated start)
        boxes_list = [[box_a], [box_b]]
        scores_list = [[0.9], [0.9]]
        labels_list = [[0], [0]]

        boxes, scores, labels = weighted_boxes_fusion_quadrangle(
            boxes_list,
            scores_list,
            labels_list,
            weights=[1, 1],
            iou_thr=0.5,
            skip_box_thr=0.0001,
        )

        self.assertEqual(len(boxes), 1)
        # Both describe the identical box, so the fused corners must be that same box.
        np.testing.assert_allclose(np.sort(boxes[0]), np.sort(np.array(box_a)), atol=1e-4)

    def test_near_axis_aligned_tilt_fusion(self):
        # Regression: the same 0.4x0.2 box predicted tilted by +2 and -2 degrees. The
        # top-most vertex is the top-left corner in one and the top-right corner in the
        # other, so the canonical order starts at different corners. Corresponding
        # corners must still be averaged together (previously fused into a diamond
        # with area ~0.0435 instead of ~0.08).
        box_pos = rotated_box_corners(0.5, 0.5, 0.4, 0.2, 2).reshape(-1).tolist()
        box_neg = rotated_box_corners(0.5, 0.5, 0.4, 0.2, -2).reshape(-1).tolist()
        boxes_list = [[box_pos], [box_neg]]
        scores_list = [[0.9], [0.9]]
        labels_list = [[0], [0]]

        boxes, scores, labels = weighted_boxes_fusion_quadrangle(
            boxes_list,
            scores_list,
            labels_list,
            weights=[1, 1],
            iou_thr=0.5,
            skip_box_thr=0.0001,
        )

        self.assertEqual(len(boxes), 1)
        fused = boxes[0].reshape(4, 2)
        # Equal scores: the average of +2 and -2 degrees is the axis-aligned box
        # (shrunk by cos(2 deg), i.e. ~1e-4 off at the corners).
        self.assertAlmostEqual(polygon_area(fused), 0.08, places=3)
        expected = np.array([[0.3, 0.4], [0.7, 0.4], [0.7, 0.6], [0.3, 0.6]])
        np.testing.assert_allclose(fused[np.lexsort(fused.T)], expected[np.lexsort(expected.T)], atol=1e-3)

    def test_low_iou_boxes_stay_separate(self):
        boxes_list = [
            [[0.05, 0.05, 0.15, 0.05, 0.15, 0.15, 0.05, 0.15]],
            [[0.80, 0.80, 0.90, 0.80, 0.90, 0.90, 0.80, 0.90]],
        ]
        scores_list = [[0.9], [0.8]]
        labels_list = [[0], [0]]

        boxes, scores, labels = weighted_boxes_fusion_quadrangle(
            boxes_list,
            scores_list,
            labels_list,
            weights=[1, 1],
            iou_thr=0.5,
            skip_box_thr=0.0001,
        )

        self.assertEqual(len(boxes), 2)
        np.testing.assert_array_equal(labels, [0, 0])

    def test_simple_case_for_all_methods_quadrangle(self):
        boxes_list = []
        scores_list = []
        labels_list = []
        weights = []
        fixed_score = 0.8
        # A rotated (non axis-aligned) quad given canonically (CCW, top-left first).
        fixed_box = [0.45, 0.40, 0.60, 0.55, 0.55, 0.60, 0.40, 0.45]
        n_models = 5
        for _ in range(n_models):
            boxes_list.append([fixed_box])
            scores_list.append([fixed_score])
            labels_list.append([0])
            weights.append(1 / n_models)

        for conf_type in ['avg', 'max', 'box_and_model_avg', 'absent_model_aware_avg']:
            for allows_overflow in [True, False]:
                boxes, scores, labels = weighted_boxes_fusion_quadrangle(
                    boxes_list,
                    scores_list,
                    labels_list,
                    weights=weights,
                    iou_thr=0.4,
                    skip_box_thr=0.,
                    conf_type=conf_type,
                    allows_overflow=allows_overflow,
                )
                np.testing.assert_allclose(scores, [fixed_score])
                np.testing.assert_array_equal(labels, [0])
                np.testing.assert_allclose(np.sort(boxes[0]), np.sort(np.array(fixed_box)), atol=1e-4)

    def test_iou_quadrangle_matches_axis_aligned(self):
        # angle=0 quads must match closed-form axis-aligned IoU.
        # boxA covers x[0.1,0.5] y[0.1,0.5]; boxB covers x[0.15,0.55] y[0.15,0.55]
        boxA = [0.1, 0.1, 0.5, 0.1, 0.5, 0.5, 0.1, 0.5]
        boxB = [0.15, 0.15, 0.55, 0.15, 0.55, 0.55, 0.15, 0.55]

        inter = (0.5 - 0.15) * (0.5 - 0.15)
        areaA = 0.4 * 0.4
        areaB = 0.4 * 0.4
        expected = inter / (areaA + areaB - inter)
        np.testing.assert_allclose(bb_intersection_over_union_quadrangle(boxA, boxB), expected, atol=1e-6)

        # Non-overlapping
        boxC = [0.8, 0.8, 0.9, 0.8, 0.9, 0.9, 0.8, 0.9]
        np.testing.assert_allclose(bb_intersection_over_union_quadrangle(boxA, boxC), 0.0, atol=1e-9)

    def test_iou_quadrangle_matches_rotated(self):
        # A 45-degree rotated square expressed as a quad must give the same IoU
        # against an axis-aligned square as the (cx, cy, w, h, angle) rotated path.
        s = 0.2
        cx = cy = 0.5
        axis_quad = [cx - s / 2, cy - s / 2, cx + s / 2, cy - s / 2,
                     cx + s / 2, cy + s / 2, cx - s / 2, cy + s / 2]
        # square rotated 45 deg about its center -> corners at distance s/sqrt(2) along axes
        d = s / np.sqrt(2)
        rot_quad = [cx, cy - d, cx + d, cy, cx, cy + d, cx - d, cy]

        iou_quad = bb_intersection_over_union_quadrangle(axis_quad, rot_quad)
        iou_rot = bb_intersection_over_union_rotated((cx, cy, s, s, 0.0), (cx, cy, s, s, 45.0))
        np.testing.assert_allclose(iou_quad, iou_rot, atol=1e-6)

    def test_cross_validation_with_2d_wbf(self):
        # Same scenario as test_bbox.py's test_avg, expressed as axis-aligned quads.
        def to_quad(x1, y1, x2, y2):
            return [x1, y1, x2, y1, x2, y2, x1, y2]

        boxes_2d = [
            [[0.10, 0.10, 0.50, 0.50], [0.11, 0.11, 0.51, 0.51], [0.60, 0.60, 0.80, 0.80]],
            [[0.59, 0.59, 0.79, 0.79], [0.61, 0.61, 0.81, 0.81], [0.80, 0.80, 0.90, 0.90]],
        ]
        boxes_list = [[to_quad(*b) for b in model] for model in boxes_2d]
        scores_list = [[0.9, 0.8, 0.7], [0.85, 0.75, 0.65]]
        labels_list = [[1, 1, 1], [1, 1, 0]]
        weights = [2, 1]

        boxes, scores, labels = weighted_boxes_fusion_quadrangle(
            boxes_list, scores_list, labels_list, weights=weights,
            iou_thr=0.5, skip_box_thr=0.0001, conf_type='avg', allows_overflow=True,
        )
        boxes_ref, scores_ref, labels_ref = weighted_boxes_fusion(
            boxes_2d, scores_list, labels_list, weights=weights,
            iou_thr=0.5, skip_box_thr=0.0001, conf_type='avg', allows_overflow=True,
        )

        # Convert fused quads back to axis-aligned (x1,y1,x2,y2) via min/max of corners.
        xs = boxes[:, 0::2]
        ys = boxes[:, 1::2]
        converted = np.stack([xs.min(1), ys.min(1), xs.max(1), ys.max(1)], axis=1)
        np.testing.assert_allclose(converted, boxes_ref, atol=1e-5)
        np.testing.assert_allclose(scores, scores_ref, atol=1e-5)
        np.testing.assert_array_equal(labels, labels_ref)


if __name__ == "__main__":
    unittest.main()
