import unittest
import numpy as np
from ensemble_boxes import *
from ensemble_boxes.ensemble_boxes_wbf_rotated import bb_intersection_over_union_rotated
from ensemble_boxes.ensemble_boxes_wbf_rotated import rotated_box_corners, polygon_area


class TestWBFRotated(unittest.TestCase):
    def test_same_angle_fusion(self):
        boxes_list = [
            [[0.50, 0.50, 0.20, 0.10, 30.0]],
            [[0.51, 0.51, 0.22, 0.11, 30.0]],
        ]
        scores_list = [[0.9], [0.8]]
        labels_list = [[0], [0]]
        weights = [1, 1]

        boxes, scores, labels = weighted_boxes_fusion_rotated(
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
        np.testing.assert_allclose(boxes[0][0], (0.50 * 0.9 + 0.51 * 0.8) / (0.9 + 0.8))
        np.testing.assert_allclose(boxes[0][1], (0.50 * 0.9 + 0.51 * 0.8) / (0.9 + 0.8))
        np.testing.assert_allclose(boxes[0][2], (0.20 * 0.9 + 0.22 * 0.8) / (0.9 + 0.8))
        np.testing.assert_allclose(boxes[0][3], (0.10 * 0.9 + 0.11 * 0.8) / (0.9 + 0.8))
        np.testing.assert_allclose(boxes[0][4], 30.0, atol=1e-4)
        np.testing.assert_allclose(scores[0], (0.9 + 0.8) / 2)
        np.testing.assert_array_equal(labels, [0])

    def test_angle_wraparound(self):
        boxes_list = [
            [[0.50, 0.50, 0.20, 0.19, -89.0]],
            [[0.50, 0.50, 0.20, 0.19, 89.0]],
        ]
        scores_list = [[0.9], [0.9]]
        labels_list = [[0], [0]]
        weights = [1, 1]

        boxes, scores, labels = weighted_boxes_fusion_rotated(
            boxes_list,
            scores_list,
            labels_list,
            weights=weights,
            iou_thr=0.5,
            skip_box_thr=0.0001,
            conf_type='avg',
        )

        self.assertEqual(len(boxes), 1)
        fused_angle = boxes[0][4]
        # Fused angle should sit at the +/-90 boundary, not collapse to 0
        # (which is what naive linear averaging of -89 and 89 would give).
        near_boundary = (abs(fused_angle - 90) < 1e-2) or (abs(fused_angle - (-90)) < 1e-2)
        self.assertTrue(near_boundary, "Expected fused angle near +/-90, got {}".format(fused_angle))
        self.assertGreater(abs(fused_angle), 1.0)

    def test_low_iou_boxes_stay_separate(self):
        boxes_list = [
            [[0.10, 0.10, 0.10, 0.05, 0.0]],
            [[0.80, 0.80, 0.10, 0.05, 45.0]],
        ]
        scores_list = [[0.9], [0.8]]
        labels_list = [[0], [0]]
        weights = [1, 1]

        boxes, scores, labels = weighted_boxes_fusion_rotated(
            boxes_list,
            scores_list,
            labels_list,
            weights=weights,
            iou_thr=0.5,
            skip_box_thr=0.0001,
        )

        self.assertEqual(len(boxes), 2)
        np.testing.assert_array_equal(labels, [0, 0])

    def test_simple_case_for_all_methods_rotated(self):
        boxes_list = []
        scores_list = []
        labels_list = []
        weights = []
        fixed_score = 0.8
        fixed_box = [0.5, 0.5, 0.2, 0.1, 15.0]
        n_models = 5
        for _ in range(n_models):
            boxes_list.append([fixed_box])
            scores_list.append([fixed_score])
            labels_list.append([0])
            weights.append(1 / n_models)

        for conf_type in ['avg', 'max', 'box_and_model_avg', 'absent_model_aware_avg']:
            for allows_overflow in [True, False]:
                boxes, scores, labels = weighted_boxes_fusion_rotated(
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
                np.testing.assert_allclose(boxes[0], fixed_box, atol=1e-4)

    def test_w_h_swap_correction(self):
        boxes_list = [[[0.5, 0.5, 0.05, 0.10, 30.0]]]
        scores_list = [[0.9]]
        labels_list = [[0]]

        boxes, scores, labels = weighted_boxes_fusion_rotated(
            boxes_list,
            scores_list,
            labels_list,
            weights=[1],
            iou_thr=0.5,
            skip_box_thr=0.0001,
        )

        self.assertEqual(len(boxes), 1)
        # w must end up as the longer edge (le90 convention), and h the shorter
        self.assertGreaterEqual(boxes[0][2], boxes[0][3])
        np.testing.assert_allclose(boxes[0][2], 0.10, atol=1e-4)
        np.testing.assert_allclose(boxes[0][3], 0.05, atol=1e-4)
        # angle should be rotated by +/-90 degrees from the original 30 -> -60 or 120(wrapped)
        np.testing.assert_allclose(boxes[0][4], -60.0, atol=1e-4)

    def test_axis_aligned_cross_validation(self):
        # Same scenario as test_bbox.py's test_avg, expressed as rotated boxes with angle=0.
        def to_rotated(x1, y1, x2, y2):
            return [(x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1, 0.0]

        boxes_list = [
            [
                to_rotated(0.10, 0.10, 0.50, 0.50),  # cluster 2
                to_rotated(0.11, 0.11, 0.51, 0.51),  # cluster 2
                to_rotated(0.60, 0.60, 0.80, 0.80),  # cluster 1
            ],
            [
                to_rotated(0.59, 0.59, 0.79, 0.79),  # cluster 1
                to_rotated(0.61, 0.61, 0.81, 0.81),  # cluster 1
                to_rotated(0.80, 0.80, 0.90, 0.90),  # cluster 3
            ],
        ]
        scores_list = [[0.9, 0.8, 0.7], [0.85, 0.75, 0.65]]
        labels_list = [[1, 1, 1], [1, 1, 0]]
        weights = [2, 1]

        boxes, scores, labels = weighted_boxes_fusion_rotated(
            boxes_list,
            scores_list,
            labels_list,
            weights=weights,
            iou_thr=0.5,
            skip_box_thr=0.0001,
            conf_type='avg',
            allows_overflow=True,
        )

        boxes_2d, scores_2d, labels_2d = weighted_boxes_fusion(
            [
                [[0.10, 0.10, 0.50, 0.50], [0.11, 0.11, 0.51, 0.51], [0.60, 0.60, 0.80, 0.80]],
                [[0.59, 0.59, 0.79, 0.79], [0.61, 0.61, 0.81, 0.81], [0.80, 0.80, 0.90, 0.90]],
            ],
            scores_list,
            labels_list,
            weights=weights,
            iou_thr=0.5,
            skip_box_thr=0.0001,
            conf_type='avg',
            allows_overflow=True,
        )

        # Convert rotated (cx,cy,w,h,angle=0) back to (x1,y1,x2,y2) for comparison.
        x1 = boxes[:, 0] - boxes[:, 2] / 2.0
        y1 = boxes[:, 1] - boxes[:, 3] / 2.0
        x2 = boxes[:, 0] + boxes[:, 2] / 2.0
        y2 = boxes[:, 1] + boxes[:, 3] / 2.0
        boxes_converted = np.stack([x1, y1, x2, y2], axis=1)

        np.testing.assert_allclose(boxes_converted, boxes_2d, atol=1e-5)
        np.testing.assert_allclose(scores, scores_2d, atol=1e-5)
        np.testing.assert_array_equal(labels, labels_2d)

    def test_iou_rotated_matches_axis_aligned(self):
        # angle=0 rotated IoU must match closed-form axis-aligned IoU.
        boxA = (0.3, 0.3, 0.4, 0.4, 0.0)  # x1,y1,x2,y2 = 0.1,0.1,0.5,0.5
        boxB = (0.35, 0.35, 0.4, 0.4, 0.0)  # x1,y1,x2,y2 = 0.15,0.15,0.55,0.55

        def axis_aligned_iou(bA, bB):
            xA = max(bA[0], bB[0])
            yA = max(bA[1], bB[1])
            xB = min(bA[2], bB[2])
            yB = min(bA[3], bB[3])
            inter = max(xB - xA, 0) * max(yB - yA, 0)
            areaA = (bA[2] - bA[0]) * (bA[3] - bA[1])
            areaB = (bB[2] - bB[0]) * (bB[3] - bB[1])
            return inter / (areaA + areaB - inter)

        expected = axis_aligned_iou((0.1, 0.1, 0.5, 0.5), (0.15, 0.15, 0.55, 0.55))
        actual = bb_intersection_over_union_rotated(boxA, boxB)
        np.testing.assert_allclose(actual, expected, atol=1e-6)

        # Non-overlapping boxes -> IoU 0
        boxC = (0.9, 0.9, 0.1, 0.1, 0.0)
        np.testing.assert_allclose(bb_intersection_over_union_rotated(boxA, boxC), 0.0, atol=1e-9)

    def test_iou_rotated_45_degree_square(self):
        # Two identical squares of side s, same center, one rotated 45 degrees.
        # Their intersection is a regular octagon; area is known in closed form:
        # A_octagon = s^2 * (2*sqrt(2) - 2) for a square rotated 45deg about its own center
        # intersected with the unrotated square of side s.
        s = 0.2
        boxA = (0.5, 0.5, s, s, 0.0)
        boxB = (0.5, 0.5, s, s, 45.0)

        area_square = s * s
        area_octagon = (s ** 2) * (2 * np.sqrt(2) - 2)
        union = 2 * area_square - area_octagon
        expected_iou = area_octagon / union

        actual = bb_intersection_over_union_rotated(boxA, boxB)
        np.testing.assert_allclose(actual, expected_iou, atol=1e-6)


class TestRotatedBoxCorners(unittest.TestCase):
    def test_axis_aligned_corners(self):
        # angle=0: a w=0.4, h=0.2 box at (0.5, 0.5) has axis-aligned corners.
        corners = rotated_box_corners(0.5, 0.5, 0.4, 0.2, 0.0)
        xs, ys = corners[:, 0], corners[:, 1]
        np.testing.assert_allclose(sorted(xs), [0.3, 0.3, 0.7, 0.7])
        np.testing.assert_allclose(sorted(ys), [0.4, 0.4, 0.6, 0.6])

    def test_area_preserved_under_rotation(self):
        # Shoelace area of the 4 corners must equal w*h for any angle.
        for angle in [-90, -60, -33.3, 0, 17, 45, 89, 90, 175]:
            corners = rotated_box_corners(0.5, 0.5, 0.3, 0.15, angle)
            np.testing.assert_allclose(polygon_area(corners), 0.3 * 0.15, atol=1e-9)

    def test_edge_lengths_match_w_and_h(self):
        # Consecutive corners are separated by the box's side lengths (w then h, alternating).
        w, h, angle = 0.4, 0.2, 37.0
        c = rotated_box_corners(0.5, 0.5, w, h, angle)
        side01 = np.linalg.norm(c[1] - c[0])
        side12 = np.linalg.norm(c[2] - c[1])
        np.testing.assert_allclose(side01, w, atol=1e-9)
        np.testing.assert_allclose(side12, h, atol=1e-9)

    def test_90_degree_rotation_swaps_extent(self):
        # Rotating the width-edge by 90deg makes the bounding extent h wide and w tall.
        w, h = 0.4, 0.2
        c = rotated_box_corners(0.5, 0.5, w, h, 90.0)
        xs, ys = c[:, 0], c[:, 1]
        np.testing.assert_allclose(xs.max() - xs.min(), h, atol=1e-9)
        np.testing.assert_allclose(ys.max() - ys.min(), w, atol=1e-9)

    def test_width_edge_orientation(self):
        # The width-edge direction (corner0 -> corner1) makes `angle` with the x-axis.
        for angle in [0.0, 20.0, 45.0, -30.0]:
            c = rotated_box_corners(0.5, 0.5, 0.4, 0.2, angle)
            edge = c[1] - c[0]
            measured = np.degrees(np.arctan2(edge[1], edge[0]))
            np.testing.assert_allclose(measured, angle, atol=1e-6)

    def test_180_periodicity_same_rectangle(self):
        # A box at angle and angle+180 is the same physical rectangle (same corner set).
        c1 = rotated_box_corners(0.5, 0.5, 0.4, 0.2, 25.0)
        c2 = rotated_box_corners(0.5, 0.5, 0.4, 0.2, 205.0)
        # Same set of points, possibly in a different order.
        s1 = np.array(sorted(map(tuple, np.round(c1, 8))))
        s2 = np.array(sorted(map(tuple, np.round(c2, 8))))
        np.testing.assert_allclose(s1, s2, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
