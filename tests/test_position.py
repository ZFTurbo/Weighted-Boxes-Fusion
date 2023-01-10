from ensemble_boxes import weighted_boxes_fusion
import numpy as np


class TestWBF:
    def test_one(self):
        thresh = 0.001
        iou_thr = 0.0
        boxes_list = []
        boxes_list.append([0.0, 0.0, 0.4, 0.6])
        boxes_list.append([0.0, 0.0, 0.2, 0.4])
        expected_box = [0.0, 0.0, 0.3, 0.5]
        scores_list = [1.0, 1.0]
        expected_score = 1.0
        labels_list = [3, 3]
        expected_label = 3
        positions_list = []
        positions_list.append([10, 20.0, 30.0])
        positions_list.append([12.0, 22, 32.0])
        expected_positions = [11.0, 21.0, 31.0]
        # checking if ints and floats work well
        print(f"boxes: {boxes_list}")
        print(f"scores: {scores_list}")
        print(f"labels: {labels_list}")
        print(f"positions: {positions_list}")

        # Merge boxes for single model predictions
        boxes, scores, labels, positions = weighted_boxes_fusion(
            [boxes_list], [scores_list], [labels_list], [positions_list], weights=None, iou_thr=iou_thr,
            skip_box_thr=thresh, conf_type='avg', allows_overflow=False)

        errors = []
        # replace assertions by conditions
        if not len(boxes) == 1:
            errors.append(f"incorrect number of results: {len(boxes)}")
        if not np.allclose(boxes[0], expected_box, rtol=1e-05, atol=1e-08, equal_nan=False):
            errors.append(f"Boxes values don't match. Expected: {expected_box} Got: {boxes[0]}")
        if not np.allclose(scores[0], expected_score, rtol=1e-05, atol=1e-08, equal_nan=False):
            errors.append(f"scores values don't match. Expected: {expected_score} Got: {scores[0]}")
        if not np.allclose(labels[0], expected_label, rtol=1e-05, atol=1e-08, equal_nan=False):
            errors.append(f"labels values don't match. Expected: {expected_label} Got: {labels[0]}")
        if not np.allclose(positions[0], expected_positions, rtol=1e-05, atol=1e-08, equal_nan=False):
            errors.append(f"positions values don't match. Expected: {expected_positions} Got: {positions[0]}")

        # assert no error message has been registered, else print messages
        assert not errors, "errors occured:\n{}".format("\n".join(errors))
