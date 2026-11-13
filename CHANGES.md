# Release notes
All notable changes to this project will be documented in this file.

## Changes since v1.0.9

### Added

- Added `weighted_boxes_fusion_rotated` for oriented boxes in `(cx, cy, w, h, angle)` format, with angles in degrees using the long-edge convention. Normalize all four spatial values by the same scale. The implementation aligns near-square boxes before averaging their angles and supports normalized edge lengths up to `sqrt(2)`. [#69](https://github.com/ZFTurbo/Weighted-Boxes-Fusion/pull/69)
- Added `weighted_boxes_fusion_quadrangle` for boxes defined by four vertices in any order. It aligns the vertices to the highest-scoring box in each cluster before averaging. [#70](https://github.com/ZFTurbo/Weighted-Boxes-Fusion/pull/70)
- Added Jupyter notebooks and sample aerial imagery, annotations, and predictions for rotated and quadrangle WBF. [#72](https://github.com/ZFTurbo/Weighted-Boxes-Fusion/pull/72)

### Fixed

- NMS validates prediction lengths per model and skips models with empty or mismatched predictions. [`8ab5085`](https://github.com/ZFTurbo/Weighted-Boxes-Fusion/commit/8ab50856cf1c8be9feb9f472bdf184ad6627398f)
- `nms`, `soft_nms`, and `nms_method` return empty arrays of shapes `(0, 4)`, `(0,)`, and `(0,)` when no boxes remain after input filtering or zero-area removal. [#76](https://github.com/ZFTurbo/Weighted-Boxes-Fusion/pull/76)
- Weighted NMS calls preserve the caller's score list, so repeated calls on the same predictions apply model weights once per call. [#77](https://github.com/ZFTurbo/Weighted-Boxes-Fusion/pull/77)
- Linear and Gaussian Soft-NMS return the decayed confidence scores for retained boxes. This can change detection rankings and downstream metrics. The published COCO Soft-NMS benchmark predates this fix. [#78](https://github.com/ZFTurbo/Weighted-Boxes-Fusion/pull/78)

### Performance

- Skip polygon clipping for rotated and quadrangle boxes that cannot overlap. Rotated WBF checks circumscribed circles, and quadrangle WBF checks axis-aligned bounds. [#71](https://github.com/ZFTurbo/Weighted-Boxes-Fusion/pull/71)
- Speed up polygon clipping shared by rotated and quadrangle WBF by reusing vertex inside tests and clip-edge differences. [#74](https://github.com/ZFTurbo/Weighted-Boxes-Fusion/pull/74)
- Reduce repeated array access and temporary allocations when updating axis-aligned WBF clusters. Confidence scores are collected only in `max` mode. [#75](https://github.com/ZFTurbo/Weighted-Boxes-Fusion/pull/75)

### Documentation and development

- Document notable adaptations and implementations of WBF in `docs/notable_adoptions.md` and link to it from the README. [#66](https://github.com/ZFTurbo/Weighted-Boxes-Fusion/pull/66)
- Add pinned development dependencies, a virtual environment setup script, and README instructions for running the test suite. [#67](https://github.com/ZFTurbo/Weighted-Boxes-Fusion/pull/67)
- Update the development pytest pin from 8.4.2 to 9.0.3. [#68](https://github.com/ZFTurbo/Weighted-Boxes-Fusion/pull/68)

##  v1.0.9
- WBF 1D variant for line segments was added. It was reported that 1D variant can be useful in Named-entity recognition (NER) type of tasks for Natural Language Processing (NLP) problems. Check discussion [here](https://www.kaggle.com/c/feedback-prize-2021/discussion/313389).
- Small comments/syntax fixes, removed unused functions, removed unused numba dependency. Slightly increased speed for default 'avg' method.
- Added new version of WBF, which works faster because of more vectorized structure. It's avoiding cycles and if-statements. Changes are allowed to increase speed 20-30%. Score on test data a little bit decreased from 0.598214 to 0.597297.
To use: ```from ensemble_boxes.ensemble_boxes_wbf_experimental import weighted_boxes_fusion_experimental```
- NLP benchmark was added. It's used for WBF 1D variant. Credits for benchmarks and idea of using WBF for NLP task goes to [Chris Deotte](https://www.kaggle.com/cdeotte) and [Udbhav Bamba](https://www.kaggle.com/ubamba98).
- Benchmarks moved into separate folders (benchmark_oid, benchmark_coco, benchmark_nlp)
##  v1.0.8
- Speed up of find_matching_box function. See details [here](https://github.com/ZFTurbo/Weighted-Boxes-Fusion/pull/48).
  OID benchmark: 285 sec -> 242 sec. COCO benchmark: 1055 sec -> 643 sec

##  v1.0.7
- Fixed incorrect values after WBF for allows_overflow = False mode. See details [here](https://github.com/ZFTurbo/Weighted-Boxes-Fusion/pull/41).
- Fixed incorrect values after WBF for conf_type = 'max' mode. See details [here](https://github.com/ZFTurbo/Weighted-Boxes-Fusion/pull/42).
##  v1.0.6
- Added 2 new methods for average boxes: '_box_and_model_avg_' and '_absent_model_aware_avg_'. See details [here](https://github.com/ZFTurbo/Weighted-Boxes-Fusion/pull/25).
Both methods fix issue with confidences larger than 1 after ensemble. Also it gives better results for cases when there are more than 1 box goes to cluster from same model.
One of these methods will replace default 'avg' in later releases after proper testing. Thanks [@i-aki-y](https://github.com/i-aki-y) for great PR.
- Added first version of unit tests.
##  v1.0.5
- Added [benchmark](https://github.com/ZFTurbo/Weighted-Boxes-Fusion/tree/master/benchmark) files

##  v1.0.4
- Added many input data checks in all methods: set to 0.0 coords < 0, set to 1.0 coords > 1, swap cases where x2 < x1 and y2 < y1, remove zero area boxes.
- Added numba @jit(nopython=True) for critical functions (NMS, NMW and WBF). Speed up around x2 times (tested on example_oid.py).
- Added support for 3D boxes with function weighted_boxes_fusion_3d
##  v1.0.1
- Fixed bug with incorrect work of WBF and NMW algorithms if provided boxes was unsorted and skip_box_thr was larger than 0.0.
