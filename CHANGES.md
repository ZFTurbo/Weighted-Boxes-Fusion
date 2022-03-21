# Release notes
All notable changes to this project will be documented in this file.

##  v1.0.9
- WBF 1D variant for line segments was added. It was reported that 1D variant can be useful in Named-entity recognition (NER) type of tasks for Natural Language Processing (NLP) problems. Check discussion [here](https://www.kaggle.com/c/feedback-prize-2021/discussion/313389).
- Small comments/syntax fixes, removed unused functions, removed unused numba dependency. Slightly increased speed for default 'avg' method.
- Added new version of WBF, which works faster because of more vectorized structure. It's avoiding cycles and if-statements. Changes are allowed to increase speed 20-30%. Score on test data a little bit decreased from 0.598214 to 0.597297.
To use: ```from ensemble_boxes.ensemble_boxes_wbf_experimental import weighted_boxes_fusion_experimental```

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

