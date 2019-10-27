## Weighted boxes fusion

Repository contains implementation of method "_Weighted boxes fusion (WBF)_" for ensembling boxes obtained from different object detection models for the same image. This method is good replacement for NMS and Soft-NMS methods, which is widely used now.

## Requirements

Python 3.*, Numpy

# Installation

`pip install ensemble_boxes`

## Usage examples

Coordinates for boxes expected to be normalized e.g in range [0; 1]. Order: x1, y1, x2, y2. 

#### NMS, Soft-NMS, NMW

```
from ensemble_boxes_nms import *
# Ensemble for several models predictions
boxes, scores, labels = nms_method(boxes_list, scores_list, labels_list, method=method, weights=weights, iou_thr=iou_thr, sigma=sigma, thresh=thresh)
# Ensemble for single model predictions
boxes, scores, labels = nms_method([boxes_list], [scores_list], [labels_list], weights=None, method=method, iou_thr=iou_thr, sigma=sigma, thresh=thresh)
```

#### WBF

```
from ensemble_boxes_wbf import *
# Ensemble for several models predictions
boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, method=method, weights=weights, iou_thr=iou_thr, sigma=sigma, thresh=thresh)
# Ensemble for single model predictions
boxes, scores, labels = weighted_boxes_fusion([boxes_list], [scores_list], [labels_list], weights=None, method=method, iou_thr=iou_thr, sigma=sigma, thresh=thresh)
```

More examples can be found in [example.py](./example.py)

## Accuracy and speed comparison

Comparison was made on predictions from 5 different object detection models trained on [Open Images Dataset](https://storage.googleapis.com/openimages/web/index.html) (500 classes).

Model scores at local validation: 
* Model 1: mAP(0.5) 0.5164
* Model 2: mAP(0.5) 0.5019
* Model 3: mAP(0.5) 0.5144
* Model 4: mAP(0.5) 0.5152
* Model 5: mAP(0.5) 0.4910

#### Accuracy

* Best NMS Result: mAP(0.5) 0.5642 (IOU Thr = 0.6)
* Best Soft-NMS Result: mAP(0.5) 0.5616 (Sigma = 0.1, Confidence Thr = 0.001)
* Best NMW Result: mAP(0.5) 0.5667 (IOU Thr = 0.5)
* Best WBF Result: mAP(0.5) 0.5982 (IOU Thr: 0.6) 

#### Speed

* NMS Time: ~50 seconds
* Soft-NMS Time: ~80 seconds
* WBF Time: ~167 seconds

You can download predictions as well as ground truth from here: [test_data.zip]()
Ensemble script for them is available here: [example_oid.py](./example_oid.py)

## Description of WBF method

_Details later_
