[![DOI](https://zenodo.org/badge/217881799.svg)](https://zenodo.org/badge/latestdoi/217881799)

## Weighted boxes fusion

Repository contains Python implementation of several methods for ensembling boxes from object detection models: 

* Non-maximum Suppression (NMS)
* Soft-NMS [[1]](https://arxiv.org/abs/1704.04503)
* Non-maximum weighted (NMW) [[2]](http://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w14/Zhou_CAD_Scale_Invariant_ICCV_2017_paper.pdf)
* **Weighted boxes fusion (WBF)** [[3]](https://arxiv.org/abs/1910.13302) - new method which gives better results comparing to others 

## Requirements

Python 3.*, Numpy, Numba

# Installation

`pip install ensemble-boxes`

## Usage examples

Coordinates for boxes expected to be normalized e.g in range [0; 1]. Order: x1, y1, x2, y2. 

Example of boxes ensembling for 2 models below. 
* First model predicts 5 boxes, second model predicts 4 boxes.
* Confidence scores for each box model 1: [0.9, 0.8, 0.2, 0.4, 0.7]
* Confidence scores for each box model 2: [0.5, 0.8, 0.7, 0.3]
* Labels (classes) for each box model 1: [0, 1, 0, 1, 1]
* Labels (classes) for each box model 2: [1, 1, 1, 0]
* We set weight for 1st model to be 2, and weight for second model to be 1.
* We set intersection over union for boxes to be match: iou_thr = 0.5
* We skip boxes with confidence lower than skip_box_thr = 0.0001

```python
from ensemble_boxes import *

boxes_list = [[
    [0.00, 0.51, 0.81, 0.91],
    [0.10, 0.31, 0.71, 0.61],
    [0.01, 0.32, 0.83, 0.93],
    [0.02, 0.53, 0.11, 0.94],
    [0.03, 0.24, 0.12, 0.35],
],[
    [0.04, 0.56, 0.84, 0.92],
    [0.12, 0.33, 0.72, 0.64],
    [0.38, 0.66, 0.79, 0.95],
    [0.08, 0.49, 0.21, 0.89],
]]
scores_list = [[0.9, 0.8, 0.2, 0.4, 0.7], [0.5, 0.8, 0.7, 0.3]]
labels_list = [[0, 1, 0, 1, 1], [1, 1, 1, 0]]
weights = [2, 1]

iou_thr = 0.5
skip_box_thr = 0.0001
sigma = 0.1

boxes, scores, labels = nms(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr)
boxes, scores, labels = soft_nms(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, sigma=sigma, thresh=skip_box_thr)
boxes, scores, labels = non_maximum_weighted(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
```

#### Single model

If you need to apply NMS or any other method to single model predictions you can call function like that:

```python
from ensemble_boxes import *
# Merge boxes for single model predictions
boxes, scores, labels = weighted_boxes_fusion([boxes_list], [scores_list], [labels_list], weights=None, method=method, iou_thr=iou_thr, thresh=thresh)
```

More examples can be found in [example.py](examples/example.py)

#### 3D version

There is support for 3D boxes in WBF method with `weighted_boxes_fusion_3d` function. Check example of usage in [example_3d.py](examples/example_3d.py)

#### 1D version

There is support for 1D line segments in WBF method with `weighted_boxes_fusion_1d` function. Check example of usage in [example_1d.py](examples/example_1d.py). It was reported that 1D variant can be useful in Named-entity recognition (NER) type of tasks for Natural Language Processing (NLP) problems. Check discussion [here](https://www.kaggle.com/c/feedback-prize-2021/discussion/313389).

## Benchmarks

* Benchmark for [Open Images Dataset (5 models)](benchmark_oid/README.md)
* Benchmark for [COCO Dataset (10 models)](benchmark_coco/README.md)
* Benchmark for [NLP Dataset (10 models)](benchmark_nlp/README.md) - example for one-dimensional WBF variant

## Description of WBF method and citation

* https://arxiv.org/abs/1910.13302 (updated: 2020.08)
* https://authors.elsevier.com/c/1ca0dxnVK3cWY 

If you find this code useful please cite:

```
@article{solovyev2021weighted,
  title={Weighted boxes fusion: Ensembling boxes from different object detection models},
  author={Solovyev, Roman and Wang, Weimin and Gabruseva, Tatiana},
  journal={Image and Vision Computing},
  pages={1-6},
  year={2021},
  publisher={Elsevier}
}
```
