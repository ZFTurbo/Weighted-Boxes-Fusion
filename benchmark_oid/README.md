## OID benchmark

Comparison was made for ensemble of 5 different object detection models predictions trained on [Open Images Dataset](https://storage.googleapis.com/openimages/web/index.html) (500 classes).

Model scores at local validation: 
* Model 1: mAP(0.5) 0.5164
* Model 2: mAP(0.5) 0.5019
* Model 3: mAP(0.5) 0.5144
* Model 4: mAP(0.5) 0.5152
* Model 5: mAP(0.5) 0.4910

| Method | mAP(0.5) Result | Best params | Elapsed time (sec) | 
| ------ | --------------- | ----------- | ------------ |
| NMS | **0.5642** | IOU Thr: 0.5 | 47 |
| Soft-NMS | **0.5616** | Sigma: 0.1, Confidence Thr: 0.001 | 88 |
| NMW | **0.5667** | IOU Thr: 0.5 | 171 |
| WBF | **0.5982** | IOU Thr: 0.6 | 249 |

You can download model predictions as well as ground truth labels from here: [test_data.zip](https://github.com/ZFTurbo/Weighted-Boxes-Fusion/releases/download/v1.0/test_data.zip)

Ensemble script for them is available here: [run_benchmark_oid.py](run_benchmark_oid.py)