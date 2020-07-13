## COCO benchmark

Here you can find predictions for COCO validation from different freely available pretrained object detection models:
* [EfficientDet](https://github.com/google/automl/tree/master/efficientdet) [[1](https://arxiv.org/abs/1911.09070)]
* [DetectoRS](https://github.com/joe-siyuan-qiao/DetectoRS) [[2](https://arxiv.org/abs/2006.02334)]

| Model | COCO validation mAP(0.5...0.95) |  COCO validation mAP(0.5...0.95) Mirror |
| ------ | --------------- |  --------------- | 
| EffNet-B0 | **33.6** | **33.5** |  
| EffNet-B1 | **39.2** | **39.2** |
| EffNet-B2 | **42.5** | **42.6** |
| EffNet-B3 | **45.9** | **45.5** |
| EffNet-B4 | **49.0** | **48.8** |
| EffNet-B5 | **50.5** | **50.2** |
| EffNet-B6 | **51.3** | **51.1** |
| EffNet-B7 | **52.1** | **51.9** |
| DetectoRS + ResNeXt-101 | **51.5** | **51.5** |
| DetectoRS + Resnet50 | **49.6** | **49.6** |

### Benchmark files

[Download ~287 MB](https://github.com/ZFTurbo/Weighted-Boxes-Fusion/releases/download/v1.0.5/benchmark.zip)

## Ensemble results

There is python code to get high score on COCO validation using WBF method: [run_benchmark.py](run_benchmark.py)

WBF with weights: [0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 5, 5, 7, 7, 9, 9, 8, 8, 5, 5] and IoU = 0.7 gives **55.8** on COCO validation.

``` 
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.558
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.740
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.616
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.399
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.605
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.702
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.404
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.681
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.748
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.619
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.788
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.875
```

## Requirements

numpy, pandas, pycocotools