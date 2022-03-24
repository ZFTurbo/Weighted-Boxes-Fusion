## NLP benchmark

As [it was found out](https://www.kaggle.com/c/feedback-prize-2021/discussion/313389) WBF method in one-dimensional variant works good for NER tasks in NLP problems. Here you can find benchmark which is based on
[Feedback Prize - Evaluating Student Writing Kaggle](https://www.kaggle.com/c/feedback-prize-2021/overview) competition. 
Benchmark consists of OOF predictions of 10 different NLP models for the competition dataset. WBF method allowed achieve 2nd place in competition.
Credits for benchmarks and idea of using WBF for NLP task goes to [Chris Deotte](https://www.kaggle.com/cdeotte) and [Udbhav Bamba](https://www.kaggle.com/ubamba98).   


| Model | Validation score |
| ------ | --------------- | 
| lsg-large | **0.7026** |  
| longformer-lstm | **0.7024** |
| deberta-jaccard | **0.6990** |
| deberta-large-v3| **0.6945** |
| deberta-xlarge-v2 | **0.6955** |
| bird-base-1024 | **0.6746** |
| deberta-large | **0.6950** |
| deberta-xlarge | **0.6984** |
| funnel-large | **0.6880** |
| yoso | **0.6516** |

### Benchmark files

[Download ~16 MB](https://github.com/ZFTurbo/Weighted-Boxes-Fusion/releases/download/v1.0.8/benchmark_nlp.zip)

## Ensemble results

There is python code to get high score on validation using WBF method: [run_benchmark_nlp.py](run_benchmark_nlp.py)

WBF with IoU = 0.33 gives **0.7403** on validation (best model before ensemble gives only 0.7026).
