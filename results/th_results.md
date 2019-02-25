# Thai Text Classification Benchmark

All codes can be found at [thai2fit](https://github.com/cstorm125/thai2fit/).

## Versions

* Python>=3.6
* PyTorch>=1.0
* fastai>=1.0.38

## [wongnai-corpus](https://github.com/wongnai/wongnai-corpus)

Results are based on evaluation of [Wongnai Challenge: Review Rating Prediction](https://www.kaggle.com/c/wongnai-challenge-review-rating-prediction/leaderboard). Codes can be reproduced on Colab with this [notebook](https://github.com/cstorm125/thai2fit/blob/master/wongnai_cls/classification.ipynb).

| model     | micro_f1_public | micro_f1_private | 
|-----------|-----------------|------------------|
| **ULMFit** | **0.59590**          | **0.59731**           |
| fastText | 0.5145          | 0.5109           |
| LinearSVC | 0.5022          | 0.4976           |
| Kaggle Score | 0.59139          | 0.58139          |
| BERT | 0.56612 | 0.57057 |
