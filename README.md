[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multifit-efficient-multi-lingual-language/cross-lingual-document-classification-on-2)](https://paperswithcode.com/sota/cross-lingual-document-classification-on-2?p=multifit-efficient-multi-lingual-language)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multifit-efficient-multi-lingual-language/cross-lingual-document-classification-on)](https://paperswithcode.com/sota/cross-lingual-document-classification-on?p=multifit-efficient-multi-lingual-language)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multifit-efficient-multi-lingual-language/cross-lingual-document-classification-on-1)](https://paperswithcode.com/sota/cross-lingual-document-classification-on-1?p=multifit-efficient-multi-lingual-language)

# MultiFiT: Efficient Multi-lingual Language Model Fine-tuning
Code to reproduce the paper "[MultiFiT: Efficient Multi-lingual Language Model Fine-tuning](https://arxiv.org/abs/1909.04761)".

Here is a blog post with an introducing to our paper: http://nlp.fast.ai/classification/2019/09/10/multifit.html 

This repository contains a small framework on top of fastai v1.0; the code is compatible with v1.0.47 up to v1.0.59 (the current as of 2019.11.03). 
The results between fastai versions may differ due to optimizations added to fastai. Our models were trained using 1.0.47.

The framework was rewritten to make it easier to use with the newest fastai.

We released 7 language models trained on corresponding Wikipedia dumps:
   - de_multifit_paper_version
   - es_multifit_paper_version
   - fr_multifit_paper_version
   - it_multifit_paper_version
   - ja_multifit_paper_version
   - ru_multifit_paper_version
   - zh_multifit_paper_version
   
To fetch the model just use `multifit.from_pretrained` function. 
Here are some example notebook showing how to train a classifier using a pretrained models.
- [./notebooks/CLS-JA.ipynb](./notebooks/CLS-JA.ipynb) - example of classifier trained on amazon CLS JA music.
- [./notebooks/MLDoc-JA-multifit_fp16.ipynb](./notebooks/MLDoc-JA-multifit_fp16.ipynb) - example of a faster multifit training using fp16 on MDLDoc.

## Results
### MLDoc 
Document classification results on MLDoc dataset [Schwenk and Li, 2018](https://arxiv.org/abs/1805.09821)

| Model          |   de      |   es      |   fr      |   it      |   ja      |   ru      |   zh       |
|----------------|-----------|-----------|-----------|-----------|-----------|-----------|------------|
|LASER           |   92.70   |   88.75   |   90.80   |   85.93   |   85.15   |   84.65   |  88.98     |
| MultiBERT      |   94.0    |   95.15   |   93.20   |   85.82   |   87.48   |   86.85   |  90.72     |
| MultiFiT       | **95.90** | **96.07** | **94.77** | **90.25** | **90.03** |  **87.65**| **92.52**  |

### Amazon CLS
Sentiment classification results on CLS dataset [Prettenhofer and Stein, 2010](https://dl.acm.org/citation.cfm?doid=2036264.2036277)

|          |         DE            |          FR           |               JA     |
|----------|-----------------------|-----------------------|----------------------|
| MultiBERT| 86.05 / 84.90 / 82.00 | 86.15 / 86.90 / 86.65 | 80.87 / 82.83 / 79.95|
| MultiFiT | 93.19 / 90.54 / 93.00 | 91.25 / 89.55 / 93.40 | 86.29 / 85.75 / 86.59|


## How to use it with fastai v1.0
You can use the pretrained models with fastai library as follows:
```
from fastai.text import *
import multifit

exp = multifit.from_pretrained("name of the model")
fa_config =  exp.pretrain_lm.tokenizer.get_fastai_config(add_open_file_processor=True)
data_lm = (TextList.from_folder(imdb_path, **fa_config)
            .filter_by_folder(include=['train', 'test', 'unsup']) 
            .split_by_rand_pct(0.1)
            .label_for_lm()           
            .databunch(bs=bs))
learn = exp.finetune_lm.get_learner(data_lm)  
# learn is a preconfigured fastai learner with a pretrained model loaded
learn.fit_one_cycle(10)
learn.save_encoder("enc")
...
```

## Reproducing the results
This repository is a rewrite of the original training scripts so it lacks all the scripts used in the paper. 
We are working on a port to fastai v2.0 and then we will be adding the scripts that show how to reproduce the results. 
In case you need to use the scripts faster you can access the original scripts [here](https://github.com/n-waves/multifit/tree/ulmfit-multilingual-original-scripts).

## Citation 
```
@article{Eisenschlos2019MultiFit,
  title={MultiFiT: Efﬁcient Multi-lingual Language Model Fine-tuning},
  author={Julian Eisenschlos, Sebastian Ruder, Piotr Czapla, Marcin Kardas, Sylvain Gugger, Jeremy Howard}
  journal={Proceedings of EMNLP-IJCNLP 2019},
  year={2019}
} 
```
