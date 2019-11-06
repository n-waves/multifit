
# MultiFiT: Efficient Multi-lingual Language Model Fine-tuning
Code to reproduce the paper "[MultiFiT: Efficient Multi-lingual Language Model Fine-tuning](https://arxiv.org/abs/1909.04761)".

It is a small framework on top of fastai v1.0; the code is compatible with v1.0.47 up to v1.0.59 (the current as of 2019.11.03). The results between fastai versions may differ due to optimizations added to fast.ai. Our models were trained using 1.0.47.

The framework was rewritten to make it easier to use with the newest fastai; the original code is still available in the ulmfit-multilingual branch.

We released 7 language models trained on corresponding Wikipedia dumps:
   - de_multifit_paper_version
   - es_multifit_paper_version
   - fr_multifit_paper_version
   - it_multifit_paper_version
   - ja_multifit_paper_version
   - ru_multifit_paper_version
   - zh_multifit_paper_version
  
Here is an example notebook that shows how this pretarined models can be used to train model on `cls/ja-music`.

### Fast.ai Integration
You can use the pretrained models with fastai library as follows:
```
exp = ulmfit.from_pretrained("name of the model")
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