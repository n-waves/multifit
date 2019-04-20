# ulmfit-multilingual
Repository used for collaboration on application of ulmfit for multiple languages, it helps with pertraining and uses the 
fastai v1 . (The version in n-waves/fastai:ulmfit_multilingual)

# How to train classifier

```
$ LANG=en
$ python -m ulmfit lm --dataset-path data/wiki/${LANG}-100 --tokenizer='f' --nl 3 --name 'orig' --max-vocab 60000 \ 
        --lang ${LANG} --qrnn=False - train 10 --bs=50 --drop_mult=0  --label-smoothing-eps=0.0
...
Model name: data/wiki/en-100/models/f60k/lstm_orig.m
...

$ python -m ulmfit cls --dataset-path data/imdb --base-lm-path  data/wiki/${LANG}-100/models/f60k/lstm_orig.m  \
        --lang=${LANG} --name orig - train 20 --bs 18 --num-cls-epochs=4 --lr_sched=1cycle --label-smoothing-eps=0.1   
```

You can re-evaluate classifiers by running
```
python -m ulmfit eval --glob="imdb/models/*/lstm_*.m"
```

The same command can be used to quickly trian multiple classifiers, by adding the `--name` parameter:
```
python -m ulmfit eval --glob="imdb/models/*/lstm_nl3.m" --name "nl3-my-test1" --num-cls-epochs 4 --label-smoothing-eps=0.1 --lr_sched=1cycle
``` 

To create a tar with model simply run 
```
python -m ulmfit tar data/imdb/models/f60k/lstm_nl3.m
```

## data directory strucutre

Directory structure after changes to the way we process wiki dumps.
```
data
├── imdb
│   ├── aclImdb
│   ├── imdb_lm
│   └── tmp
├── wiki
│   ├── de-100
│   │   └── models
│   ├── de-100-unk
│   │   └── models
│   ├── de-2
│   │   └── models
│   ├── de-2-unk
│   │   └── models
│   ├── de-all
│   │   └── models
│   ├── wikitext-103
│   │   └── models
│   └── wikitext-2
│       └── models
├── wiki_dumps
├── wiki_extr
│   └── de
│       ├── AA
│       ├── AB
...
        └── CC
└── xnli
    ├── XNLI-1.0
    └── XNLI-MT-1.0
        ├── multinli
        └── xnli
```

## how to contribute
We have a fork of fastai to propose changes to fastai.text, with a branch for this project:
 https://github.com/n-waves/fastai/tree/ulmfit_multilingual  

Let us know that you want to start collaboration on fastai forum thread: [Multilingual ULMFIT](https://forums.fast.ai/t/multilingual-ulmfit/28117)
and you will get access to both repositories.

- Follow the [developer installation of fastai](https://github.com/fastai/fastai#developer-install)
- Add n-waves/fastai as additional remote as described here: https://help.github.com/articles/adding-a-remote/

Here is what I did:
```bash
$ cd fastai
$ git remote add n-waves https://github.com/n-waves/fastai.git
$ git remote -v 
n-waves	https://github.com/n-waves/fastai.git (fetch)
n-waves	https://github.com/n-waves/fastai.git (push)
origin	https://github.com/fastai/fastai.git (fetch)
origin	https://github.com/fastai/fastai.git (push)

$ git fetch n-waves
$ git checkout ulmfit_multilingual
Branch 'ulmfit_multilingual' set up to track remote branch 'ulmfit_multilingual' from 'n-waves'.
Switched to a new branch 'ulmfit_multilingual'

$ git push --set-upstream n-waves ulmfit_multilingual  # to automatically push ulmfit_multilingual branch to the n-waves repo
```


## Running tests

To run the tests, the following data is necessary:

- wikitext-2 (prepared by `./prepare_wiki-en.sh`, along with wikitext-103)
- imdb (prepared by `./prepare_imdb.sh`)

then simply run tests, e.g. `pytest .`
