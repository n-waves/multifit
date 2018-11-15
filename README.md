# ulmfit-multilingual
Temporary repository used for collaboration on application of for multiple languages.

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

## Repo structure

- `fastai_contrib`  -- anything that can be ported to fastai once we finish the project like:  NLI models, Sentence Piece tok.,
- `ulmfit`  
    - `data`  -- scripts to fetch and prepare data: wikipedia, xnli, classification data sets  
    - `lm` -- scripts to train language models
    - `bilm` -- scripts to train biLM ELMo style, Bert style
    - `class`  -- scripts to test classifiers on multiple languages
    - `xnli` -- scripts to test nli 
