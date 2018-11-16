import fire
import pytest

import ulmfit.pretrain_lm
import ulmfit.train_clas
from fastai import *
from fastai.text import *
from fastai_contrib.utils import  *
"""
It is a mixture of a pytest unit test and woven together to compose an end to end functional test. 
"""

def check_data_exists():
    data = get_data_folder()

    wt2 = data / "wiki" / "wikitext-2"
    imdb = data / "imdb"
    ensure_paths_exists(wt2 / "en.wiki.train.tokens",
                        imdb / "train.csv",
                        message="We don't run data preparation scripts automatically as it takes ages, run prepare_wiki-en.sh & prepare_imdb.sh")
    return imdb, wt2

def test_pretrain_lm():
    imdb,wt2 = check_data_exists()
    lm_name="end-to-end-test-quick"
    cuda_id=0
    results = ulmfit.pretrain_lm.pretrain_lm(
        dir_path=wt2,
        lang='en',
        cuda_id=cuda_id,
        qrnn=True,
        subword=False,
        max_vocab=1000,
        bs=80,
        num_epochs=1,
        name=lm_name,
        ds_pct=0.03
    )
    assert results['accuracy'] > 0.30

    results = ulmfit.train_clas.new_train_clas(
                data_dir=get_data_folder(),
                lang='en', pretrain_name=lm_name, model_dir=wt2/'models',
                qrnn=True,
                cuda_id=cuda_id,
                fine_tune=True,
                max_vocab=1000,
                bs=20, bptt=70, name=lm_name+'-imdb-clas',
                dataset='imdb',
                ds_pct=0.03)
