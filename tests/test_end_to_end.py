import os
import glob
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

def delete_test_models():
    wt2 = data / 'wiki' / 'wikitext-2'
    imdb = data / 'imdb'

    # delete test models from the pretraining step
    for test_file in glob.iglob(f'{str(wt2)}/models/end-to-end-test*'):
        if os.path.isfile(test_file): os.remove(test_file)

    # delete test vocab and model of sentencepiece training
    for test_file in [wt2 / 'models' / 'spm.model', 
                      wt2 / 'models' / 'spm.vocab']:
        if os.path.isfile(test_file): os.remove(test_file)

    # delete test models from the finetuning/classifier training step
    for test_file in glob.iglob(f'{str(imdb)}/models/end-to-end-test*'):
        if os.path.isfile(test_file): os.remove(test_file)


def check_data_exists():
    data = get_data_folder()

    wt2 = data / 'wiki' / 'wikitext-2'
    imdb = data / 'imdb'
    ensure_paths_exists(wt2 / 'en.wiki.train.tokens',
                        imdb / 'train.csv',
                        message="We don't run data preparation"
                                " scripts automatically as it takes ages,"
                                " run prepare_wiki-en.sh & prepare_imdb.sh")
    return imdb, wt2


def test_ulmfit_default_end_to_end():
    """  Test ulmfit with (default) Moses tokenizer on small wikipedia dataset.
    """
    imdb, wt2 = check_data_exists()
    lm_name = 'end-to-end-test-default'
    cuda_id = 0
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

    delete_test_models()


def test_ulmfit_sentencepiece_end_to_end():
    """ Test ulmfit with sentencepiece tokenizer on small wikipedia dataset.
    """
    imdb, wt2 = check_data_exists()
    lm_name = 'end-to-end-test-spm'
    cuda_id = 0
    results = ulmfit.pretrain_lm.pretrain_lm(
        dir_path=wt2,
        lang='en',
        cuda_id=cuda_id,
        qrnn=True,
        subword=True,
        max_vocab=1000,
        bs=80,
        num_epochs=1,
        name=lm_name,
    )

    assert results['accuracy'] > 0.30

    # NOTE: ds_pct is not available for sentencepiece -- tests are on the complete dataset
    #       sentencepiece for finetuning/classification is currently not implemented

    delete_test_models()
