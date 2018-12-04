import os
import glob
import fire
import ulmfit.pretrain_lm
import ulmfit.train_clas
from fastai import *
from fastai.text import *
from fastai_contrib.utils import *

"""
It is a mixture of a pytest unit test and woven together to compose an end to end functional test. 
"""

import fastai.core
fastai.core.defaults.cpus = 1

def copy_head(src_fn, dst_fn, n=1000):
    with src_fn.open("r") as s, dst_fn.open("w") as d:
        for i in range(n):
            d.write(s.readline())


def get_test_data():
    data = get_data_folder()
    wt = data / "wiki" / "wikitext-2"
    imdb = data / "imdb"

    test_data = data / "test"
    shutil.rmtree(test_data)

    test_wt = test_data / 'wikitext-s'
    test_imdb = test_data / 'imdb'
    test_wt.mkdir(exist_ok=True, parents=True)
    test_imdb.mkdir(exist_ok=True, parents=True)

    sz=1
    # we use the same text to see if models can overfit
    copy_head(wt / 'en.wiki.train.tokens', test_wt / 'en.wiki.train.tokens', n=10*sz)
    copy_head(wt / 'en.wiki.train.tokens', test_wt / 'en.wiki.valid.tokens', n=6*sz)
    copy_head(wt / 'en.wiki.train.tokens', test_wt / 'en.wiki.test.tokens', n=6*sz)
    copy_head(imdb / 'train.csv', test_imdb / 'train.csv', n=10*sz)
    copy_head(imdb / 'train.csv', test_imdb / 'test.csv', n=6*sz)

    return test_data, test_wt


def test_ulmfit_works_with_relative_paths():
    """  Test ulmfit with (default) Moses tokenizer on small wikipedia dataset.
    """
    os.chdir(get_data_folder()/"..")


    test_data, wt2 = get_test_data()
    lm_name = 'end-to-end-test-default'
    cuda_id = 0
    exp = ulmfit.pretrain_lm.LMHyperParams(
        dataset_path=wt2.relative_to(Path.cwd()),
        lang='en',
        qrnn=True,
        max_vocab=1000,
        bs=2,
        name=lm_name)

    exp.train_lm(num_epochs=1)

    #assert exp.results['accuracy'] > 0.02

    exp2 = ulmfit.train_clas.CLSHyperParams.from_lm(test_data / 'imdb', exp.model_dir)
    exp2.train_cls(num_lm_epochs=1, unfreeze=False, bs=4,)

    # should work for the second time as well

    exp2 = ulmfit.train_clas.CLSHyperParams.from_lm(test_data / 'imdb', exp.model_dir)
    exp2.train_cls(num_lm_epochs=0, unfreeze=False, bs=4, )


def test_ulmfit_default_end_to_end():
    """  Test ulmfit with (default) Moses tokenizer on small wikipedia dataset.
    """
    test_data, wt2 = get_test_data()
    lm_name = 'end-to-end-test-default'
    cuda_id = 0
    exp = ulmfit.pretrain_lm.LMHyperParams(
        dataset_path=wt2,
        lang='en',
        qrnn=True,
        max_vocab=1000,
        bs=2,
        name=lm_name)

    exp.train_lm(num_epochs=1)

    #assert exp.results['accuracy'] > 0.02

    exp2 = ulmfit.train_clas.CLSHyperParams.from_lm(test_data / 'imdb', exp.model_dir)
    exp2.train_cls(num_lm_epochs=0, unfreeze=False, bs=4,)

def test_ulmfit_fastai_end_to_end():
    """ Test ulmfit with sentencepiece tokenizer on small wikipedia dataset.
    """
    test_data, wt2 = get_test_data()
    lm_name = 'end-to-end-test-fastai'
    cuda_id = 0
    exp = ulmfit.pretrain_lm.LMHyperParams(
        dataset_path=wt2,
        lang='en',
        cuda_id=cuda_id,
        qrnn=True,
        tokenizer='f',
        max_vocab=100,
        bs=2,
        name=lm_name,
    )
    exp.train_lm(num_epochs=1)
    exp2 = ulmfit.train_clas.CLSHyperParams.from_lm(test_data / 'imdb', exp.model_dir)
    exp2.train_cls(num_lm_epochs=0, unfreeze=False, bs=4, )

def test_ulmfit_fastai_bidir_end_to_end():
    """ Test ulmfit with sentencepiece tokenizer on small wikipedia dataset.
    """
    test_data, wt2 = get_test_data()
    lm_name = 'end-to-end-test-fastai'
    cuda_id = 0
    exp = ulmfit.pretrain_lm.LMHyperParams(
        dataset_path=wt2,
        lang='en',
        cuda_id=cuda_id,
        qrnn=True,
        bidir=True,
        tokenizer='f',
        max_vocab=100,
        bs=2,
        name=lm_name,
    )
    exp.train_lm(num_epochs=1)
    exp2 = ulmfit.train_clas.CLSHyperParams.from_lm(str(test_data / 'imdb'), str(exp.model_dir))
    exp2.train_cls(num_lm_epochs=0, unfreeze=False, bs=4, )

def test_ulmfit_moses_fa_bidir_end_to_end():
    """ Test ulmfit with sentencepiece tokenizer on small wikipedia dataset.
    """
    test_data, wt2 = get_test_data()
    lm_name = 'end-to-end-test-fastai'
    cuda_id = 0
    exp = ulmfit.pretrain_lm.LMHyperParams(
        dataset_path=wt2,
        lang='en',
        cuda_id=cuda_id,
        qrnn=True,
        bidir=True,
        tokenizer='vf',
        max_vocab=100,
        bs=2,
        name=lm_name,
    )
    exp.train_lm(num_epochs=1)
    exp2 = ulmfit.train_clas.CLSHyperParams.from_lm(test_data / 'imdb', exp.model_dir)
    exp2.train_cls(num_lm_epochs=0, unfreeze=False, bs=4, )

def test_ulmfit_sentencepiece_end_to_end():
    """ Test ulmfit with sentencepiece tokenizer on small wikipedia dataset.
    """
    test_data, wt2 = get_test_data()
    lm_name = 'end-to-end-test-spm'
    cuda_id = 0
    exp = ulmfit.pretrain_lm.LMHyperParams(
        dataset_path=wt2,
        lang='en',
        cuda_id=cuda_id,
        qrnn=True,
        tokenizer=ulmfit.pretrain_lm.Tokenizers.SUBWORD,
        max_vocab=100,
        bs=2,
        name=lm_name,
    )
    exp.train_lm(num_epochs=1)
    # not supported yet
    # exp2 = ulmfit.train_clas.CLSHyperParams.from_lm(test_data / 'imdb', exp.model_dir)
    # exp2.train_cls(num_lm_epochs=0, unfreeze=False, bs=4, )


if __name__ == "__main__":
    fire.Fire()  # allows using all functions via CLI

