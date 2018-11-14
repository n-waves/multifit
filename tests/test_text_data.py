import pytest
import fastai.text

from fastai import *
from fastai.text import *

import fastai_contrib.data as contrib_data

def text_df(labels):
    data = []
    texts = ["fast ai is a cool project", "hello world"] * 20
    for ind, text in enumerate(texts):
        sample = {}
        sample["label"] = labels[ind%len(labels)]
        sample["text"] = text
        data.append(sample)
    return pd.DataFrame(data)

###################### UPDATED CODE
def test_should_load_backwards_lm():
    path = untar_data(URLs.IMDB_SAMPLE)
    df = text_df(['neg','pos'])

    data = TextLMDataBunch.from_df(path, train_df=df, valid_df=df, label_cols=0, text_cols=["text"], bs=2,
                                   lm_type=contrib_data.LanguageModelType.BwdLM,
                                   ld_cls=contrib_data.LanguageModelLoader)
    lml = data.train_dl.dl
    lml.data = lml.batchify(np.concatenate([lml.dataset.x.items[i] for i in range(len(lml.dataset))]))
    batch = lml.get_batch(lml.data, 0, 70)

    assert batch[0].shape == (70, lml.bs)
    assert batch[1].shape == (70*lml.bs,)


    as_text = [lml.dataset.vocab.itos[x] for x in batch[0][:,0]]
    np.testing.assert_array_equal(as_text[:5], ["world", "hello", '1', 'xxfld', 'project',])

def test_should_load_bi_lm():
    path = untar_data(URLs.IMDB_SAMPLE)
    df = text_df(['neg', 'pos'])

    data = TextLMDataBunch.from_df(path, train_df=df, valid_df=df, label_cols=0, text_cols=["text"], bs=2,
                                   lm_type=contrib_data.LanguageModelType.BiLM,
                                   ld_cls=contrib_data.LanguageModelLoader)
    lml = data.train_dl.dl
    lml.data = lml.batchify(np.concatenate([lml.dataset.x.items[i] for i in range(len(lml.dataset))]))
    batch = lml.get_batch(lml.data, 0, 70)

    assert batch[0].shape == (70, lml.bs, 2)
    assert batch[1].shape == (70*lml.bs, 2)

    as_text = [lml.dataset.vocab.itos[x] for x in batch[0][:, 0, 0]]
    np.testing.assert_array_equal(as_text[:7], "xxfld 1 fast ai is a cool".split())

    as_text = [lml.dataset.vocab.itos[x] for x in batch[0][:,0,1]]
    np.testing.assert_array_equal(as_text[:5], ["world", "hello", '1', 'xxfld', 'project',])

###################### NEW CODE

