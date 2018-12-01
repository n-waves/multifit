"""
Train a classifier on top of a language model trained with `pretrain_lm.py`.
Optionally fine-tune LM before.
"""
import fastai
import numpy as np
import pickle

import torch
from fastai.text import TextLMDataBunch, TextClasDataBunch, language_model_learner, text_classifier_learner
from fastai import fit_one_cycle, accuracy
from fastai_contrib.data import LanguageModelType
from fastai_contrib.learner import bilm_text_classifier_learner, bilm_learner, accuracy_fwd, accuracy_bwd
from fastai_contrib.utils import PAD, UNK, read_clas_data, PAD_TOKEN_ID, DATASETS, TRN, VAL, TST, ensure_paths_exists
from fastai.text.transform import Vocab

import fire
from collections import Counter
from pathlib import Path

from ulmfit.pretrain_lm import LMHyperParams


class CLSHyperParams(LMHyperParams):
    # dir_path -> data/imdb/

    def __post_init__(self, *args, **kwargs):
        super().__post_init__(*args, **kwargs)
        self.dataset_dir=self.dataset_path

    @property
    def need_fine_tune_lm(self): return not (self.model_dir/f"enc_best.pth").exists()

    def train_cls(self, num_lm_epochs, unfreeze=True, bs=70):
        data_clas, data_lm = self.load_cls_data(bs)

        if self.need_fine_tune_lm: self.train_lm(num_lm_epochs, data_lm=data_lm)
        learn = self.create_cls_learner(data_clas)

        try:
            learn.load('cls_last')
            print("Loading last classfier")
        except FileNotFoundError:
            learn.load_encoder("enc_best")

        learn.true_wd = False
        print("Starting classifier training")
        learn.fit_one_cycle(1, 5e-2, moms=(0.8, 0.7), wd=1e-7)
        if unfreeze:
            learn.freeze_to(-2)
            learn.fit_one_cycle(1, slice(5e-2 / (2.6 ** 4), 5e-2), moms=(0.8, 0.7), wd=1e-7)

            learn.freeze_to(-3)
            learn.fit_one_cycle(1, slice(5e-4 / (2.6 ** 4), 5e-4), moms=(0.8, 0.7), wd=1e-7)

            learn.unfreeze()
            learn.fit_one_cycle(2, slice(1e-2 / (2.6 ** 4), 1e-2), moms=(0.8, 0.7), wd=1e-7)

        print(f"Saving models at {learn.path / learn.model_dir}")
        learn.save('cls_last', with_opt=False)
        return learn

    def create_cls_learner(self, data_clas):
        fastai.text.learner.default_dropout['language'] = self.dps
        classifier_learner = bilm_text_classifier_learner if self.bidir else text_classifier_learner
        learn = classifier_learner(data_clas, bptt=self.bptt, pad_token=PAD_TOKEN_ID,
            path=self.model_dir.parent, model_dir=self.model_dir.name,
            qrnn=self.qrnn, emb_sz=self.emb_sz, nh=self.nh, nl=self.nl, drop_mult=self.drop_mult)

        learn.metrics = [accuracy_fwd, accuracy_bwd] if self.bidir else [accuracy]
        return learn

    def load_cls_data(self, bs):
        tmp_dir = self.cache_dir
        tmp_dir.mkdir(exist_ok=True)
        vocab_file = tmp_dir / f'vocab_{self.lang}.pkl'
        if not (tmp_dir / f'{TRN}_{self.lang}_ids.npy').exists():
            print('Reading the data...')
            toks, lbls = read_clas_data(self.dataset_dir, self.dataset_dir.name, self.lang)
            # create the vocabulary
            counter = Counter(word for example in toks[TRN] + toks[TST] + toks[VAL] for word in example)
            itos = [word for word, count in counter.most_common(n=self.max_vocab)]
            itos.insert(0, PAD)
            itos.insert(0, UNK)
            vocab = Vocab(itos)
            stoi = vocab.stoi
            with open(vocab_file, 'wb') as f:
                pickle.dump(vocab, f)
            ids = {}
            for split in [TRN, VAL, TST]:
                ids[split] = np.array([([stoi.get(w, stoi[UNK]) for w in s])
                                       for s in toks[split]])
                np.save(tmp_dir / f'{split}_{self.lang}_ids.npy', ids[split])
                np.save(tmp_dir / f'{split}_{self.lang}_lbl.npy', lbls[split])
        else:
            print('Loading the pickled data...')
            ids, lbls = {}, {}
            for split in [TRN, VAL, TST]:
                ids[split] = np.load(tmp_dir / f'{split}_{self.lang}_ids.npy')
                lbls[split] = np.load(tmp_dir / f'{split}_{self.lang}_lbl.npy')
            with open(vocab_file, 'rb') as f:
                vocab = pickle.load(f)
        print(f'Train size: {len(ids[TRN])}. Valid size: {len(ids[VAL])}. '
              f'Test size: {len(ids[TST])}.')
        for split in [TRN, VAL, TST]:
            ids[split] = np.array([np.array(e, dtype=np.int) for e in ids[split]])
            lbls[split] = np.array([np.array(e, dtype=np.int) for e in lbls[split]])
        data_lm = TextLMDataBunch.from_ids(path=tmp_dir, vocab=vocab, train_ids=np.concatenate([ids[TRN], ids[TST]]),
                                           valid_ids=ids[VAL], bs=bs, bptt=self.bptt, lm_type=self.lm_type)
        #  TODO TextClasDataBunch allows tst_ids as input, but not tst_lbls?
        data_clas = TextClasDataBunch.from_ids(
            path=tmp_dir, vocab=vocab, train_ids=ids[TRN], valid_ids=ids[VAL],
            train_lbls=lbls[TRN], valid_lbls=lbls[VAL], bs=bs, classes={l: l for l in lbls[TRN]})

        print(f"Sizes of train_ds {len(data_clas.train_ds)}, valid_ds {len(data_clas.valid_ds)}")
        return data_clas, data_lm

if __name__ == '__main__':
    fire.Fire(CLSHyperParams)
