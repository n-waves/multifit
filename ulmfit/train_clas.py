"""
Train a classifier on top of a language model trained with `pretrain_lm.py`.
Optionally fine-tune LM before.
"""
from sacremoses import MosesTokenizer

import fastai
import numpy as np
import pickle

from fastai import *
from fastai.text import *

import torch
from fastai.text import TextLMDataBunch, TextClasDataBunch, language_model_learner, text_classifier_learner
from fastai import fit_one_cycle, accuracy
from fastai_contrib.data import LanguageModelType
from fastai_contrib.learner import bilm_text_classifier_learner, bilm_learner, accuracy_fwd, accuracy_bwd
from fastai_contrib.utils import PAD, UNK, read_clas_data, PAD_TOKEN_ID, DATASETS, TRN, VAL, TST, ensure_paths_exists, \
    get_sentencepiece
from fastai.text.transform import Vocab


import fire
from collections import Counter
from pathlib import Path

from ulmfit.pretrain_lm import LMHyperParams, Tokenizers, ENC_BEST


class MosesTokenizerFunc(BaseTokenizer):
    "Wrapper around a MosesTokenizer to make it a `BaseTokenizer`."
    def __init__(self, lang:str):
        self.tok = MosesTokenizer(lang)

    def tokenizer(self, t:str) -> List[str]:
        return self.tok.tokenize(t, return_str=False, escape=False)

    def add_special_cases(self, toks:Collection[str]):
        for w in toks:
            assert len(self.tokenizer(w))==1, f"Tokenizer is unable to keep {w} as one token!"

class CLSHyperParams(LMHyperParams):
    # dir_path -> data/imdb/
    use_test_for_validation=False

    def __post_init__(self, *args, **kwargs):
        super().__post_init__(*args, **kwargs)
        self.dataset_dir=self.dataset_path

    @property
    def need_fine_tune_lm(self): return not (self.model_dir/f"enc_best.pth").exists()

    def train_cls(self, num_lm_epochs, unfreeze=True, bs=40, true_wd=True, drop_mul_lm=0.3, drop_mul_cls=0.5):
        data_clas, data_lm = self.load_cls_data(bs)

        if self.need_fine_tune_lm: self.train_lm(num_lm_epochs, data_lm=data_lm, true_wd=true_wd, drop_mult=drop_mul_lm)
        learn = self.create_cls_learner(data_clas, drop_mult=drop_mul_cls)
        try:
            learn.load('cls_last')
            print("Loading last classifier")
        except FileNotFoundError:
            learn.load_encoder(ENC_BEST)
        if true_wd:
            learn.true_wd = True
            print("Starting classifier training")
            learn.freeze_to(-1)
            learn.fit_one_cycle(1, 2e-2, moms=(0.8, 0.7))
            if unfreeze:
                learn.freeze_to(-2)
                learn.fit_one_cycle(1, slice(1e-2 / (2.6 ** 4), 1e-2), moms=(0.8, 0.7))
                learn.freeze_to(-3)
                learn.fit_one_cycle(2, slice(1e-3 / (2.6 ** 4), 1e-3), moms=(0.8, 0.7))
                learn.unfreeze()
                learn.fit_one_cycle(2, slice(1e-3 / (2.6 ** 4), 1e-3), moms=(0.8, 0.7))
        else:
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

    def create_cls_learner(self, data_clas, dps=None, **kwargs):
        fastai.text.learner.default_dropout['language'] = dps or self.dps
        trn_args=dict(drop_mult=self.drop_mult, bptt=self.bptt, clip=self.clip,)
        trn_args.update(kwargs)
        classifier_learner = bilm_text_classifier_learner if self.bidir else text_classifier_learner
        learn = classifier_learner(data_clas,  pad_token=PAD_TOKEN_ID,
            path=self.model_dir.parent, model_dir=self.model_dir.name,
            qrnn=self.qrnn, emb_sz=self.emb_sz, nh=self.nh, nl=self.nl, **trn_args)
        return learn

    def load_cls_data(self, bs, **kwargs):
        if self.dataset_dir.name == 'imdb':
            return self.load_cls_data_imdb(bs, **kwargs)
        else:
            assert self.tokenizer is Tokenizers.MOSES, "XNLI does not support other tokenizers than Moses"
            return self.load_cls_data_old_for_xnli(bs, **kwargs)

    def load_cls_data_imdb(self, bs, force=False, use_test_for_validation=False):
        trn_df = pd.read_csv(self.dataset_path / 'train.csv', header=None)
        tst_df = pd.read_csv(self.dataset_path / 'test.csv', header=None)
        unsp_df = pd.read_csv(self.dataset_path / 'unsup.csv', header=None)

        lm_trn_df = pd.concat([unsp_df, trn_df, tst_df])
        val_len = max(int(len(lm_trn_df) * 0.1), 2)
        lm_trn_df = lm_trn_df[val_len:]
        lm_val_df = lm_trn_df[:val_len]

        if use_test_for_validation:
            val_len = max(int(len(tst_df) * 0.1), 2)
            tst_len = len(tst_df) - val_len
            val_df = trn_df[:tst_len]
        else:
            val_len = max(int(len(trn_df) * 0.1), 2)
            trn_len = len(trn_df) - val_len
            trn_df, val_df = trn_df[:trn_len], trn_df[trn_len:]


        if self.tokenizer is Tokenizers.SUBWORD:
            #TODO Fix me to make sure it trains correct dictionary
            args = get_sentencepiece(self.dataset_path, self.dataset_path / 'train.csv', self.name, vocab_size=self.max_vocab)
        elif self.tokenizer is Tokenizers.MOSES:
            args = dict(tokenizer=Tokenizer(tok_func=MosesTokenizerFunc, lang='en', pre_rules=[], post_rules=[]))
        elif self.tokenizer is Tokenizers.MOSES_FA:
            args = dict(tokenizer=Tokenizer(tok_func=MosesTokenizerFunc, lang='en')) # use default pre/post rules
        elif self.tokenizer is Tokenizers.FASTAI:
            args = dict()
        else:
            raise ValueError(
                f"self.tokenizer has wrong value {self.tokenizer}, Allowed values are taken from {Tokenizers}")

        try:
            if force: raise FileNotFoundError("Forcing reloading of caches")
            data_lm = TextLMDataBunch.load(self.cache_dir, 'lm', lm_type=self.lm_type, bs=bs)
            print(f"Tokenized data loaded, lm.trn {len(data_lm.train_ds)}, lm.val {len(data_lm.valid_ds)}")
        except FileNotFoundError:
            print(f"Running tokenization...")
            data_lm = TextLMDataBunch.from_df(path=self.cache_dir, train_df=lm_trn_df, valid_df=lm_val_df,
                                              max_vocab=self.max_vocab, bs=bs, lm_type=self.lm_type, **args)
            print(f"Saving tokenized: cls.trn {len(data_lm.train_ds)}, cls.val {len(data_lm.valid_ds)}")
            data_lm.save('lm')

        try:
            if force: raise FileNotFoundError("Forcing reloading of caches")
            data_cls = TextClasDataBunch.load(self.cache_dir, '.', bs=bs)
            print(f"Tokenized data loaded, cls.trn {len(data_cls.train_ds)}, cls.val {len(data_cls.valid_ds)}")
        except FileNotFoundError:
            args['vocab'] = data_lm.vocab  # make sure we use the same vocab for classifcation
            print(f"Running tokenization...")
            data_cls = TextClasDataBunch.from_df(path=self.cache_dir, train_df=trn_df, valid_df=val_df,
                                                 test_df=tst_df, max_vocab=self.max_vocab, bs=bs, **args)
            print(f"Saving tokenized: cls.trn {len(data_cls.train_ds)}, cls.val {len(data_cls.valid_ds)}")
            data_cls.save('.')
        print('Size of vocabulary:', len(data_lm.vocab.itos))
        print('First 20 words in vocab:', data_lm.vocab.itos[:20])
        return data_cls, data_lm


    def load_cls_data_old_for_xnli(self, bs):
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
