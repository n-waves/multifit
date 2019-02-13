"""
Train a classifier on top of a language model trained with `pretrain_lm.py`.
Optionally fine-tune LM before.
"""
from sacremoses import MosesTokenizer

import fastai
import torch

from fastai import *
from fastai.callbacks import CSVLogger, SaveModelCallback
from fastai.text import *
from fastai_contrib import utils

from fastai_contrib.data import LanguageModelType
from fastai_contrib.learner import bilm_text_classifier_learner, bilm_learner, accuracy_fwd, accuracy_bwd
from fastai_contrib.utils import PAD, UNK, read_clas_data, PAD_TOKEN_ID, DATASETS, TRN, VAL, TST, ensure_paths_exists, \
    get_sentencepiece, MosesTokenizerFunc
from fastai.text.transform import Vocab

import fire
from collections import Counter
from pathlib import Path

from ulmfit.pretrain_lm import LMHyperParams, Tokenizers, ENC_BEST


class CLSHyperParams(LMHyperParams):
    # dir_path -> data/imdb/
    use_test_for_validation=False

    bicls_head:str = 'BiPoolingLinearClassifier'

    def __post_init__(self, *args, **kwargs):
        super().__post_init__(*args, **kwargs)
        self.dataset_dir=self.dataset_path

    @property
    def need_fine_tune_lm(self): return not (self.model_dir/f"enc_best.pth").exists()

    def train_cls(self, num_lm_epochs, unfreeze=True, bs=40, true_wd=True, drop_mul_lm=0.3, drop_mul_cls=0.5,
                   use_test_for_validation=False, num_cls_epochs=2, limit=None, noise=0.0, cls_max_len=20*70):
        assert use_test_for_validation == False, "use_test_for_validation=True is not supported"

        data_clas, data_lm, data_tst = self.load_cls_data(bs, limit=limit, noise=noise)

        if self.need_fine_tune_lm: self.train_lm(num_lm_epochs, data_lm=data_lm, true_wd=true_wd, drop_mult=drop_mul_lm)
        learn = self.create_cls_learner(data_clas, drop_mult=drop_mul_cls, max_len=cls_max_len)
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
                learn.fit_one_cycle(1, slice(5e-3 / (2.6 ** 4), 5e-3), moms=(0.8, 0.7))
                learn.unfreeze()
                learn.fit_one_cycle(num_cls_epochs, slice(1e-3 / (2.6 ** 4), 1e-3), moms=(0.8, 0.7))
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
                learn.fit_one_cycle(num_cls_epochs, slice(1e-2 / (2.6 ** 4), 1e-2), moms=(0.8, 0.7), wd=1e-7)
        print(f"Saving models at {learn.path / learn.model_dir}")
        learn.save('cls_last', with_opt=False)

        self.validate_cls('cls_best', bs=bs, limit=limit, data_tst=data_tst, learn=learn)
        return None

    def validate_cls(self, save_name='cls_last', limit=None, bs=40, data_tst=None, learn=None):
        if data_tst is None:
            _, _, data_tst = self.load_cls_data(bs, limit=limit)
        if learn is None:
            learn = self.create_cls_learner(data_tst, drop_mult=0.3)
            learn.unfreeze()
        learn.load(save_name)
        print(f"Loss and accuracy using ({save_name}):", learn.validate(data_tst.valid_dl))

    def create_cls_learner(self, data_clas, dps=None, **kwargs):
        assert self.bidir == False, "bidirectional model is not yet supported"
        config = dict(emb_sz=self.emb_sz, n_hid=self.nh, n_layers=self.nl, pad_token=PAD_TOKEN_ID, qrnn=self.qrnn, bidir=self.bidir)
        config.update(dps or self.dps)
        trn_args=dict(bptt=self.bptt, clip=self.clip)
        trn_args.update(kwargs)
        learn = text_classifier_learner(data_clas, AWD_LSTM, config=config,
            pretrained=False, path=self.model_dir.parent, model_dir=self.model_dir.name, **trn_args)

        if self.pretrained_model is not None:
            print("Loading pretrained model")
            model_path = untar_data(self.pretrained_model, data=False)
            fnames = [list(model_path.glob(f'*.{ext}'))[0] for ext in ['pth', 'pkl']]
            learn.load_pretrained(*fnames, strict=False)
            learn.freeze()

        learn.callback_fns += [partial(CSVLogger, filename=f"{learn.model_dir}/cls-history"),
                               partial(SaveModelCallback, every='improvement', name='cls_best')]
        return learn

    def load_cls_data(self, bs, **kwargs):
        add_trn_to_lm = True
        lang = self.lang
        use_moses = True
        if 'xnli' in str(self.dataset_dir):
            NotImplementedError("Support for Xnli is not implemented yet")
        if 'imdb' in self.dataset_dir.name:
            add_trn_to_lm = True
        if 'mldoc' in str(self.dataset_dir):
            add_trn_to_lm = False  # False as trn_df is contained in unsup already
            lang = self.lang

        data = self.load_data(lang=lang,
                              add_trn_to_lm=add_trn_to_lm,
                              use_moses=use_moses,
                              **kwargs)
        return self.databunches(bs, **data)

    def load_data(self, lang='', **kwargs):
        prefix = '' if lang == '' else lang+'.'
        trn_df = pd.read_csv(self.dataset_path / f'{prefix}train.csv', header=None)
        tst_df = pd.read_csv(self.dataset_path / f'{prefix}test.csv', header=None)
        val_fn = self.dataset_path / f'{prefix}dev.csv'
        if val_fn.exists():
            print("Loading validation", val_fn)
            val_df = pd.read_csv(val_fn, header=None)
        else:
            val_df = None

        unsup_df = pd.read_csv(self.dataset_path / f'{prefix}unsup.csv', header=None)

        if val_df is None:
            print("Validation set not found using 10% of trn")
            val_len = max(int(len(trn_df) * 0.1), 2)
            trn_len = len(trn_df) - val_len
            trn_df, val_df = trn_df[:trn_len], trn_df[trn_len:]

        kwargs.update(dict(trn_df=trn_df, val_df=val_df, tst_df=tst_df, unsup_df=unsup_df))
        return kwargs

    def databunches(self, bs, trn_df, val_df, tst_df, unsup_df, add_trn_to_lm=True, use_moses=False, force=False, limit=None, noise=0.0):
        lm_trn_df = pd.concat([unsup_df, val_df, tst_df] + ([trn_df] if add_trn_to_lm else []))
        val_len = max(int(len(lm_trn_df) * 0.1), 2)
        lm_trn_df = lm_trn_df[val_len:]
        lm_val_df = lm_trn_df[:val_len]

        args = self.tokenzier_to_fastai_args(sp_data_func=lambda: trn_df[1], use_moses=use_moses)
        try:
            if force: raise FileNotFoundError("Forcing reloading of caches")
            data_lm = TextLMDataBunch.load(self.cache_dir, 'lm', bs=bs)
            print(f"Tokenized data loaded, lm.trn {len(data_lm.train_ds)}, lm.val {len(data_lm.valid_ds)}")
        except FileNotFoundError:
            print(f"Running tokenization...")
            data_lm = TextLMDataBunch.from_df(path=self.cache_dir, train_df=lm_trn_df, valid_df=lm_val_df,
                                              max_vocab=self.max_vocab, bs=bs, **args)
            print(f"Saving tokenized: cls.trn {len(data_lm.train_ds)}, cls.val {len(data_lm.valid_ds)}")
            data_lm.save('lm')

        cls_name="."
        if limit is not None:
            print("Limiting data set to:", limit)
            trn_df = trn_df[:limit]
            val_df = val_df[:limit]
            cls_name=f'{cls_name}limit{limit}'
        if noise > 0.0:
            count = len(trn_df)
            labels = trn_df[0].unique()
            assert np.issubdtype(labels.dtype, np.integer), "noise only works on numerical numbers"
            modulo = labels.max()+1
            idx_to_distrub = np.random.permutation(count)[:int(count * noise)]
            trn_df.loc[idx_to_distrub, [0]] = (trn_df.loc[idx_to_distrub, [0]] + 1) % modulo
            print(f"Added noise to {len(idx_to_distrub)} examples, only {(count-len(idx_to_distrub))/count} have correct labels")
            cls_name = f'{cls_name}noise{noise}'
        try:
            if force: raise FileNotFoundError("Forcing reloading of caches")
            data_cls = TextClasDataBunch.load(self.cache_dir, cls_name, bs=bs)
            print(f"Tokenized data loaded, cls.trn {len(data_cls.train_ds)}, cls.val {len(data_cls.valid_ds)}")
        except FileNotFoundError:
            print(f"Running tokenization...")
            args['vocab'] = data_lm.vocab  # make sure we use the same vocab for classifcation
            data_cls = TextClasDataBunch.from_df(path=self.cache_dir, train_df=trn_df, valid_df=val_df,
                                                 max_vocab=self.max_vocab, bs=bs, **args)
            print(f"Saving tokenized: cls.trn {len(data_cls.train_ds)}, cls.val {len(data_cls.valid_ds)}")
            data_cls.save(cls_name)

        # Hack to load test dataset with labels
        try:
            if force: raise FileNotFoundError("Forcing reloading of caches")
            data_tst = TextClasDataBunch.load(self.cache_dir, 'tst', bs=bs)
        except FileNotFoundError:
            args['vocab'] = data_lm.vocab  # make sure we use the same vocab for classifcation
            data_tst = TextClasDataBunch.from_df(path=self.cache_dir, train_df=val_df, valid_df=tst_df,
                                                 max_vocab=self.max_vocab, bs=bs, **args)
            data_tst.save('tst')

        #$data_cls.test_dl = data_tst.valid_dl
        #data_cls.test_ds = data_tst.valid_ds # AttributeError: can't set attribute
        print('Size of vocabulary:', len(data_lm.vocab.itos))
        print('First 20 words in vocab:', data_lm.vocab.itos[:20])
        return data_cls, data_lm, data_tst

if __name__ == '__main__':
    fire.Fire(CLSHyperParams)

##

