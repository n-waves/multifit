import pathlib
from dataclasses import asdict
from string import Template

import fastai

from fastai import *
from fastai.callbacks import CSVLogger
import fastai.text
from fastai.text import *
import torch
from ulmfit.datasets.utils import read_whitespace_file, \
    validate, UNK
from fastai_contrib.text_data import MosesPreprocessingFunc, get_sentencepiece_fastai, \
    make_data_bunch_from_df
import pickle

from pathlib import Path


def istitle(line):
    return len(re.findall(r'^ ?= [^=]* = ?$', line)) != 0


def read_wiki_articles(filename):
    articles = []

    with open(filename, encoding='utf8') as f:
        lines = f.readlines()
    current_article = []
    for i, line in enumerate(lines):
        current_article.append(line)
        if i < len(lines) - 2 and lines[i + 1].strip() == "" and istitle(lines[i + 2]):
            articles.append("".join(current_article))
            current_article = []
    articles.append("".join(current_article))
    print(f"Wiki text was split to {len(articles)} articles")
    df = pd.DataFrame({'0': np.zeros(len(articles)), 'texts': np.array(articles, dtype=np.object)})
    if len(df.columns) == 1:
        df.insert(0, 'label', 0)
    return df


def read_clas_csv(fn):
    df = pd.read_csv(fn, header=None).fillna("na")
    if len(df.columns) == 1:
        df.insert(0, 'label', 0)
    return df


@dataclass
class Dataset:
    dataset_path: Path
    use_tst_for_lm: bool = True
    noise: float = 0.0
    limit: int = None

    def __post_init__(self):
        self.add_trn_to_lm = True
        self._trn_df = None
        self._tst_df = None
        self._val_df = None

        if 'wiki' in str(self.dataset_path) and len(list(self.dataset_path.glob('*.wiki.*.tokens'))) >= 2:
            self._post_init_tokenized_wiki()
        elif 'reddit' in str(self.dataset_path):
            self._post_init_default_csv(
                lang='en',
                uses_moses=False,
                add_trn_to_lm=True,
                use_lang_as_prefix=False)

        elif 'xnli' in str(self.dataset_path):
            raise NotImplementedError("Support for XNLI is not implemented yet")
        elif 'imdb' in self.dataset_path.name:
            self._post_init_default_csv(
                lang='en',
                uses_moses=True,
                add_trn_to_lm=True,
                use_lang_as_prefix=False)
        elif 'mldoc' in str(self.dataset_path):
            self._post_init_default_csv(
                lang=self._language_from_dataset_path(),
                uses_moses=False,
                add_trn_to_lm=False,
                use_lang_as_prefix=True)
        elif 'hate' in str(self.dataset_path):
            self._post_init_default_csv(
                lang=self._language_from_dataset_path(),
                uses_moses=False,
                add_trn_to_lm=True,
                use_lang_as_prefix=True)
        else:
            raise NotImplementedError(f"Not supported dataset {self.dataset_path}")

    def _post_init_default_csv(self, lang, uses_moses, add_trn_to_lm, use_lang_as_prefix):
        self.lang = lang
        self.uses_moses = uses_moses
        self.add_trn_to_lm = add_trn_to_lm
        self.use_tst_for_lm = False
        self.label_column = 0

        prefix = f"{self.lang}." if use_lang_as_prefix else ""

        self.trn_path = self.dataset_path / f'{prefix}train.csv'
        self.val_path = self.dataset_path / f'{prefix}dev.csv'

        self._read_data = read_clas_csv

        self.trn_path = self.dataset_path / f'{prefix}train.csv'
        self.val_path = self.dataset_path / f'{prefix}dev.csv'
        self.tst_path = self.dataset_path / f'{prefix}test.csv'
        self.unsup_path = self.dataset_path / f'{prefix}unsup.csv'

    def _post_init_tokenized_wiki(self):
        self.uses_moses = True
        self.use_tst_for_lm = False
        self.add_trn_to_lm = True
        self.lang = self._language_from_dataset_path()

        self._read_data = read_wiki_articles

        self.trn_path = self.dataset_path / f'{self.lang}.wiki.train.tokens'
        self.val_path = self.dataset_path / f'{self.lang}.wiki.valid.tokens'
        self.tst_path = self.dataset_path / f'{self.lang}.wiki.test.tokens'
        self.unsup_path = self.dataset_path / f'{self.lang}.wiki.unsup.tokens'

    def _language_from_dataset_path(self):
        lang, size = self.dataset_path.name.split('-')
        return lang

    def _load_n_cache_supervised_data(self):
        if not self._trn_df is not None or not self._tst_df  is not None or not self._val_df is not None:
            trn_df = self._read_data(self.trn_path)
            tst_df = self._read_data(self.tst_path)
            val_df = self._read_data(self.val_path) if self.val_path.exists() else None

            if val_df is None:
                print("Validation set not found using 10% of trn")
                val_len = max(int(len(trn_df) * 0.1), 2)
                trn_len = len(trn_df) - val_len
                trn_df, val_df = trn_df[:trn_len], trn_df[trn_len:]

            self._trn_df, self._val_df, self._tst_df = trn_df, val_df, tst_df

        return self._trn_df, self._val_df, self._tst_df

    def load_supervised_data(self):
        trn_df, val_df, tst_df = self._load_n_cache_supervised_data()
        if self.noise > 0.0 is not None:
            trn_df = self._add_noise(trn_df, self.noise)
            val_df = self._add_noise(val_df, self.noise)

        if self.limit is not None:
            print("Limiting data set to:", self.limit)
            trn_df = trn_df[:self.limit]
            val_df = val_df[:self.limit]

        return trn_df, val_df, tst_df

    def load_unsupervised_data(self):
        trn_df, val_df, tst_df = self._load_n_cache_supervised_data()
        unsup_df = self._read_data(self.unsup_path) if self.unsup_path.exists() else None
        lm_trn_df = pd.concat(
            ([trn_df] if self.add_trn_to_lm else []) +
            ([unsup_df] if unsup_df is not None else []) +
            ([tst_df] if self.use_tst_for_lm else []))

        # val_len = max(int(len(lm_trn_df) * 0.1), 2)
        # lm_trn_df = lm_trn_df[val_len:]
        # lm_val_df = lm_trn_df[:val_len]

        return lm_trn_df, val_df

    def _add_noise(self, trn_df, noise):
        count = len(trn_df)
        labels = trn_df[0].unique()
        assert np.issubdtype(labels.dtype, np.integer), "noise only works on numerical numbers"
        modulo = labels.max() + 1
        idx_to_distrub = np.random.permutation(count)[:int(count * noise)]
        trn_df.loc[idx_to_distrub, [0]] = (np.random.randint(1, modulo - 1, size=len(idx_to_distrub)) +
                                           trn_df.loc[idx_to_distrub][0]) % modulo
        print(
            f"Added noise to {len(idx_to_distrub)} examples, only {(count - len(idx_to_distrub)) / count} have correct labels")
        return trn_df


@dataclass
class ULMFiTDataset(Dataset):
    tokenizer: str = 'f'
    max_vocab: int = 60000

    def __post_init__(self):
        super().__post_init__()
        tokenizer_prefix = f"{self.tokenizer}{self.max_vocab // 1000}k"
        self.cache_path = self.dataset_path / "models" / tokenizer_prefix
        self._vocab = None

    def use_base_model_subword_vocabulary(self, base_lm_path: Path):
        """
            In case of subwoard vocabularies reuse the base model vocabulary during tokenization.
            For word tokenization we still generate new vocabulary for each dataset,
            and we expect finetuning to handle the conversion
        """
        # reuse base model sentencepiece vocabulary
        self.cache_path.mkdir(exist_ok=True, parents=True)
        if base_lm_path and (base_lm_path / '..' / 'spm.vocab').exists() and \
           (base_lm_path.parent.resolve() != self.cache_path.resolve()):
            shutil.copy(str(base_lm_path / '..' / 'itos.pkl'), str(self.cache_path))
            shutil.copy(str(base_lm_path / '..' / 'spm.model'), str(self.cache_path))
            shutil.copy(str(base_lm_path / '..' / 'spm.vocab'), str(self.cache_path))

        # TODO: implement / maybe put the vocabulary md5 to the file names and keep spm models together?
        # sp12k/7599013a8ce538b2e3d4405684221ecaf26bcba1.lm
        # sp12k/7599013a8ce538b2e3d4405684221ecaf26bcba1-vocab.link
        # then we don't need the use_vocabulary, we can just use load_lm_databunch(using_vacab=XXX)
        # we could put spm.model and spm.vocab to the model folder it self then, and copy /link it when we use the orignal model
        # for the time being we can simply compy the spm.model on the right spot and raise an error ir the two are different?

    def load_lm_databunch(self, bs, bptt):
        lm_suffix = bptt if bptt != 70 else ""
        lm_suffix += self.use_tst_for_lm if "" else "-notst"
        data_lm = self._databunch(f"lm{lm_suffix}",
                                  bunch_class=TextLMDataBunch,
                                  data_loader=self.load_unsupervised_data,
                                  bptt=bptt,
                                  bs=bs)

        with (self.cache_path / "itos.pkl").open('wb') as f:
            pickle.dump(data_lm.vocab.itos, f)
        self._vocab = data_lm.vocab

        print('Size of vocabulary:', len(data_lm.vocab.itos))
        print('First 20 words in vocab:', data_lm.vocab.itos[:20])

        return data_lm

    def load_vocab(self):
        if self._vocab is None:
            self._vocab = self.load_lm_databunch(bs=20, bptt=70).vocab
        return self._vocab

    def load_clas_databunch(self, bs):
        vocab = self.load_vocab()

        cls_name = "cls"
        if self.limit is not None:
            cls_name = f'{cls_name}limit{self.limit}'
        if self.noise > 0.0:
            cls_name = f'{cls_name}noise{self.noise}'

        args = dict(vocab=vocab, bunch_class=TextClasDataBunch, bs=bs)
        data_cls = self._databunch(cls_name, data_loader=lambda: self.load_supervised_data()[:2], **args)
        # Hack to load test dataset with labels
        data_tst = self._databunch('tst', data_loader=lambda: self.load_supervised_data()[1:], **args)
        return data_cls, data_tst

    def _databunch(self, name, bunch_class, data_loader, bs, **args):
        bunch_path = self.cache_path / name
        if bunch_path.exists():
            databunch = load_data(self.cache_path, name, bs=bs)
        else:
            print(f"Running tokenization {name}...")
            args.update(**self._get_processor(ds_need_moses=not self.uses_moses)) #TODO depends on the previous model
            train_df, valid_df = data_loader()
            databunch = make_data_bunch_from_df(cls=bunch_class,
                                                path=self.cache_path,
                                                train_df=train_df,
                                                valid_df=valid_df,
                                                max_vocab=self.max_vocab,
                                                mark_fields=True,
                                                text_cols=list(train_df.columns.values)[1:],
                                                **args)
            databunch.save(name)
        print(f"Data {name}, trn: {len(databunch.train_ds)}, val: {len(databunch.valid_ds)}")
        return databunch

    def _get_processor(self, ds_need_moses):
        return {
            'fsp': self._get_processor_sentence_piece,
            'f': self._get_processor_pure_fastai,
            'm': self._get_processor_pure_moses,
            'mf': self._get_processor_moses_fastai,

            'sp': self._get_processor_sentence_piece,  # deprecated
            'v': self._get_processor_pure_moses,  # deprecated
            'vf': self._get_processor_moses_fastai,  # deprecated
        }.get(self.tokenizer)(ds_need_moses)

    def _get_processor_sentence_piece(self, ds_need_moses):
        moses_preproc = [MosesPreprocessingFunc(self.lang)] if ds_need_moses else []
        return get_sentencepiece_fastai(
            cache_dir=self.cache_path,
            vocab_size=self.max_vocab,
            lang=self.lang,
            pre_rules=moses_preproc + defaults.text_pre_rules)

    def _get_processor_pure_moses(self, ds_need_moses):
        moses_preproc = [MosesPreprocessingFunc(self.lang)] if ds_need_moses else []
        return dict(tokenizer=Tokenizer(tok_func=BaseTokenizer,
                                        lang=self.lang,
                                        pre_rules=moses_preproc,
                                        post_rules=[]))

    def _get_processor_moses_fastai(self, ds_need_moses):
        moses_preproc = [MosesPreprocessingFunc(self.lang)] if ds_need_moses else []
        return dict(tokenizer=Tokenizer(tok_func=BaseTokenizer,
                                        lang=self.lang,
                                        pre_rules=moses_preproc + defaults.text_pre_rules,
                                        post_rules=defaults.text_post_rules))

    def _get_processor_pure_fastai(self, ds_need_moses):
        if ds_need_moses:
            warn("Fast ai dont use moses, make sure you trained from wikpiedia that wasm't tokenized with moses.")
        return dict()
