import tempfile

from fastai.text import *
from fastai_contrib.text_data import MosesPreprocessingFunc, \
    make_data_bunch_from_df, SPProcessor2

def read_wiki_articles(filename):
    def istitle(line):
        return len(re.findall(r'^ ?= [^=]* = ?$', line)) != 0
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

    noise: float = 0.0
    limit: int = None

    ds_type: str = None
    lang: str = None
    uses_moses: bool = False
    add_trn_to_lm: bool = True
    use_tst_for_lm: bool = False
    label_column: int = 0
    read_data: Callable = read_clas_csv

    trn_name: Path = 'train.csv'
    val_name: Path = 'dev.csv'
    tst_name: Path = 'test.csv'
    unsup_name: Path = 'unsup.csv'

    def __post_init__(self):
        self.add_trn_to_lm = True
        self._trn_df = None
        self._tst_df = None
        self._val_df = None

        path = str(self.dataset_path)
        if 'wiki' in path and len(list(self.dataset_path.glob('*.wiki.*.tokens'))) >= 2:
            self._post_init_tokenized_wiki()
        elif 'wiki' in path and len(list(self.dataset_path.glob('wiki.*.tokens'))) >= 2:
            self._post_init_tokenized_wiki(wiki103=True)
        elif 'reddit' in path:
            self._post_init_default_csv(
                lang='en',
                uses_moses=False,
                add_trn_to_lm=True,
                use_lang_as_prefix=False)
        elif 'xnli' in path:
            raise NotImplementedError("Support for XNLI is not implemented yet")
        elif 'imdb' in path:
            self._post_init_default_csv(
                lang='en',
                uses_moses=False,
                add_trn_to_lm=True,
                use_lang_as_prefix=False)
        elif 'mldoc' in path:
            self._post_init_default_csv(
                lang=self._language_from_dataset_path(),
                uses_moses=False,
                add_trn_to_lm=False,
                use_lang_as_prefix=True)
        elif 'cls' in path:
            self._post_init_default_csv(
                lang=self._language_from_dataset_path(),
                uses_moses=False,
                add_trn_to_lm=True,
                use_lang_as_prefix=True)
        elif 'hate' in path:
            self._post_init_default_csv(
                lang=self._language_from_dataset_path(),
                uses_moses=False,
                add_trn_to_lm=True,
                use_lang_as_prefix=True)
        else:
            self.read_data = read_clas_csv
            self.trn_path = self.dataset_path / self.trn_name
            self.val_path = self.dataset_path / self.val_name
            self.tst_path = self.dataset_path / self.tst_name
            self.unsup_path = self.dataset_path / self.unsup_name

    def _post_init_default_csv(self, lang, uses_moses, add_trn_to_lm, use_lang_as_prefix):
        self.lang = lang
        self.uses_moses = uses_moses
        self.add_trn_to_lm = add_trn_to_lm
        self.use_tst_for_lm = False
        self.label_column = 0
        self.read_data = read_clas_csv

        prefix = f"{self.lang}." if use_lang_as_prefix else ""
        self.trn_path = self.dataset_path / f'{prefix}train.csv'
        self.val_path = self.dataset_path / f'{prefix}dev.csv'
        self.tst_path = self.dataset_path / f'{prefix}test.csv'
        self.unsup_path = self.dataset_path / f'{prefix}unsup.csv'

    def _post_init_tokenized_wiki(self, wiki103=False):
        self.uses_moses = True
        self.use_tst_for_lm = False
        self.add_trn_to_lm = True
        self.lang = self._language_from_dataset_path()

        self.read_data = read_wiki_articles
        if wiki103:
            prefix=""
        else:
            prefix=f"{self.lang}."

        self.trn_path = self.dataset_path / f'{prefix}wiki.train.tokens'
        self.val_path = self.dataset_path / f'{prefix}wiki.valid.tokens'
        self.tst_path = self.dataset_path / f'{prefix}wiki.test.tokens'
        self.unsup_path = self.dataset_path / f'{prefix}wiki.unsup.tokens'

    def _language_from_dataset_path(self):
        #TODO: Duplicate with training function
        lang, *size = self.dataset_path.name.split('-')
        if lang == "wikitext":
            lang = "en"
        return lang

    def _load_n_cache_supervised_data(self):
        if not self._trn_df is not None or not self._tst_df  is not None or not self._val_df is not None:
            trn_df = self.read_data(self.trn_path)
            tst_df = self.read_data(self.tst_path)
            val_df = self.read_data(self.val_path) if self.val_path.exists() else None

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
        unsup_df = self.read_data(self.unsup_path) if self.unsup_path.exists() else None
        lm_trn_df = pd.concat(
            ([trn_df] if self.add_trn_to_lm else []) +
            ([unsup_df] if unsup_df is not None else []) +
            ([tst_df] if self.use_tst_for_lm else []))

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
    tokenizer: Tokenizer = None
    cache_path: Path = None

    def __post_init__(self):
        super().__post_init__()
        self._vocab = None

    def load_lm_databunch(self, bs, bptt):
        lm_suffix = str(bptt) if bptt != 70 else ""
        lm_suffix += "" if self.use_tst_for_lm else "-notst"
        data_lm = self.load_n_cache_databunch(f"lm{lm_suffix}",
                                              bunch_class=TextLMDataBunch,
                                              data_loader=self.load_unsupervised_data,
                                              bptt=bptt,
                                              bs=bs)

        with (self.cache_path / "itos.pkl").open('wb') as f:
            pickle.dump(data_lm.vocab.itos, f)
        self._vocab = data_lm.vocab

        print('Size of vocabulary:', len(data_lm.vocab.itos))
        print('First 20 words in vocab:', data_lm.vocab.itos[:20])
        data_lm.lang = self.lang
        return data_lm

    def _load_vocab(self):
        if self._vocab is None:
            self._vocab = self.load_lm_databunch(bs=20, bptt=70).vocab
        return self._vocab

    def load_clas_databunch(self, bs):
        vocab = self._load_vocab()

        cls_name = "cls"
        if self.limit is not None:
            cls_name = f'{cls_name}limit{self.limit}'
        if self.noise > 0.0:
            cls_name = f'{cls_name}noise{self.noise}'

        args = dict(vocab=vocab, bunch_class=TextClasDataBunch, bs=bs)
        data_cls = self.load_n_cache_databunch(cls_name, data_loader=lambda: self.load_supervised_data()[:2], **args)
        # Hack to load test dataset with labels
        data_tst = self.load_n_cache_databunch('tst', data_loader=lambda: self.load_supervised_data()[1:], **args)
        data_cls.test_dl = data_tst.valid_dl # data_tst.valid_dl holds test data
        data_cls.lang = self.lang
        return data_cls

    def load_n_cache_databunch(self, name, bunch_class, data_loader, bs, **args):
        bunch_path = self.cache_path / name
        databunch = None
        if bunch_path.exists():
            try:
                databunch = load_data(self.cache_path, name, bs=bs)
            except (AttributeError, ImportError):
                print("Unable  to load data bunch from cache - pickle issue, running processing again.")
        if databunch is None:
            print(f"Running tokenization: '{name}' ...")
            train_df, valid_df = data_loader()
            databunch = self.databunch_from_df(bunch_class, train_df, valid_df, **args)
            databunch.save(name)
        print(f"Data {name}, trn: {len(databunch.train_ds)}, val: {len(databunch.valid_ds)}")
        return databunch

    def databunch_from_df(self, bunch_class, train_df, valid_df, **args):
        args.update(**self.tokenizer.get_fastai_config(dataset_uses_moses=self.uses_moses))  # TODO depends on the previous model
        databunch = make_data_bunch_from_df(cls=bunch_class,
                                            path=self.cache_path,
                                            train_df=train_df,
                                            valid_df=valid_df,
                                            mark_fields=True,
                                            text_cols=list(train_df.columns.values)[1:],
                                            **args)
        return databunch


@dataclass
class ULMFiTTokenizer:
    arch: Any # should be ULMFiTArchitecture, we use Any to avoid circular dependencies between imports
    pretrained_path: Path = None

    def __post_init__(self):
        self.temp_dir = None
        if self.pretrained_path is None:
            self.temp_dir = tempfile.TemporaryDirectory()
            self.pretrained_path = Path(self.temp_dir.name)

    def save(self, new_path: Path, vocab: Vocab = None, learn: Learner = None):
        """
            In case of subwoard vocabularies reuse the base model vocabulary during tokenization.
            For word tokenization we still generate new vocabulary for each dataset,
            and we expect finetuning to handle the conversion
        """

        def copy_sp(path):
            print(f"Copy sp model from {path} to {new_path}")
            shutil.copy(str(path / 'itos.pkl'), str(new_path))
            shutil.copy(str(path / 'spm.model'), str(new_path))
            shutil.copy(str(path / 'spm.vocab'), str(new_path))

        # reuse base model sentencepiece vocabulary
        new_path.mkdir(exist_ok=True, parents=True)
        if self.pretrained_path is None or self.pretrained_path.resolve() == new_path.resolve():
            return
        #
        # if (self.pretrained_path.parent / 'spm.vocab').exists():
        #     copy_sp(self.pretrained_path.parent)

        if (self.pretrained_path / 'spm.vocab').exists():
            copy_sp(self.pretrained_path)

        if learn is not None:
            vocab = learn.data.vocab
        if vocab is not None:
            with (new_path / "itos.pkl").open('wb') as f:
                pickle.dump(vocab.itos, f)

    def get_processor(self, dataset_uses_moses=False):
        return {
            'fsp': self._get_processor_sentence_piece,
            'f': self._get_processor_pure_fastai,
            'm': self._get_processor_pure_moses,
            'mf': self._get_processor_moses_fastai,

            'sp': self._get_processor_sentence_piece,  # deprecated
            'v': self._get_processor_pure_moses,  # deprecated
            'vf': self._get_processor_moses_fastai,  # deprecated
        }.get(self.arch.tokenizer_type)(dataset_uses_moses)

    def get_vocab(self): return Vocab.load(self.pretrained_path / 'itos.pkl')

    def get_fastai_config(self, dataset_uses_moses=False, add_open_file_processor=False):
        processor = self.get_processor(dataset_uses_moses)
        openfile = [OpenFileProcessor()] if add_open_file_processor else []
        return {'processor': openfile + [processor]}

    @property
    def prefix(self):
        return f"{self.arch.tokenizer}{self.arch.max_vocab // 1000}k"

    def _get_processor_sentence_piece(self, ds_uses_moses):
        moses_preproc = [MosesPreprocessingFunc(self.arch.lang)] if not ds_uses_moses else []

        sp_model = self.pretrained_path / 'spm.model'
        if not sp_model.is_file():
            sp_model = None
        sp_vocab = self.pretrained_path / 'spm.vocab'
        if not sp_vocab.is_file():
            sp_vocab = None
        processor = SPProcessor2(
            pre_rules=moses_preproc + defaults.text_pre_rules,
            mark_fields=True,
            vocab_sz=self.arch.max_vocab,
            sp_model=sp_model,
            sp_vocab=sp_vocab,
            lang=self.arch.lang,
            tmp_dir=self.pretrained_path.absolute()  # absolute make sure that dataset path is not added as prefix
        )
        return processor

    def _default_processor(self, fastai_tokenizer=None):
        if fastai_tokenizer is None:
            fastai_tokenizer = Tokenizer(SpacyTokenizer, self.arch.lang)
        return [TokenizeProcessor(tokenizer=fastai_tokenizer), NumericalizeProcessor(max_vocab=self.arch.max_vocab)]

    def _get_processor_pure_moses(self, ds_uses_moses):
        #TODO make sure processor doesnot return openfile
        moses_preproc = [MosesPreprocessingFunc(self.arch.lang)] if not ds_uses_moses else []
        tokenizer = Tokenizer(tok_func=BaseTokenizer,
                              lang=self.arch.lang,
                              pre_rules=moses_preproc,
                              post_rules=[])
        return self._default_processor(tokenizer)

    def _get_processor_moses_fastai(self, ds_uses_moses):
        moses_preproc = [MosesPreprocessingFunc(self.arch.lang)] if not ds_uses_moses else []
        tokenizer = Tokenizer(tok_func=BaseTokenizer,
                              lang=self.arch.lang,
                              pre_rules=moses_preproc + defaults.text_pre_rules,
                              post_rules=defaults.text_post_rules)
        return self._default_processor(tokenizer)

    def _get_processor_pure_fastai(self, ds_uses_moses):
        if not ds_uses_moses:
            warn("Make sure your base model was not pretrained on moses tokenized Wikipedia (default for multifit).")
        tokenizer = Tokenizer(tok_func=SpacyTokenizer, lang=self.arch.lang)
        return self._default_processor(tokenizer)

    def cleanup(self):
        if self.temp_dir is not None:
            self.temp_dir.cleanup()
            self.temp_dir = None

    def __del__(self):
        self.cleanup()

