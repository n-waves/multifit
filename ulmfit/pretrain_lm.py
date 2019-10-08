"""
Script to train a model on a preprocessed Wiki dataset. Note that the dataset is
expected to have been tokenized with Moses and processed with `postprocess_wikitext.py`.
That is, the data is expected to be white-space separated and numbers are expected
to be split.
"""

import fire

from fastai.callbacks import CSVLogger
from fastai.text import *
from fastai_contrib.utils import read_whitespace_file, \
    validate, UNK, get_sentencepiece, PAD_TOKEN_ID, \
    replace_std_toks, MosesPreprocessingFunc

LM_BEST = "lm_best"
ENC_BEST = "enc_best"


class Tokenizers(Enum):
    SUBWORD='sp'
    BROKENSUBWORD = 'bsp'
    MOSES='v'
    MOSES_FA='vf'
    FASTAI='f'

def istitle(line):
    return len(re.findall(r'^ ?= [^=]* = ?$', line)) != 0

def read_wiki_articles(filename):
    articles = []
    with open(filename, encoding='utf8') as f:
        lines = f.readlines()
    current_article = []
    for i,line in enumerate(lines):
        current_article.append(line)
        if i < len(lines)-2 and lines[i+1].strip() == "" and istitle(lines[i+2]):
            articles.append("".join(current_article))
            current_article = []
    articles.append("".join(current_article))
    print(f"Wiki text was split to {len(articles)} articles")
    return pd.DataFrame({'texts': np.array(articles, dtype=np.object)})

def json_save(f, d):
    with Path(f).open("w") as fp:
        json.dump(d, fp)

def json_load(f):
    with open(f, 'r') as f:
        return json.load(f)

@dataclass
class LMHyperParams:
    dataset_path: Union[str, Path] # data_dir

    base_lm_path: Union[str, Path] = None
    backwards: str = False
    bidir: bool =False
    qrnn: bool = True
    max_vocab: int = 60000
    tokenizer: Tokenizers = Tokenizers.MOSES
    pretrained_model: str = None

    emb_sz:int = 400
    nh: int = None
    nl: int = 3

    # these hyperparameters are for training on ~100M tokens (e.g. WikiText-103)
    # for training on smaller datasets, more dropout is necessary
    dps = dict(output_p=0.25, hidden_p=0.1, input_p=0.2, embed_p=0.02, weight_p=0.15) # consider removing dps & clip from the default hyperparams and put them to train
    clip: float = 0.12
    bptt: int = 70
    # alpha and beta - defaults like in fastai/text/learner.py:RNNLearner()
    rnn_alpha: float = 2  # activation regularization (AR)
    rnn_beta: float = 1  # temporal activation regularization (TAR)

    lang: str = 'en'
    name: str = None
    cuda_id: InitVar[int] = 0

    def __post_init__(self, cuda_id):
        if self.bidir and self.backwards:
            raise ValueError('Both "backwards" and "bidir" options cannot be enabled at the same time')
        if not torch.cuda.is_available():
            print('CUDA not available. Setting device=-1.')
            cuda_id = -1
        torch.cuda.set_device(cuda_id)
        self.dataset_path = Path(self.dataset_path)
        self.base_lm_path = Path(self.base_lm_path) if self.base_lm_path is not None else None
        self.tokenizer = Tokenizers(self.tokenizer) if isinstance(self.tokenizer, str) else self.tokenizer

        assert self.dataset_path.exists()
        self.cache_dir = self.dataset_path / 'models' / self.tokenizer_prefix
        self.model_dir = self.cache_dir / self.model_name

        print('Max vocab:', self.max_vocab)
        print('Cache dir:', self.cache_dir)
        print('Model dir:', self.model_dir)
        if self.nh is None: self.nh = 1550 if self.qrnn else 1150
        if self.name is None: self.name = self.lang

    @property
    def tokenizer_prefix(self): return f"{self.tokenizer.value}{self.max_vocab // 1000}k"

    @property
    def model_direction(self):
        if self.bidir:
            return 'bi'
        if self.backwards:
            return 'bwd'
        else:
            return ''

    @property
    def model_prefix(self): return self.model_direction + ('qrnn' if self.qrnn else 'lstm')

    @property
    def model_name(self): return f"{self.model_prefix}_{self.name}.m"

    @property
    def pretrained_fnames(self): return [self.base_lm_path / LM_BEST, self.base_lm_path / '../itos'] if self.base_lm_path else None

    def tokenizer_to_fastai_args(self, sp_data_func, use_moses):
        moses_preproc = [MosesPreprocessingFunc(self.lang)] if use_moses else []
        if self.tokenizer is Tokenizers.SUBWORD or self.tokenizer is Tokenizers.BROKENSUBWORD:
            if self.base_lm_path and not(self.cache_dir/"spm.model").exists(): # ensure we are using the same sentence piece model
                shutil.copy(self.base_lm_path / '..' / 'itos.pkl', self.cache_dir)
                shutil.copy(self.base_lm_path / '..' / 'spm.model', self.cache_dir)
                shutil.copy(self.base_lm_path / '..' / 'spm.vocab', self.cache_dir)
            args = get_sentencepiece(self.cache_dir,
                                     sp_data_func,
                                     vocab_size=self.max_vocab,
                                     lang=self.lang,
                                     pre_rules=moses_preproc + defaults.text_pre_rules,
                                     post_rules=defaults.text_post_rules)
        elif self.tokenizer is Tokenizers.MOSES:
            args = dict(tokenizer=Tokenizer(tok_func=BaseTokenizer,
                                            lang=self.lang,
                                            pre_rules=moses_preproc + [replace_std_toks],
                                            post_rules=[]))
        elif self.tokenizer is Tokenizers.MOSES_FA:
            args = dict(tokenizer=Tokenizer(tok_func=BaseTokenizer,
                                            lang=self.lang,
                                            pre_rules=moses_preproc + defaults.text_pre_rules,
                                            post_rules=defaults.text_post_rules))
        elif self.tokenizer is Tokenizers.FASTAI:
            args = dict()
        else:
            raise ValueError(
                f"self.tokenizer has wrong value {self.tokenizer}, Allowed values are taken from {Tokenizers}")
        return args

    def save_info(self):
        from dataclasses import asdict
        vals = {k: (str(v) if isinstance(v, Path) else v) for k,v in asdict(self).items()}
        vals.pop('name', None)
        vals.pop('lang', None)
        vals['tokenizer'] = self.tokenizer.value
        json_save(self.model_dir/'info.json', vals)
        print("Saving info", self.model_dir / 'info.json')

    def train_lm(self, num_epochs=20, data_lm=None, bs=70, true_wd=False, drop_mult=0.0, lr=5e-3, label_smoothing_eps=0.0):
        self.model_dir.mkdir(exist_ok=True, parents=True)
        data_lm = self.load_wiki_data(bs=bs) if data_lm is None else data_lm
        learn = self.create_lm_learner(data_lm, drop_mult=drop_mult, label_smoothing_eps=label_smoothing_eps)
        print("Bptt", data_lm.bptt)
        learn.true_wd = true_wd
        if num_epochs > 0:
            if self.pretrained_fnames or self.pretrained_model:
                print("Training lm from: ", self.pretrained_fnames or self.pretrained_model)
                if learn.true_wd:
                    learn.freeze_to(-1)
                    learn.fit_one_cycle(1, 1e-2, moms=(0.8, 0.7))
                    learn.unfreeze()
                    learn.fit_one_cycle(num_epochs, 1e-3, moms=(0.8, 0.7))
                else:
                    learn.freeze_to(-1)
                    learn.fit_one_cycle(1, 1e-2, moms=(0.8, 0.7), wd=1e-7)  # TODO Fix the learning rates
                    learn.unfreeze()
                    learn.fit_one_cycle(num_epochs, 1e-3, moms=(0.8, 0.7), wd=1e-7)
            else:
                print("Training lm from random weights")
                learn.unfreeze()
                if not learn.true_wd: learn.fit_one_cycle(num_epochs, lr, (0.8, 0.7), wd=1e-7)
                else:                 learn.fit_one_cycle(num_epochs, lr, (0.8, 0.7)) # TODO find proper values
        learn.save("lm_best_with_opt", with_opt=True)
        learn.save_encoder(ENC_BEST)
        learn.save(LM_BEST, with_opt=False)
        print(learn.path)

        self.save_info()
        # do we need to return `learn'? it adds noise to Fire output
        #return learn

    def create_lm_learner(self, data_lm, dps=None, label_smoothing_eps=0.0, **kwargs):
        assert self.bidir == False, "bidirectional model is not yet supported"
        config = dict(emb_sz=self.emb_sz, n_hid=self.nh, n_layers=self.nl, pad_token=PAD_TOKEN_ID, qrnn=self.qrnn,
                          tie_weights=True, out_bias=True)
        config.update(dps or self.dps)
        trn_args = dict(clip=self.clip, alpha=self.rnn_alpha, beta=self.rnn_beta)
        trn_args.update(kwargs)
        print ("Training args: ", trn_args, "dps: ", dps or self.dps)
        learn = language_model_learner(data_lm, AWD_LSTM, config=config, model_dir=self.model_dir.relative_to(data_lm.path), pretrained=False, **trn_args)
        if self.pretrained_model is not None:
            print("Loading pretrained model")
            model_path = untar_data(self.pretrained_model, data=False)
            fnames = [list(model_path.glob(f'*.{ext}'))[0] for ext in ['pth', 'pkl']]
            learn.load_pretrained(*fnames)
            learn.freeze()
        if self.pretrained_fnames is not None:
            print("Loading pretrained model")
            fnames = [f'{fn}.{ext}' for fn,ext in zip(self.pretrained_fnames, ['pth', 'pkl'])]
            learn.load_pretrained(*fnames)
            learn.freeze()
        # compared to standard Adam, we set beta_1 to 0.8
        learn.opt_fn = partial(optim.Adam, betas=(0.8, 0.99))
        learn.callback_fns += [partial(CSVLogger, filename=f"{learn.model_dir}/lm-history"),
                               # partial(SaveModelCallback, every='improvement', name='lm') disabled due to Memory issues
                               ]
        if label_smoothing_eps > 0.0:
            learn.loss_func = FlattenedLoss(LabelSmoothingCrossEntropy, eps=label_smoothing_eps)
        return learn

    def load_train_text(self):
        trn_path = self.dataset_path / f'{self.lang}.wiki.train.tokens'
        with open(trn_path) as f:
            return [line.rstrip('\n') for line in f]

    def load_wiki_data(self, bs=70):
        self.model_dir.mkdir(exist_ok=True, parents=True)
        trn_path = self.dataset_path / f'{self.lang}.wiki.train.tokens'
        val_path = self.dataset_path / f'{self.lang}.wiki.valid.tokens'
        tst_path = self.dataset_path / f'{self.lang}.wiki.test.tokens'
        for path_ in [trn_path, val_path, tst_path]:
            assert path_.exists(), f'Error: {path_} does not exist.'

        args = self.tokenizer_to_fastai_args(sp_data_func=self.load_train_text, use_moses=False)

        data_lm = self.lm_databunch(f"lm{self.bptt if self.bptt != 70 else ''}",
                          train_df=read_wiki_articles(trn_path),
                          valid_df=read_wiki_articles(val_path),
                          classes=None,
                          bs=bs,
                          text_cols='texts',
                          bptt=self.bptt,
                          **args)

        itos, stoi, trn_path = data_lm.vocab.itos, data_lm.vocab.stoi, data_lm.path
        print('Size of vocabulary:', len(itos))
        print('First 20 words in vocab:', data_lm.vocab.itos[:20])
        return data_lm

    def lm_databunch(self, name, *args, **kwargs):
        return self.databunch(name, bunch_class=TextLMDataBunch, *args, **kwargs)

    def databunch(self, name, bunch_class, train_df, valid_df, bs, force=False, **args):
        bunch_path = self.cache_dir / name
        if force and bunch_path.exist():
            print("Forcefully recreating the databunch, removing previously stored data")
            for f in bunch_path.glob("*.npy"):
                f.unlink()
            if bunch_path.isdir():
                if name != ".":
                    bunch_path.rmdir()
            else:
                bunch_path.unlink()

        if (bunch_path / 'itos.pkl').exists():
            data = bunch_class.load(self.cache_dir, name, bs=bs)
        elif bunch_path.exists():
            data = load_data(self.cache_dir, file=name, bs=bs)
        else:
            print(f"Running tokenization {name}...")
            data = bunch_class.from_df(path=self.cache_dir,
                                       train_df=train_df,
                                       valid_df=valid_df,
                                       max_vocab=self.max_vocab,
                                       bs=bs,
                                       **args)
            data.save(name)
        with open(self.cache_dir/"itos.pkl", 'wb') as f:
            pickle.dump(data.vocab.itos, f)


        print(f"Data {name}, trn: {len(data.train_ds)}, val: {len(data.valid_ds)}")
        return data

    @classmethod
    def from_lm(cls, dataset_path, base_lm_path, **kwargs) -> 'LMHyperParams':
        dataset_path = Path(dataset_path).resolve()
        base_lm_path = Path(base_lm_path).resolve()
        d = json_load(base_lm_path/'info.json')
        d['dataset_path'] = dataset_path
        d['base_lm_path'] = base_lm_path
        d.pop('bs', None)
        d.pop('drop_mult', None)
        subword = d.pop('subword', False)
        tokenizer = d.pop('tokenizer', None)
        if tokenizer is not None:
            d['tokenizer'] = Tokenizers(tokenizer)
        elif subword:
            d['tokenizer'] = Tokenizers.SUBWORD
        else:
            d['tokenizer'] = Tokenizers.MOSES

        d.update(kwargs)
        return cls(**d)
    @classmethod
    def from_json(cls, model_path:Path, **kwargs):
        model_path = Path(model_path).resolve()
        name = re.search(r"[a-z]+_(.+).m", model_path.name).group(1)
        with open(model_path / 'info.json', 'r') as f:
            d = json.load(f)
        d.update(kwargs)
        d['name'] = name
        dataset_path = path_strip(model_path, "data", "models").parent
        d['dataset_path'] = str(dataset_path)
        d['lang'] = infer_lang_from_dataset(dataset_path.name)
        return cls(**d)

def infer_lang_from_dataset(name:str):
    return name.split("-")[0]

def path_strip(path, from_folder, to_folder):
    to_p = [p for p in path.parents if p.name == to_folder][0]
    from_p = [p for p in path.parents if p.name == from_folder][0]
    return to_p.relative_to(from_p.parent)

def validate_lm(self):
    if not self.exp.subword and self.exp.max_vocab is None:
        raise NotImplementedError("figure out how to validate and save results")
        # only if we use the unpreprocessed version and the full vocabulary
        # are the perplexity results comparable to previous work
        print(f"Validating model performance with test tokens from: {trn_path}")
        tst_tok = read_whitespace_file(trn_path)
        tst_ids = np.array([([stoi.get(w, stoi[UNK]) for w in s]) for s in tst_tok])
        logloss, perplexity = validate(learn.model, tst_ids, self.exp.bptt)
        print('Test logloss:', logloss.item(), 'perplexity:', perplexity.item())

if __name__ == '__main__':
    fire.Fire(LMHyperParams)
