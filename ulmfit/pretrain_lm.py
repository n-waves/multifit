"""
Script to train a model on a preprocessed Wiki dataset. Note that the dataset is
expected to have been tokenized with Moses and processed with `postprocess_wikitext.py`.
That is, the data is expected to be white-space separated and numbers are expected
to be split.
"""
from dataclasses import InitVar

import fastai
import fire

from fastai import *
from fastai.text import *
import torch
from fastai_contrib.utils import read_file, read_whitespace_file, \
    validate, PAD, UNK, get_sentencepiece, read_clas_data, TRN, VAL, TST, PAD_TOKEN_ID
from fastai_contrib.learner import bilm_learner, accuracy_fwd, accuracy_bwd, bilm_text_classifier_learner
import pickle

from pathlib import Path

from collections import Counter
import fastai_contrib.data as contrib_data

# to install, do:
# conda install -c pytorch -c fastai fastai pytorch-nightly [cuda92]
# cupy needs to be installed for QRNN


# """
# :param dir_path: The path to the directory of the file.
# :param lang: the language unicode
# :param cuda_id: The id of the GPU. Uses GPU 0 by default or no GPU when
#                 run on CPU.
# :param qrnn: Use a QRNN. Requires installing cupy.
# :param subword: Use sub-word tokenization on the cleaned data.
# :param max_vocab: The maximum size of the vocabulary.
# :param bs: The batch size.
# :param bptt: The back-propagation-through-time sequence length.
# :param name: The name used for both the model and the vocabulary.
# :param model_dir: The path to the directory where the models should be saved
# :param bidir: whether the language model is bidirectional
# """


@dataclass
class LMHyperParams:
    dataset_path: str # data_dir

    base_lm_path: str = None
    bidir: bool =False
    qrnn: bool = True
    max_vocab: int = 60000
    subword: bool = False

    emb_sz:int = 400
    nh: int = None
    nl: int = 3

    # these hyperparameters are for training on ~100M tokens (e.g. WikiText-103)
    # for training on smaller datasets, more dropout is necessary
    drop_mult = 0.1
    dps = [0.25, 0.1, 0.2, 0.02, 0.15]
    clip: float = 0.12
    bptt: int = 70
    bs: int = 70

    lang: str = 'en'
    name: str = ''
    cuda_id: InitVar[int] = 0

    def __post_init__(self, cuda_id):
        if not torch.cuda.is_available():
            print('CUDA not available. Setting device=-1.')
            cuda_id = -1
        torch.cuda.set_device(cuda_id)
        self.dataset_path = Path(self.dataset_path)
        self.base_lm_path = Path(self.base_lm_path) if self.base_lm_path is not None else None

        assert self.dataset_path.exists()
        self.cache_dir = self.dataset_path / 'models' / self.tok_name
        self.model_dir = self.cache_dir / self.full_name

        self.model_dir.mkdir(exist_ok=True, parents=True)
        print('Batch size:', self.bs)
        print('Max vocab:', self.max_vocab)
        print('Cache dir:', self.cache_dir)
        print('Model dir:', self.model_dir)
        self.dps = np.array(self.dps)
        if self.nh is None: self.nh = 1550 if self.qrnn else 1150

    @classmethod
    def based_on(cls, base_lm_path, dataset_path, **kwargs) -> 'LMHyperParams':
        with open(base_lm_path/'info.json', 'r') as f: d = json.load(f)
        d['dataset_path'] = dataset_path
        d['base_lm_path'] = base_lm_path
        d.update(kwargs)
        return cls(**d)

    def save_info(self):
        from dataclasses import asdict
        vals = {k: (str(v) if isinstance(v, Path) else v) for k,v in asdict(self).items()}
        vals.pop('name', None)
        vals.pop('lang', None)
        with (self.model_dir / 'info.json').open("w") as fp: json.dump(vals, fp)
        print("Saving info", self.model_dir / 'info.json')

    @property
    def tok_name(self):
        pref = 'sp' if self.subword else 'v'
        voc_size = self.max_vocab // 1000
        return f"{pref}{voc_size}k"

    @property
    def full_name(self): return f"{self.model_name}_{self.name}.m"

    # todo rework
    @property
    def model_name(self): return ('bi' if self.bidir else '') + ('qrnn' if self.qrnn else 'lstm')

    @property
    def pretrained_fnames(self): return [self.base_lm_path / 'lm_best', self.base_lm_path / '../itos'] if self.base_lm_path else None

    def train_lm(self, num_lm_epochs=10, data_lm=None):
        data_lm = self.load_wiki_data() if data_lm is None else data_lm
        learn = self.create_lm_learner(data_lm)

        if num_lm_epochs > 0:
            if self.pretrained_fnames :
                learn.fit_one_cycle(1, 1e-2, moms=(0.8, 0.7))
                learn.unfreeze()
                if num_lm_epochs > 0: learn.fit_one_cycle(num_lm_epochs, 1e-3, moms=(0.8, 0.7))
            else:
                try:
                    learn.load("lm_best")
                    print("Weights loaded")
                except FileNotFoundError:
                    print("Starting from random weights")
                learn.fit_one_cycle(num_lm_epochs, 5e-3, (0.8, 0.7), wd=1e-7)
            opt_state_path = self.model_dir / 'opt_state.pth'
            print(f"Saving optimiser state at {opt_state_path}")
            torch.save(learn.opt.opt.state_dict(), opt_state_path)
        learn.save_encoder("enc_best")
        learn.save("lm_best", with_opt=False)
        print(learn.path)

        self.save_info()
        return learn

    def create_lm_learner(self, data_lm):
        fastai.text.learner.default_dropout['language'] = self.dps
        lm_learner = bilm_learner if self.bidir else language_model_learner

        learn = lm_learner(data_lm, bptt=self.bptt, emb_sz=self.emb_sz, nh=self.nh, nl=self.nl, pad_token=PAD_TOKEN_ID,
                           drop_mult=self.drop_mult, tie_weights=True, model_dir= self.model_dir.relative_to(data_lm.path),
                           bias=True, qrnn=self.qrnn, clip=self.clip, pretrained_fnames=self.pretrained_fnames)
        # compared to standard Adam, we set beta_1 to 0.8
        learn.opt_fn = partial(optim.Adam, betas=(0.8, 0.99))
        learn.true_wd = False
        print("true_wd: ", learn.true_wd)
        learn.metrics = [accuracy_fwd, accuracy_bwd] if self.bidir else [accuracy]
        return learn

    @property
    def lm_type(self):
        return contrib_data.LanguageModelType.BiLM if self.bidir else contrib_data.LanguageModelType.FwdLM

    def load_wiki_data(self):
        trn_path = self.dataset_path / f'{self.lang}.wiki.train.tokens'
        val_path = self.dataset_path / f'{self.lang}.wiki.valid.tokens'
        tst_path = self.dataset_path / f'{self.lang}.wiki.test.tokens'
        for path_ in [trn_path, val_path, tst_path]:
            assert path_.exists(), f'Error: {path_} does not exist.'
        if self.subword:
            # apply sentencepiece tokenization
            trn_path = self.dataset_path / f'{self.lang}.wiki.train.tokens'
            val_path = self.dataset_path / f'{self.lang}.wiki.valid.tokens'

            read_file(trn_path, 'train')
            read_file(val_path, 'valid')

            sp = get_sentencepiece(self.dataset_path, trn_path, self.name, vocab_size=self.max_vocab)

            lm_type = contrib_data.LanguageModelType.BiLM if self.bidir else contrib_data.LanguageModelType.FwdLM

            data_lm = TextLMDataBunch.from_csv(self.dataset_path, 'train.csv', **sp, bs=self.bs, bptt=self.bptt, lm_type=lm_type)
            itos = data_lm.train_ds.vocab.itos
            stoi = data_lm.train_ds.vocab.stoi
        else:
            # read the already whitespace separated data without any preprocessing
            trn_tok = read_whitespace_file(trn_path)
            val_tok = read_whitespace_file(val_path)
            itos_fname = self.cache_dir / f'itos.pkl'
            if not itos_fname.exists():
                # create the vocabulary
                cnt = Counter(word for sent in trn_tok for word in sent)
                itos = [o for o, c in cnt.most_common(n=self.max_vocab)]
                itos.insert(1, PAD)  #   set pad id to 1 to conform to fast.ai standard
                assert UNK in itos, f'Unknown words are expected to have been replaced with {UNK} in the data.'

                # save vocabulary
                print(f"Saving vocabulary as {itos_fname}")
                with open(itos_fname, 'wb') as f:
                    pickle.dump(itos, f)
            else:
                print("Loading itos:", itos_fname)
                itos = np.load(itos_fname)
            vocab = Vocab(itos)
            stoi = vocab.stoi

            trn_ids = np.array([([stoi.get(w, stoi[UNK]) for w in s]) for s in trn_tok])
            val_ids = np.array([([stoi.get(w, stoi[UNK]) for w in s]) for s in val_tok])



            # data_lm = TextLMDataBunch.from_ids(dir_path, trn_ids, [], val_ids, [], len(itos))
            data_lm = TextLMDataBunch.from_ids(path=self.dataset_path, vocab=vocab, train_ids=trn_ids,
                                               valid_ids=val_ids, bs=self.bs, bptt=self.bptt,
                                               lm_type=self.lm_type)
        itos, stoi, trn_path = data_lm.vocab.itos, data_lm.vocab.stoi, data_lm.path
        print('Size of vocabulary:', len(itos))
        print('First 10 words in vocab:', ', '.join([itos[i] for i in range(10)]))
        return data_lm

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
