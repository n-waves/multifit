"""
Script to train a model on a preprocessed Wiki dataset. Note that the dataset is
expected to have been tokenized with Moses and processed with `postprocess_wikitext.py`.
That is, the data is expected to be white-space separated and numbers are expected
to be split.
"""
import fastai
import fire
import numpy as np

from fastai import DataBunch, partial, optim, fit_one_cycle
from fastai.text import LanguageModelLoader, get_language_model, RNNLearner, TextLMDataBunch, NumericalizedDataset, \
    Vocab, language_model_learner
import torch
from fastai_contrib.utils import read_file, read_whitespace_file,\
    DataStump, validate, PAD, UNK
from fastai_contrib.learner import bilm_learner

import pickle

from pathlib import Path

from collections import Counter

# to install, do:
# conda install -c pytorch -c fastai fastai pytorch-nightly [cuda92]
# cupy needs to be installed for QRNN


def pretrain_lm(dir_path, cuda_id=0, qrnn=True, clean=True, max_vocab=60000,
                bs=70, bptt=70, name='wt-103', model_dir='models', num_epochs=10,
                bidir=True):
    """
    :param dir_path: The path to the directory of the file.
    :param cuda_id: The id of the GPU. Uses GPU 0 by default or no GPU when
                    run on CPU.
    :param qrrn: Use a QRNN. Requires installing cupy.
    :param clean: Train on the clean
    :param max_vocab: The maximum size of the vocabulary.
    :param bs: The batch size.
    :param bptt: The back-propagation-through-time sequence length.
    :param name: The name used for both the model and the vocabulary.
    :param model_dir: The path to the directory where the models should be saved
    :param bidir: whether the language model is bidirectional
    """
    if not torch.cuda.is_available():
        print('CUDA not available. Setting device=-1.')
        cuda_id = -1
    torch.cuda.set_device(cuda_id)

    dir_path = Path(dir_path)
    assert dir_path.exists()
    model_dir = Path(model_dir)
    model_dir.mkdir(exist_ok=True)
    print('Batch size:', bs)
    print('Max vocab:', max_vocab)
    model_name = 'qrnn' if qrnn else 'lstm'
    if qrnn:
        print('Using QRNNs...')

    trn_path = dir_path / 'wiki.train.tokens'
    val_path = dir_path / 'wiki.valid.tokens'
    tst_path = dir_path / 'wiki.test.tokens'
    for path_ in [trn_path, val_path, tst_path]:
        assert path_.exists(), f'Error: {path_} does not exist.'

    if clean:
        # read the already whitespace separated data without any preprocessing
        trn_tok = read_whitespace_file(trn_path)
        val_tok = read_whitespace_file(val_path)

        # create the vocabulary
        cnt = Counter(word for sent in trn_tok for word in sent)
        itos = [o for o,c in cnt.most_common(n=max_vocab)]
        itos.insert(1, PAD)  #  set pad id to 1 to conform to fast.ai standard
        assert UNK in itos, f'Unknown words are expected to have been replaced with {UNK} in the data.'
        stoi = {w: i for i, w in enumerate(itos)}

        vocab = Vocab(itos)
        stoi = vocab.stoi
        trn_ids = np.array([([stoi.get(w, stoi[UNK]) for w in s]) for s in trn_tok])
        val_ids = np.array([([stoi.get(w, stoi[UNK]) for w in s]) for s in val_tok])

        # data_lm = TextLMDataBunch.from_ids(dir_path, trn_ids, [], val_ids, [], len(itos))
        data_lm = TextLMDataBunch.create(
            train_ds=NumericalizedDataset(vocab, trn_ids, labels=np.zeros(len(trn_ids), dtype=np.int)),
            valid_ds=NumericalizedDataset(vocab, val_ids, labels=np.zeros(len(val_ids), dtype=np.int)),
            bs=bs, bptt=bptt)
    else:
        # apply fastai preprocessing and tokenization
        read_file(trn_path, 'train')
        read_file(val_path, 'valid')

        data_lm = TextLMDataBunch.from_csv(dir_path, max_vocab=max_vocab)
        itos = data_lm.train_ds.vocab.itos
        stoi = data_lm.train_ds.vocab.stoi

    print('Size of vocabulary:', len(itos))
    print('First 10 words in vocab:', ', '.join([itos[i] for i in range(10)]))

    # these hyperparameters are for training on ~100M tokens (e.g. WikiText-103)
    # for training on smaller datasets, more dropout is necessary
    if qrnn:
        emb_sz, nh, nl = 400, 1550, 3
        #dps = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        dps = np.array([0.25, 0.1, 0.2, 0.02, 0.15]) * 0.1
        drop_mult = 0.1
    else:
        emb_sz, nh, nl = 400, 1150, 3
        # emb_sz, nh, nl = 400, 1150, 3
        dps = np.array([0.25, 0.1, 0.2, 0.02, 0.15])
        drop_mult = 0.1

    fastai.text.learner.default_dropout['language'] = dps

    lm_learner = bilm_learner if bidir else language_model_learner
    learn = lm_learner(data_lm, bptt=bptt, emb_sz=emb_sz, nh=nh, nl=nl, pad_token=1,
                       drop_mult=drop_mult, tie_weights=True,
                       bias=True, qrnn=qrnn, clip=0.12)
    # compared to standard Adam, we set beta_1 to 0.8
    learn.opt_fn = partial(optim.Adam, betas=(0.8, 0.99))
    learn.true_wd = False

    # save vocabulary
    print('Saving vocabulary...')
    with open(model_dir / f'itos_{name}.pkl', 'wb') as f:
        pickle.dump(itos, f)

    fit_one_cycle(learn, num_epochs, 5e-3, (0.8, 0.7), wd=1e-7)

    if clean and max_vocab is None:
        # only if we use the unpreprocessed version and the full vocabulary
        # are the perplexity results comparable to previous work
        tst_tok = read_whitespace_file(trn_path)
        tst_ids = np.array([([stoi.get(w, stoi[UNK]) for w in s]) for s in tst_tok])
        logloss, perplexity = validate(model, tst_ids, bptt)
        print('Test logloss:', logloss.item(), 'perplexity:', perplexity.item())

    learn.save(f'{model_name}_{name}')
    torch.save(learn.opt.opt.state_dict(), learn.path / learn.model_dir / f'{model_name}3_{name}_state.pth')


if __name__ == '__main__':
    fire.Fire(pretrain_lm)
