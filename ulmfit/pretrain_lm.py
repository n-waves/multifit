"""
Script to train a model on a preprocessed Wiki dataset. Note that the dataset is
expected to have been tokenized with Moses and processed with `postprocess_wikitext.py`.
That is, the data is expected to be white-space separated and numbers are expected
to be split.
"""
import fastai
import fire
import numpy as np

from fastai import *
from fastai.text import *
import torch
from fastai_contrib.utils import read_file, read_whitespace_file,\
    DataStump, validate, PAD, UNK, get_sentencepiece

import pickle

from pathlib import Path

from collections import Counter

# to install, do:
# conda install -c pytorch -c fastai fastai pytorch-nightly [cuda92]
# cupy needs to be installed for QRNN

def pretrain_lm(dir_path, lang='en', cuda_id=0, qrnn=True, subword=False, max_vocab=60000,
                bs=70, bptt=70, name='wt-103', num_epochs=10, ds_pct=1.0):
    """
    :param dir_path: The path to the directory of the file.
    :param lang: the language unicode
    :param cuda_id: The id of the GPU. Uses GPU 0 by default or no GPU when
                    run on CPU.
    :param qrnn: Use a QRNN. Requires installing cupy.
    :param subword: Use sub-word tokenization on the cleaned data.
    :param max_vocab: The maximum size of the vocabulary.
    :param bs: The batch size.
    :param bptt: The back-propagation-through-time sequence length.
    :param name: The name used for both the model and the vocabulary.
    :param model_dir: The path to the directory where the models should be saved
    """
    results = {}
    model_dir = 'models' # removed from params, as it is absolute models location in train_clas and here it is relative
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

    trn_path = dir_path / f'{lang}.wiki.train.tokens'
    val_path = dir_path / f'{lang}.wiki.valid.tokens'
    tst_path = dir_path / f'{lang}.wiki.test.tokens'
    for path_ in [trn_path, val_path, tst_path]:
        assert path_.exists(), f'Error: {path_} does not exist.'

    if subword:
        # apply sentencepiece tokenization
        trn_path = dir_path / f'{lang}.wiki.train.tokens'
        val_path = dir_path / f'{lang}.wiki.valid.tokens'

        read_file(trn_path, 'train')
        read_file(val_path, 'valid')

        sp = get_sentencepiece(dir_path, trn_path, name, vocab_size=max_vocab)

        data_lm = TextLMDataBunch.from_csv(dir_path, 'train.csv', **sp)
        itos = data_lm.train_ds.vocab.itos
        stoi = data_lm.train_ds.vocab.stoi
    else:
        # read the already whitespace separated data without any preprocessing
        trn_tok = read_whitespace_file(trn_path)
        val_tok = read_whitespace_file(val_path)
        if ds_pct < 1.0:
            trn_tok = trn_tok[:max(20, int(len(trn_tok) * ds_pct))]
            val_tok = val_tok[:max(20, int(len(val_tok) * ds_pct))]
            print(f"Limiting data sets to {ds_pct*100}%, trn {len(trn_tok)}, val: {len(val_tok)}")

        # create the vocabulary
        cnt = Counter(word for sent in trn_tok for word in sent)
        itos = [o for o,c in cnt.most_common(n=max_vocab)]
        itos.insert(1, PAD)  #  set pad id to 1 to conform to fast.ai standard
        assert UNK in itos, f'Unknown words are expected to have been replaced with {UNK} in the data.'

        vocab = Vocab(itos)
        stoi = vocab.stoi

        # save vocabulary
        itos_fname = model_dir / f'itos_{name}.pkl'
        print(f"Saving vocabulary as {itos_fname}")
        results['itos_fname'] = itos_fname
        with open(itos_fname, 'wb') as f:
            pickle.dump(itos, f)

        trn_ids = np.array([([stoi.get(w, stoi[UNK]) for w in s]) for s in trn_tok])
        val_ids = np.array([([stoi.get(w, stoi[UNK]) for w in s]) for s in val_tok])

        # data_lm = TextLMDataBunch.from_ids(dir_path, trn_ids, [], val_ids, [], len(itos))
        data_lm = TextLMDataBunch.from_ids(path=dir_path, vocab=vocab, train_ids=trn_ids,
                                           valid_ids=val_ids, bs=bs, bptt=bptt)

    print('Size of vocabulary:', len(itos))
    print('First 10 words in vocab:', ', '.join([itos[i] for i in range(10)]))

    # these hyperparameters are for training on ~100M tokens (e.g. WikiText-103)
    # for training on smaller datasets, more dropout is necessary
    if qrnn:
        emb_sz, nh, nl = 400, 1550, 3
        #dps = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        dps = np.array([0.25, 0.1, 0.2, 0.02, 0.15])
        drop_mult = 0.1
    else:
        emb_sz, nh, nl = 400, 1150, 3
        # emb_sz, nh, nl = 400, 1150, 3
        dps = np.array([0.25, 0.1, 0.2, 0.02, 0.15])
        drop_mult = 0.1

    fastai.text.learner.default_dropout['language'] = dps * drop_mult
    learn = language_model_learner(data_lm, bptt=bptt, emb_sz=emb_sz, nh=nh, nl=nl, pad_token=1,
                           drop_mult=drop_mult, tie_weights=True, model_dir=model_dir,
                           bias=True, qrnn=qrnn, clip=0.12)
    # compared to standard Adam, we set beta_1 to 0.8
    learn.opt_fn = partial(optim.Adam, betas=(0.8, 0.99))
    learn.true_wd = False

    fit_one_cycle(learn, num_epochs, 5e-3, (0.8, 0.7), wd=1e-7)

    if not subword and max_vocab is None:
        # only if we use the unpreprocessed version and the full vocabulary
        # are the perplexity results comparable to previous work
        print(f"Validating model performance with test tokens from: {trn_path}")
        tst_tok = read_whitespace_file(trn_path)
        tst_ids = np.array([([stoi.get(w, stoi[UNK]) for w in s]) for s in tst_tok])
        logloss, perplexity = validate(learn.model, tst_ids, bptt)
        print('Test logloss:', logloss.item(), 'perplexity:', perplexity.item())

    print(f"Saving models at {learn.path / learn.model_dir}")
    learn.save(f'{model_name}_{name}')

    opt_state_path = learn.path / learn.model_dir / f'{model_name}3_{name}_state.pth'
    print(f"Saving optimiser state at {opt_state_path}")
    torch.save(learn.opt.opt.state_dict(), opt_state_path)

    results['accuracy'] = learn.validate()[1]
    return results

if __name__ == '__main__':
    fire.Fire(pretrain_lm)
