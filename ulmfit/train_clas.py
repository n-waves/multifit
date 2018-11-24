"""
Train a classifier on top of a language model trained with `pretrain_lm.py`.
Optionally fine-tune LM before.
"""
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


def new_train_clas(data_dir, lang='en', cuda_id=0, pretrain_name='wt103', model_dir='models',
                   qrnn=False, num_lm_epochs=10,
                   fine_tune=True, max_vocab=60000, bs=20, bptt=70, name='imdb-clas',
                   dataset='imdb', bidir=False, ds_pct=1.0, train=True):
    """
    :param data_dir: The path to the `data` directory
    :param lang: the language unicode
    :param cuda_id: The id of the GPU. Uses GPU 0 by default or no GPU when
                    run on CPU.
    :param pretrain_name: name of the pretrained model
    :param model_dir: The path to the directory where the pretrained model is saved
    :param qrrn: Use a QRNN. Requires installing cupy.
    :param fine_tune: Fine-tune the pretrained language model
    :param max_vocab: The maximum size of the vocabulary.
    :param bs: The batch size.
    :param bptt: The back-propagation-through-time sequence length.
    :param name: The name used for both the model and the vocabulary.
    :param dataset: The dataset used for evaluation. Currently only IMDb and
                    XNLI are implemented. Assumes dataset is located in `data`
                    folder and that name of folder is the same as dataset name.
    """
    results={}
    if not torch.cuda.is_available():
        print('CUDA not available. Setting device=-1.')
        cuda_id = -1
    torch.cuda.set_device(cuda_id)

    print(f'Dataset: {dataset}. Language: {lang}.')
    assert dataset in DATASETS, f'Error: {dataset} processing is not implemented.'
    assert (dataset == 'imdb' and lang == 'en') or not dataset == 'imdb',\
        'Error: IMDb is only available in English.'

    data_dir = Path(data_dir)
    assert data_dir.name in ['data', 'test'],\
        f'Error: Name of data directory should be data, not {data_dir.name}.'
    dataset_dir = data_dir / dataset
    model_dir = Path(model_dir)


    if qrnn:
        print('Using QRNNs...')
    model_name = 'qrnn' if qrnn else 'lstm'
    lm_name = f'{model_name}_{pretrain_name}'
    pretrained_fname = (lm_name, f'itos_{pretrain_name}')

    ensure_paths_exists(data_dir,
                        dataset_dir,
                        model_dir,
                        model_dir/f"{pretrained_fname[0]}.pth",
                        model_dir/f"{pretrained_fname[1]}.pkl")

    if bidir:
        print("BiLM")
        classifier_learner = bilm_text_classifier_learner
        lm_learner = bilm_learner
    else:
        classifier_learner = text_classifier_learner
        lm_learner = language_model_learner

    lm_type = LanguageModelType.BiLM if bidir else LanguageModelType.FwdLM
    data_clas, data_lm = get_datasets(dataset, dataset_dir, bptt, bs, lang, max_vocab, ds_pct, lm_type=lm_type)

    if qrnn:
        emb_sz, nh, nl = 400, 1550, 3
    else:
        emb_sz, nh, nl = 400, 1150, 3

    lm_enc_finetuned  = f"{lm_name}_{dataset}_enc"
    if fine_tune and not (model_dir/f"{lm_enc_finetuned}.pth").exists():
        print('Fine-tuning the language model...', lm_enc_finetuned)
        learn = lm_learner(
            data_lm, bptt=bptt, emb_sz=emb_sz, nh=nh, nl=nl, qrnn=qrnn,
            pad_token=PAD_TOKEN_ID,
            pretrained_fnames=pretrained_fname,
            path=model_dir.parent, model_dir=model_dir.name,
            drop_mult=0.3)
        if bidir:
            learn.metrics = [accuracy_fwd, accuracy_bwd]
        else:
            learn.metrics = [accuracy]

        learn.fit_one_cycle(1, 1e-2, moms=(0.8, 0.7))
        learn.unfreeze()
        if num_lm_epochs > 0: learn.fit_one_cycle(num_lm_epochs, 1e-3, moms=(0.8, 0.7))

        # save encoder
        learn.save_encoder(lm_enc_finetuned)


    learn = classifier_learner(data_clas, bptt=bptt, pad_token=PAD_TOKEN_ID,
                                  path=model_dir.parent, model_dir=model_dir.name,
                                  qrnn=qrnn, emb_sz=emb_sz, nh=nh, nl=nl, drop_mult=0.5)

    try:
        print(f"Loading classifier {model_name}_{name}")
        learn.load(f'{model_name}_{name}')

    except FileNotFoundError:
        learn.load_encoder(lm_enc_finetuned)
        print("loading encoder")
        train = True

    if train:
        learn.true_wd = False
        print("Starting classifier training")
        learn.fit_one_cycle(1, 5e-2, moms=(0.8, 0.7), wd=1e-7)

        learn.freeze_to(-2)
        learn.fit_one_cycle(1, slice(5e-2 / (2.6 ** 4), 5e-2), moms=(0.8, 0.7), wd=1e-7)

        learn.freeze_to(-3)
        learn.fit_one_cycle(1, slice(5e-4 / (2.6 ** 4), 5e-4), moms=(0.8, 0.7), wd=1e-7)

        learn.unfreeze()
        learn.fit_one_cycle(2, slice(1e-2 / (2.6 ** 4), 1e-2), moms=(0.8, 0.7), wd=1e-7)

        print(f"Saving models at {learn.path / learn.model_dir}")
        learn.save(f'{model_name}_{name}')

    results['accuracy'] = learn.recorder.metrics[-1][0]
    return results


def get_datasets(dataset, dataset_dir, bptt, bs, lang, max_vocab, ds_pct, lm_type):
    tmp_dir = dataset_dir / 'tmp'
    tmp_dir.mkdir(exist_ok=True)
    vocab_file = tmp_dir / f'vocab_{lang}.pkl'
    if not (tmp_dir / f'{TRN}_{lang}_ids.npy').exists():
        print('Reading the data...')
        toks, lbls = read_clas_data(dataset_dir, dataset, lang)
        # create the vocabulary
        counter = Counter(word for example in toks[TRN]+toks[TST]+toks[VAL] for word in example)
        itos = [word for word, count in counter.most_common(n=max_vocab)]
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
            np.save(tmp_dir / f'{split}_{lang}_ids.npy', ids[split])
            np.save(tmp_dir / f'{split}_{lang}_lbl.npy', lbls[split])
    else:
        print('Loading the pickled data...')
        ids, lbls = {}, {}
        for split in [TRN, VAL, TST]:
            ids[split] = np.load(tmp_dir / f'{split}_{lang}_ids.npy')
            lbls[split] = np.load(tmp_dir / f'{split}_{lang}_lbl.npy')
        with open(vocab_file, 'rb') as f:
            vocab = pickle.load(f)
    print(f'Train size: {len(ids[TRN])}. Valid size: {len(ids[VAL])}. '
          f'Test size: {len(ids[TST])}.')
    if ds_pct < 1.0:
        print(f"Making the dataset smaller {ds_pct}")
    for split in [TRN, VAL, TST]:
        ids[split] = np.array([np.array(e, dtype=np.int) for e in ids[split]])
        lbls[split] = np.array([np.array(e, dtype=np.int) for e in lbls[split]])
    data_lm = TextLMDataBunch.from_ids(path=tmp_dir, vocab=vocab, train_ids=np.concatenate([ids[TRN],ids[TST]]),
                                       valid_ids=ids[VAL], bs=bs, bptt=bptt, lm_type=lm_type)
    #  TODO TextClasDataBunch allows tst_ids as input, but not tst_lbls?
    data_clas = TextClasDataBunch.from_ids(
        path=tmp_dir, vocab=vocab, train_ids=ids[TRN], valid_ids=ids[VAL],
        train_lbls=lbls[TRN], valid_lbls=lbls[VAL], bs=bs, classes={l:l for l in lbls[TRN]})

    print(f"Sizes of train_ds {len(data_clas.train_ds)}, valid_ds {len(data_clas.valid_ds)}")
    return data_clas, data_lm


if __name__ == '__main__':
    fire.Fire(new_train_clas)
