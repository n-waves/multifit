"""
Utility methods for data processing.
"""
import pandas as pd
import numpy as np
import fire
from fastai import F, to_device
import torch
from tqdm import tqdm
import re
import csv
import pathlib
import tarfile
from sklearn import model_selection

EOS = '<eos>'
UNK = '<unk>'
PAD = '<pad>'
PAD_TOKEN_ID = 1

number_match_re = re.compile(r'^([0-9]+[,.]?)+$')
number_split_re = re.compile(r'([,.])')

CLASSES = ['neg', 'pos', 'unsup']


def get_texts(path):
    texts, labels = [],[]
    for idx, label in enumerate(CLASSES):
        for fname in (path/label).glob('*.*'):
            texts.append(fname.open('r', encoding='utf-8').read())
            labels.append(idx)
    return np.array(texts), np.array(labels)


def prepare_imdb(file_path: str, prepare_lm = False):
    """
    function to extract aclImdb and combine into fastai standard format of labels and then text
    columns

    Args:
        file_path: path to the aclImdb.tgz
        prepare_lm (bool): prepare file for language model finetuning

    Returns:
        None
    """

    file_path = pathlib.Path(file_path)
    dir_path = pathlib.Path(file_path.stem).resolve()
    assert tarfile.is_tarfile(file_path), "this is not a valid targz file"

    if not dir_path.exists():
        print(f"Extracting {file_path} to {dir_path}. This may take a long time...")
        tgz_file = tarfile.open(file_path)
        tgz_file.extractall()
        assert dir_path.exists()
        print(f"Extracted to {dir_path}")

    CLAS_PATH = dir_path / 'imdb_clas'
    CLAS_PATH.mkdir(exist_ok=True)

    LM_PATH = dir_path /'imdb_lm'
    LM_PATH.mkdir(exist_ok=True)

    # processing the split files to create train.csv and test.csv in fastai format
    col_names = ['labels', 'text']
    trn_texts, trn_labels = get_texts(dir_path/ 'train')
    val_texts, val_labels = get_texts(dir_path / 'test')
    np.random.seed(42)
    trn_idx = np.random.permutation(len(trn_texts))
    val_idx = np.random.permutation(len(val_texts))
    trn_texts = trn_texts[trn_idx]
    val_texts = val_texts[val_idx]
    trn_labels = trn_labels[trn_idx]
    val_labels = val_labels[val_idx]

    df_trn = pd.DataFrame({'text': trn_texts, 'labels': trn_labels}, columns=col_names)
    df_val = pd.DataFrame({'text': val_texts, 'labels': val_labels}, columns=col_names)
    print(f"df_trn has {len(df_trn)} rows, while df_val has {len(df_val)} rows")
    print(f"Writing them to {CLAS_PATH}")
    df_trn[df_trn['labels'] != 2].to_csv(CLAS_PATH / 'train.csv', header=False, index=False)
    df_val.to_csv(CLAS_PATH / 'test.csv', header=False, index=False)

    (CLAS_PATH / 'classes.txt').open('w', encoding='utf-8').writelines(f'{o}\n' for o in CLASSES)

    if prepare_lm:
        print("Preparing LM data")
        trn_texts, val_texts = model_selection.train_test_split(
            np.concatenate([trn_texts, val_texts]), test_size=0.1)
        print(f"trn_texts has {len(trn_texts)} samples, while val_texts has {len(val_texts)} rows")
        print(f"Writing them to {LM_PATH}")
        df_trn = pd.DataFrame({'text': trn_texts, 'labels': [0] * len(trn_texts)}, columns=col_names)
        df_val = pd.DataFrame({'text': val_texts, 'labels': [0] * len(val_texts)}, columns=col_names)

        df_trn.to_csv(LM_PATH / 'train.csv', header=False, index=False)
        df_val.to_csv(LM_PATH / 'test.csv', header=False, index=False)


def replace_number(token):
    """Replaces a number and returns a list of one or multiple tokens."""
    if number_match_re.match(token):
        return number_split_re.sub(r' @\1@ ', token)
    return token


def read_file(file_path, outname):
    """Reads a text file and writes it to a .csv."""
    with open(file_path, encoding='utf8') as f:
        text = f.readlines()
    df = pd.DataFrame(
        {'text': np.array(text), 'labels': np.zeros(len(text))},
        columns=['labels', 'text'])
    df.to_csv(file_path.parent / f'{outname}.csv', header=False, index=False)


def read_whitespace_file(filepath):
    """Reads a file and prepares the tokens."""
    tokens = []
    with open(filepath, encoding='utf-8') as f:
        for line in f:
            # newlines are replaced with EOS
            tokens.append(line.split() + [EOS])
    return np.array(tokens)


def read_imdb(file_path, mt):
    toks, lbls = [], []
    print(f'Reading {file_path}...')
    with open(file_path, encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            label, text = row
            lbls.append(label)
            raw_tokens = mt.tokenize(text, return_str=True).split(' ') + [EOS]
            tokens = []
            for token in raw_tokens:
                if number_match_re.match(token):
                    tokens += number_split_re.sub(r' @\1@ ', token).split()
                else:
                    tokens.append(token)
            toks.append(tokens)
    return np.array(toks), np.array(lbls)


class DataStump:
    """Placeholder class as LanguageModelLoader requires object with ids attribute."""
    def __init__(self, ids):
        self.ids = ids
        self.loss_func = F.cross_entropy


def validate(model, ids, bptt=2000):
    """
    Return the validation loss and perplexity of a model
    :param model: model to test
    :param ids: data on which to evaluate the model
    :param bptt: bptt for this evaluation (doesn't change the result, only the speed)
    From https://github.com/sgugger/Adam-experiments/blob/master/lm_val_fns.py#L34
    """
    data = TextReader(np.concatenate(ids), bptt)
    model.eval()
    model.reset()
    total_loss, num_examples = 0., 0
    for inputs, targets in tqdm(data):
        outputs, raws, outs = model(to_device(inputs, None))
        p_vocab = F.softmax(outputs, 1)
        for i, pv in enumerate(p_vocab):
            targ_pred = pv[targets[i]]
            total_loss -= torch.log(targ_pred.detach())
        num_examples += len(inputs)
    mean = total_loss / num_examples  # divide by total number of tokens
    return mean, np.exp(mean)


class TextReader():
    """ Returns a language model iterator that iterates through batches that are of length N(bptt,5)
    The first batch returned is always bptt+25; the max possible width.  This is done because of they way that pytorch
    allocates cuda memory in order to prevent multiple buffers from being created as the batch width grows.
    From: https://github.com/sgugger/Adam-experiments/blob/master/lm_val_fns.py#L3
    """
    def __init__(self, nums, bptt, backwards=False):
        self.bptt,self.backwards = bptt,backwards
        self.data = self.batchify(nums)
        self.i,self.iter = 0,0
        self.n = len(self.data)

    def __iter__(self):
        self.i,self.iter = 0,0
        while self.i < self.n-1 and self.iter<len(self):
            res = self.get_batch(self.i, self.bptt)
            self.i += self.bptt
            self.iter += 1
            yield res

    def __len__(self): return self.n // self.bptt

    def batchify(self, data):
        data = np.array(data)[:,None]
        if self.backwards: data=data[::-1]
        return torch.LongTensor(data)

    def get_batch(self, i, seq_len):
        source = self.data
        seq_len = min(seq_len, len(source) - 1 - i)
        return source[i:i+seq_len], source[i+1:i+1+seq_len].view(-1)


if __name__ == "__main__":
    fire.Fire()  # allows using all functions via CLI e.g. python utils.py prepare_imdb aclImdb.tgz