"""
Utility methods for data processing.
"""
import pandas as pd
import numpy as np
from fastai import F, to_device
import torch
from tqdm import tqdm
import re
import csv
from functools import reduce
from fastai.text.data import TextDataset
from fastai.text.transform import Tokenizer, BaseTokenizer, Vocab, default_rules
from fastai.torch_core import *
from pathlib import Path

EOS = '<eos>'
UNK = '<unk>'
PAD = '<pad>'
PAD_TOKEN_ID = 1

number_match_re = re.compile(r'^([0-9]+[,.]?)+$')
number_split_re = re.compile(r'([,.])')

class SentencepieceTokenizer(BaseTokenizer):
    def __init__(self, path:PathOrStr, cache_name:str='tmp'):
        try:
            import sentencepiece as spm  
        except ImportError:
            raise Exception('sentencepiece module is missing: run `pip install sentencepiece`')
        self.tok = spm.SentencePieceProcessor()
        self.tok.Load(str(Path(path) / cache_name / 'm.model'))
    def tokenizer(self, t:str) -> List[str]:
        return self.tok.EncodeAsPieces(t)
    def add_special_cases(self, toks:Collection[str]):
        pass

def get_sentencepiece(path:PathOrStr, dataset:TextDataset, rules:ListRules=None,
                      cache_name:str='tmp', vocab_size:int=30000, 
                      model_type:str='unigram', input_sentence_size:int=1E7, 
                      pad_idx:int=PAD_TOKEN_ID):
    try:
        import sentencepiece as spm  
    except ImportError:
        raise Exception('sentencepiece module is missing: run `pip install sentencepiece`')
    
    path = Path(path)
    os.makedirs(path / cache_name, exist_ok=True)
    rules = rules if rules else default_rules
    
    if not os.path.isfile(path / cache_name / 'm.model') or not os.path.isfile(path / 'itos.pkl'):
        raw_text = reduce(lambda t, rule: rule(t), rules, '\n'.join(dataset.x))
        raw_text_path = path / cache_name / 'all_text.txt'
        with open(raw_text_path, 'w') as f:
            f.write(raw_text)
      
        sp_params = f'--input={raw_text_path} --pad_id={pad_idx} --unk_id=0' \
                    f'--character_coverage=1.0 --bos_id=-1 --eos_id=-1 ' \
                    f'--input_sentence_size={int(input_sentence_size)} ' \
                    f'--model_prefix={path / cache_name / "m"} ' \
                    f'--vocab_size={vocab_size} --model_type={model_type} '
        spm.SentencePieceTrainer.Train(sp_params)
  
        with open(path / cache_name / 'm.vocab', 'r') as f:
            vocab = [line.split('\t')[0] for line in f.readlines()]
            vocab[0] = UNK
            vocab[pad_idx] = PAD
  
        pickle.dump(vocab, open(path / 'itos.pkl', 'wb'))
    
    vocab = Vocab(pickle.load(open(path / 'itos.pkl', 'rb')))
    spt = SentencepieceTokenizer(path, cache_name)
    tokenizer = Tokenizer(tok_func=lambda lang: spt, rules=rules)
    
    return {'tokenizer': tokenizer, 'vocab': vocab}

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