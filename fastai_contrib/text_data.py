import os
import pathlib
import pickle
from functools import reduce
from typing import Collection, List

from pandas import DataFrame
from sacremoses import MosesTokenizer

import fastai
from fastai.basic_data import DataBunch

from fastai.core import ListRules, PathOrStr, defaults, IntsOrStrs, is_listy
from fastai.data_block import ItemLists
from fastai.text import Tokenizer, BaseTokenizer, Vocab, SPProcessor, TextList, TextLMDataBunch


class MosesPreprocessingFunc():
    def __init__(self, lang: str):
        self.mt = MosesTokenizer(lang)

    def __call__(self, t: str) -> str:
        return self.mt.tokenize(t, return_str=True, escape=True)


class SentencePieceTokenizer(Tokenizer):
    "Put together rules and a tokenizer function to tokenize text with multiprocessing."
    def __init__(self, spm_model, lang:str='en', pre_rules:ListRules=None,
                 post_rules:ListRules=None, special_cases:Collection[str]=None, n_cpus:int=None):
        # moses is added to preprocessing functions
        super().__init__(self.tok_fun_with_sp, lang, pre_rules, post_rules, special_cases, n_cpus)
        self.spm_model = spm_model

    def tok_fun_with_sp(self, lang):
        try:
            import sentencepiece as spm
        except ImportError:
            raise Exception('sentencepiece module is missing: run `pip install sentencepiece`')
        tok = BaseTokenizer(lang)
        tok.sp = spm.SentencePieceProcessor()
        tok.sp.Load(str(self.spm_model))
        return tok

    def process_text(self, t:str, tok:BaseTokenizer) -> List[str]:
        "Process one text `t` with tokenizer `tok`."
        toks = super().process_text(t, tok)
        toks = tok.sp.EncodeAsPieces(" ".join(toks))
        return toks


full_char_coverage_langs = ["bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr", "ga", "hr", "hu",
                       "it","lt","lv","mt","nl","pl","pt","ro","sk","sl","sv"] # all European langus


def get_sentencepiece(cache_dir:PathOrStr, load_text, pre_rules: ListRules=None, post_rules:ListRules=None,
                      vocab_size:int=30000, model_type:str='unigram', input_sentence_size:int=1E7, lang='en', fixed_character_coverage=False):
    try:
        import sentencepiece as spm
    except ImportError:
        raise Exception('sentencepiece module is missing: run `pip install sentencepiece`')

    cache_dir = pathlib.Path(cache_dir)
    pre_rules = pre_rules if pre_rules is not None else defaults.text_pre_rules
    post_rules = post_rules if post_rules is not None else defaults.text_post_rules

    special_cases = defaults.text_spec_tok # + ['xxlink', 'xxuser', 'xxnumber', 'xxemoji', 'yyemoji']
    if not os.path.isfile(cache_dir / 'spm.model') or not os.path.isfile(cache_dir / f'itos.pkl'):
        # load the text from the train tokens file
        text = load_text()
        text = filter(lambda x: len(x.rstrip(" ")), text)
        text = (reduce(lambda t, rule: rule(t), pre_rules, line) for line in text)
        def cleanup_n_postprocess(t):
            t = t.split()
            for r in post_rules:
                t = r(t)
            return ' '.join(t)
        text = map(cleanup_n_postprocess, text)
        raw_text_path = cache_dir / 'all_text.txt'
        with open(raw_text_path, 'w') as f: f.write("\n".join(text))

        if fixed_character_coverage:
            char_coverage = 0.9995
        else:
            char_coverage = 1 if lang in full_char_coverage_langs else 0.99

        sp_params = [
            f"--input={raw_text_path}",
            f"--character_coverage={char_coverage}",
            f"--unk_id={len(special_cases)}",
            f"--pad_id=-1",
            f"--bos_id=-1",
            f"--eos_id=-1",
            f"--max_sentence_length=20480",
            f"--input_sentence_size={int(input_sentence_size)}",
            f"--user_defined_symbols={','.join(special_cases)}",
            f"--model_prefix={cache_dir/'spm'}",
            f"--vocab_size={vocab_size} --model_type={model_type}"]
        spm.SentencePieceTrainer.Train(" ".join(sp_params))

        with open(cache_dir / 'spm.vocab', 'r') as f:
            vocab = [line.split('\t')[0] for line in f.readlines()]

        pickle.dump(vocab, open(cache_dir/ f'itos.pkl', 'wb'))
    # todo add post rules
    vocab = Vocab(pickle.load(open(cache_dir / f'itos.pkl', 'rb')))
    # We cannot use lambdas or local methods here, since `tok_func` needs to be
    # pickle-able in order to be called in subprocesses when multithread tokenizing
    tokenizer = SentencePieceTokenizer(cache_dir/'spm.model',
                                lang=lang,
                                pre_rules=pre_rules,
                                post_rules=post_rules)
    return {'tokenizer': tokenizer, 'vocab': vocab}

class SPProcessor2(SPProcessor):
    def process(self, ds):
        super().process(ds)
        ds.vocab.sp_model = self.sp_model
        ds.vocab.sp_vocab = self.sp_vocab

def get_sentencepiece_fastai(cache_dir: PathOrStr,  pre_rules: ListRules = None,
                          post_rules: ListRules = None,
                          vocab_size: int = 30000, lang='en'):
    cache_dir = pathlib.Path(cache_dir)

    sp_model = cache_dir / 'spm.model'
    if not sp_model.is_file():
        sp_model = None
    sp_vocab = cache_dir / 'spm.vocab'
    if not sp_vocab.is_file():
        sp_vocab = None
    processor = SPProcessor2(
        pre_rules=pre_rules,
        post_rules=post_rules,
        mark_fields=True,
        vocab_sz=vocab_size,
        sp_model=sp_model,
        sp_vocab=sp_vocab,
        lang=lang,
        tmp_dir=cache_dir.absolute()  # absolute make sure that dataset path is not added as prefix
    )
    return {'processor': processor}

# temporary loading function as from_df does not support processors
def make_data_bunch_from_df(cls, path: PathOrStr, train_df: DataFrame, valid_df: DataFrame,
            tokenizer: Tokenizer = None, vocab: Vocab = None, classes: Collection[str] = None,
            text_cols: IntsOrStrs = 1,
            label_cols: IntsOrStrs = 0, label_delim: str = None, chunksize: int = 10000,
            max_vocab: int = 60000,
            min_freq: int = 2, mark_fields: bool = False, include_bos: bool = True,
            include_eos: bool = False, processor=None, **kwargs) -> DataBunch:
    "Create a `TextDataBunch` from DataFrames. `kwargs` are passed to the dataloader creation."
    assert processor is None or tokenizer is None, "Processor and tokenizer are mutually exclusive."

    if processor is None:
        processor = fastai.text.data._get_processor(tokenizer=tokenizer, vocab=vocab, chunksize=chunksize, max_vocab=max_vocab,
                                   min_freq=min_freq, mark_fields=mark_fields,
                                   include_bos=include_bos, include_eos=include_eos)

    if classes is None and is_listy(label_cols) and len(label_cols) > 1: classes = label_cols
    src = ItemLists(path, TextList.from_df(train_df, path, cols=text_cols, processor=processor),
                    TextList.from_df(valid_df, path, cols=text_cols, processor=processor))
    if cls == TextLMDataBunch:
        src = src.label_for_lm()
    else:
        if label_delim is not None:
            src = src.label_from_df(cols=label_cols, classes=classes, label_delim=label_delim)
        else:
            src = src.label_from_df(cols=label_cols, classes=classes)
    return src.databunch(**kwargs)