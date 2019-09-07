import pathlib
from typing import Collection
from pandas import DataFrame
from sacremoses import MosesTokenizer
import fastai
from fastai.basic_data import DataBunch
from fastai.core import ListRules, PathOrStr, IntsOrStrs, is_listy
from fastai.data_block import ItemLists
from fastai.text import Tokenizer, Vocab, SPProcessor, TextList, TextLMDataBunch


class MosesPreprocessingFunc():
    def __init__(self, lang: str):
        self.mt = MosesTokenizer(lang)

    def __call__(self, t: str) -> str:
        return self.mt.tokenize(t, return_str=True, escape=True)

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