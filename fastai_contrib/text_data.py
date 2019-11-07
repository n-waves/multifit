import pathlib
from typing import Collection
from pandas import DataFrame
from sacremoses import MosesTokenizer
import fastai
from fastai.basic_data import DataBunch
from fastai.core import ListRules, PathOrStr, IntsOrStrs, is_listy
from fastai.data_block import ItemLists
from fastai.text import *


class MosesPreprocessingFunc():
    def __init__(self, lang: str):
        self.mt = MosesTokenizer(lang)

    def __call__(self, t: str) -> str:
        return self.mt.tokenize(t, return_str=True, escape=True)

try:
    from fastai.text import SPProcessor
except ImportError:

    def _join_texts(texts:Collection[str], mark_fields:bool=False, include_bos:bool=True, include_eos:bool=False):
        if not isinstance(texts, np.ndarray): texts = np.array(texts)
        if is1d(texts): texts = texts[:,None]
        df = pd.DataFrame({i:texts[:,i] for i in range(texts.shape[1])})
        bos_tok = f'{BOS} ' if include_bos else ''
        text_col = f'{bos_tok}{FLD} {1} ' + df[0].astype(str) if mark_fields else f'{bos_tok}' + df[0].astype(str)
        for i in range(1,len(df.columns)):
            text_col += (f' {FLD} {i+1} ' if mark_fields else ' ') + df[i].astype(str)
        if include_eos: text_col = text_col + f' {EOS}'
        return text_col.values

    def apply_rules(text, pre_rules=None, post_rules=None):
        "Apply `pre_rules` and `post_rules` to `text`"
        text = text.strip(' ')
        for r in ifnone(pre_rules, defaults.text_pre_rules): text = r(text)
        toks = text.split()
        for r in ifnone(post_rules, defaults.text_post_rules): toks = r(toks)
        return ' '.join(toks)

    def get_default_size(texts, max_vocab_sz):
        "Either max_vocab_sz or one quarter of the number of unique words in `texts`"
        cnt = Counter()
        for t in texts:
            cnt.update(t.split())
            if len(cnt)//4 > max_vocab_sz: return max_vocab_sz
        res = len(cnt)//4
        while res%8 != 0: res+=1
        return res

    full_char_coverage_langs = ["bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr", "ga", "hr", "hu",
                           "it","lt","lv","mt","nl","pl","pt","ro","sk","sl","sv"] # all European langs

    def train_sentencepiece(texts:Collection[str], path:PathOrStr, pre_rules: ListRules=None, post_rules:ListRules=None,
        vocab_sz:int=None, max_vocab_sz:int=30000, model_type:str='unigram', max_sentence_len:int=20480, lang='en',
        char_coverage=None, tmp_dir='tmp', enc='utf8'):
        "Train a sentencepiece tokenizer on `texts` and save it in `path/tmp_dir`"
        from sentencepiece import SentencePieceTrainer
        cache_dir = Path(path)/tmp_dir
        os.makedirs(cache_dir, exist_ok=True)
        if vocab_sz is None: vocab_sz=get_default_size(texts, max_vocab_sz)
        raw_text_path = cache_dir / 'all_text.out'
        with open(raw_text_path, 'w', encoding=enc) as f: f.write("\n".join(texts))
        spec_tokens = ['\u2581'+s for s in defaults.text_spec_tok]
        SentencePieceTrainer.Train(" ".join([
            f"--input={raw_text_path} --max_sentence_length={max_sentence_len}",
            f"--character_coverage={ifnone(char_coverage, 0.99999 if lang in full_char_coverage_langs else 0.9998)}",
            f"--unk_id={len(defaults.text_spec_tok)} --pad_id=-1 --bos_id=-1 --eos_id=-1",
            f"--user_defined_symbols={','.join(spec_tokens)}",
            f"--model_prefix={cache_dir/'spm'} --vocab_size={vocab_sz} --model_type={model_type}"]))
        raw_text_path.unlink()
        return cache_dir

    class SPProcessor(PreProcessor):
        "`PreProcessor` that tokenizes and numericalizes with `sentencepiece`"
        def __init__(self, ds:ItemList=None, pre_rules: ListRules=None, post_rules:ListRules=None, vocab_sz:int=None,
                     max_vocab_sz:int=30000, model_type:str='unigram', max_sentence_len:int=20480, lang='en',
                     char_coverage=None, tmp_dir='tmp', mark_fields:bool=False, include_bos:bool=True,
                     include_eos:bool=False, sp_model=None, sp_vocab=None, n_cpus:int=None, enc='utf8'):
            try: from sentencepiece import SentencePieceTrainer,SentencePieceProcessor
            except ImportError:
                raise Exception('sentencepiece module is missing: run `pip install sentencepiece`')
            self.pre_rules,self.post_rules,self.enc = pre_rules,post_rules,enc
            self.mark_fields,self.include_bos,self.include_eos = mark_fields,include_bos,include_eos
            self.sp_model,self.sp_vocab,self.n_cpus = sp_model,sp_vocab,ifnone(n_cpus,defaults.cpus)
            self.train_func = partial(train_sentencepiece, pre_rules=pre_rules, post_rules=post_rules, vocab_sz=vocab_sz,
                    max_vocab_sz=max_vocab_sz, model_type=model_type, max_sentence_len=max_sentence_len, lang=lang,
                    char_coverage=char_coverage, tmp_dir=tmp_dir, enc=enc)

        def process_one(self, item, join=True):
            if join: text = _join_texts([item], self.mark_fields, self.include_bos, self.include_eos)[0]
            text = apply_rules(text, pre_rules=self.pre_rules, post_rules=self.post_rules)
            return self._encode_batch([text])[0]

        def process(self, ds):
            ds.items = _join_texts(ds.items, self.mark_fields, self.include_bos, self.include_eos)
            ds.items = [apply_rules(t, pre_rules=self.pre_rules, post_rules=self.post_rules)
                        for t in progress_bar(ds.items, leave=False)]
            if self.sp_model is None or self.sp_vocab is None:
                cache_dir = self.train_func(ds.items, ds.path)
                self.sp_model,self.sp_vocab = cache_dir/'spm.model',cache_dir/'spm.vocab'
            if not getattr(self, 'vocab', False):
                with open(self.sp_vocab, 'r', encoding=self.enc) as f: self.vocab = Vocab([line.split('\t')[0] for line in f.readlines()])
            if self.n_cpus <= 1: ds.items = self._encode_batch(ds.items)
            else:
                with ProcessPoolExecutor(self.n_cpus) as e:
                    ds.items = np.array(sum(e.map(self._encode_batch, partition_by_cores(ds.items, self.n_cpus)), []))
            ds.vocab = self.vocab

        def _encode_batch(self, texts):
            from sentencepiece import SentencePieceProcessor
            tok = SentencePieceProcessor()
            tok.Load(str(self.sp_model))
            return [np.array(tok.EncodeAsIds(t)) for t in texts]

        @classmethod
        def load(cls, path:PathOrStr, tmp_dir:PathOrStr='tmp', name:str='spm'):
            cache_dir = Path(path)/tmp_dir
            return cls(sp_model=cache_dir/f'{name}.model', sp_vocab=cache_dir/f'{name}.vocab')

class SPProcessor2(SPProcessor):
    def process(self, ds):
        super().process(ds)
        ds.vocab.sp_model = self.sp_model
        ds.vocab.sp_vocab = self.sp_vocab

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