"""
Utility methods for data processing.
"""
import pathlib
import shutil
import tarfile
from typing import Dict, Tuple, List

import fire
from sacremoses import MosesTokenizer

from fastai.text import *

EOS = 'xxeos' # fastai does not use eos, but we do
SEP = 'xxsep' # special separator token for NLI

def replace_std_toks(x:str) -> str:
    "Replace standard token names with fastai supported tokens"
    # We change tokens to f'xx{token_name}' as it is not split by Moses tokenizer,
    # while f'<{token_name}>' is being split to: '<' f'{token_name}' '>'
    return x.replace('<unk>', UNK).replace('<bos>', BOS).replace('<eos>', EOS)

PAD_TOKEN_ID = 1
IMDB, XNLI, TRN, VAL, TST, EN = 'imdb', 'xnli', 'train', 'val', 'test', 'en'
DATASETS = ['imdb', 'xnli']
XNLI_PATHS = {
    TRN: 'XNLI-MT-1.0/multinli/multinli.train.%s.tsv',
    VAL: 'XNLI-1.0/xnli.dev.tsv',
    TST: 'XNLI-1.0/xnli.test.tsv'
}

CLASSES = ['neg', 'pos', 'unsup']

number_match_re = re.compile(r'^([0-9]+[,.]?)+$')
number_split_re = re.compile(r'([,.])')

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
                       "it","lt","lv","mt","nl","pl","pt","ro","sk","sl","sv"] # all European langs

def get_sentencepiece(cache_dir:PathOrStr, load_text, pre_rules: ListRules=None, post_rules:ListRules=None,
                      vocab_size:int=30000, model_type:str='unigram', input_sentence_size:int=1E7, lang='en'):
    try:
        import sentencepiece as spm
    except ImportError:
        raise Exception('sentencepiece module is missing: run `pip install sentencepiece`')

    cache_dir = pathlib.Path(cache_dir)
    pre_rules = pre_rules if pre_rules is not None else defaults.text_pre_rules
    post_rules = post_rules if post_rules is not None else defaults.text_post_rules

    special_cases = defaults.text_spec_tok
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

        char_coverage = 1 if lang in full_char_coverage_langs else 0.99

        sp_params = [
            f"--input={raw_text_path}",
            f"--character_coverage={char_coverage}",
            f"--unk_id={len(defaults.text_spec_tok)}",
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


def clear_cache_directory(path:PathOrStr, cache_name:str='tmp'):
    path = pathlib.Path(path)
    shutil.rmtree(path / cache_name)

def get_texts(path):
    texts, labels = [],[]
    for idx, label in enumerate(CLASSES):
        for fname in (path/label).glob('*.*'):
            texts.append(fname.open('r', encoding='utf-8').read())
            labels.append(idx)
    return np.array(texts), np.array(labels)


def ensure_paths_exists(*paths, message="One or more required files cannot be found."):
    error = False
    for path in paths:
        if not path.exists():
            print(f'Error: {path} does not exist.')
            error = True
    if error:
        raise FileNotFoundError(message)

def get_data_folder() -> Path:
    """
    return data folder to use for future processing
    """
    return (pathlib.Path(__file__).parent.parent / "data")

def get_scripts_folder():
    """
    return data folder to use for future processing
    """
    return (pathlib.Path(__file__).parent.parent)

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
    dir_path = pathlib.Path(file_path.parent / 'aclImdb').resolve()
    assert tarfile.is_tarfile(file_path), "this is not a valid targz file"

    if not dir_path.exists():
        print(f"Extracting {file_path} to {dir_path}. This may take a long time...")
        tgz_file = tarfile.open(file_path)
        tgz_file.extractall(path=dir_path.parent) # the aclImdb.tgz has aclImdb dir packed
        assert dir_path.exists()
        print(f"Extracted to {dir_path}")

    CLAS_PATH = dir_path.parent
    CLAS_PATH.mkdir(exist_ok=True)

    LM_PATH = dir_path.parent /'imdb_lm'
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
    df_trn[df_trn['labels'] == 2].to_csv(CLAS_PATH / 'unsup.csv', header=False, index=False)
    (CLAS_PATH / 'classes.txt').open('w', encoding='utf-8').writelines(f'{o}\n' for o in CLASSES)

def read_xnli(dir_path, lang, split, spm_path=None) -> Tuple[List[List[str]], List[str]]:
    """
    Reads XNLI data.
    :param dir_path: the path to the xnli folder
    :param lang: the language
    :param split: the split of the data that should be read (train, test, val)
    :param spm_path: path to sentencepiece model
    :return: a tuple consisting of a list of lists of tokens and a list of labels
    """
    file_path = XNLI_PATHS[split]
    if split == TRN:
        file_path = file_path % lang
    elif lang == EN:
        file_name = 'xnli.dev.en.tsv' if split == VAL else 'xnli.test.en.tsv'
        file_path = f'XNLI-MT-1.0/xnli/{file_name}'
    file_path = dir_path / file_path
    
    if spm_path is not None:
        tokenizer = SentencePieceTokenizer(spm_path,
                                           use_moses=False,
                                           lang=lang)
        tok = tokenizer.tok_fun_with_sp(lang)
        tokenize = lambda x: tokenizer.process_text(x, tok)
        print("WARNING: Sentence Piece is not tested on XNLI yet")
    else:
        tokenize = lambda x: x.split(' ')

    toks, lbls = [], []
    print(f'Reading {file_path}...')
    with open(file_path, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        for i, row in enumerate(reader):
            if i == 0:  # skip the header
                continue
            # the examples are already tokenized with Moses
            if split == TRN:
                premise, hypo, label = row
            else:
                ex_lang = row[0]
                if ex_lang != lang:
                    continue
                premise, hypo, label = row[-3], row[-2], row[1]
            
            # TODO add BOS
            premise_toks = tokenize(premise) + [EOS]
            hypo_toks = tokenize(hypo) + [EOS]

            toks.append(premise_toks + [SEP] + hypo_toks)
            lbls.append(label)
    return toks, lbls


def read_clas_data(dir_path, dataset, lang) -> Tuple[Dict[str, List[List[str]]], Dict[str, List[str]]]:
    """
    Read the dataset from the classification datasets and tokenize them.
    :param dir_path: the path to the dataset
    :param dataset: the name of the dataset
    :param lang: the language
    :return: a tuple consisting of:
             1. a dictionary mapping splits to a list of lists of tokens
             2. a dictionary mapping splits to a list of labels
    """
    processors = {
        'xnli': read_xnli
    }
    processor = processors[dataset]

    toks, lbls = {}, {}
    toks[TRN], lbls[TRN] = processor(dir_path, lang, TRN)
    toks[TST], lbls[TST] = processor(dir_path, lang, TST)

    if dataset == IMDB:
        # for IMDb, we need to split off a separate validation set
        # note that we train and fine-tune ULMFiT on the full training set in the paper
        # to do this, we can just keep the training set the same
        val_len = max(int(len(toks[TRN]) * 0.1), 2) # fastai does not work with validation set of size 1
        trn_len = len(toks[TRN]) - val_len
        toks[TRN], toks[VAL] = toks[TRN][:trn_len], toks[TRN][trn_len:]
        lbls[TRN], lbls[VAL] = lbls[TRN][:trn_len], lbls[TRN][trn_len:]
    else:
        toks[VAL], lbls[VAL] = processor(dir_path, lang, VAL)
    return toks, lbls

def replace_number(token):
    """Replaces a number and returns a list of one or multiple tokens."""
    if number_match_re.match(token):
        return number_split_re.sub(r' @\1@ ', token)
    return token


def read_file(file_path, outname=None):
    """Reads a text file and writes it to a .csv."""
    with open(file_path, encoding='utf8') as f:
        text = f.readlines()
    df = pd.DataFrame(
        {'text': text, 'labels': np.zeros(len(text))},
        columns=['labels', 'text'])
    if outname is not None:
        df.to_csv(file_path.parent / f'{outname}.csv', header=False, index=False)
    return df



def read_whitespace_file(filepath):
    """Reads a file and prepares the tokens."""
    tokens = []
    with open(filepath, encoding='utf-8') as f:
        for line in f:
            # newlines are replaced with EOS
            tokens.append(line.split() + [EOS])
    return np.array(tokens)

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
