import pytest
from fastai import *
from fastai.text import *

pytestmark = pytest.mark.integration

print(sys.path)
import fastai_contrib.data as contrib_data

from fastai_contrib.learner import bilm_learner, accuracy_fwd, bilm_text_classifier_learner


def read_file(fname):
    texts = []
    with open(fname, 'r') as f:
        texts = f.readlines()
    labels = [0] * len(texts)
    df = pd.DataFrame({'labels':labels, 'texts':texts}, columns = ['labels', 'texts'])
    return df

def prep_human_numbers():
    path = untar_data(URLs.HUMAN_NUMBERS)
    df_trn = read_file(path/'train.txt')
    df_val = read_file(path/'valid.txt')
    return path, df_trn, df_val

def manual_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

@pytest.fixture(scope="module")
def learn():
    path, df_trn, df_val = prep_human_numbers()
    data = TextLMDataBunch.from_df(path, df_trn, df_val, tokenizer=Tokenizer(BaseTokenizer))
    learn = language_model_learner(data, emb_sz=100, nl=1, drop_mult=0.1)
    learn.fit_one_cycle(4, 5e-3)
    return learn

def text_df(n_labels):
    data = []
    texts = ["fast ai is a cool project", "hello world"]
    for ind, text in enumerate(texts):
        sample = {}
        for label in range(n_labels): sample[label] = ind%2
        sample["text"] = text
        data.append(sample)
    df = pd.DataFrame(data)
    return df

###################### NEW CODE

def test_val_loss(learn):
    assert learn.validate()[1] > 0.5


def test_bilm_classifier_loads_encoder():
    n_labels=2
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'tmp')
    os.makedirs(path)
    try:
        df = text_df(n_labels=1)
        lmdf = df#[["text"]]
        print(lmdf.head())
        lmdata = TextLMDataBunch.from_df(path, lmdf, lmdf, tokenizer=Tokenizer(BaseTokenizer),
                                       lm_type=contrib_data.LanguageModelType.BiLM)
        learn = bilm_learner(lmdata, emb_sz=100, nl=1, drop_mult=0.1, qrnn=False)
        learn.save_encoder("enc")
        data = TextClasDataBunch.from_df(path, train_df=df, valid_df=df, label_cols=list(range(n_labels)), text_cols=["text"])
        classifier = bilm_text_classifier_learner(data, emb_sz=100, nl=1, drop_mult=0.1, qrnn=False)
        print(last_layer(classifier.model), )
        classifier.load_encoder("enc")
        classifier.fit(1)
    finally:
        shutil.rmtree(path)


def test_bilm_lstm_can_be_trained():
    manual_seed()
    path, df_trn, df_val = prep_human_numbers()
    data = TextLMDataBunch.from_df(path, df_trn, df_val, tokenizer=Tokenizer(BaseTokenizer),
                                   lm_type = contrib_data.LanguageModelType.BiLM)

    learn = bilm_learner(data, emb_sz=100, nl=1, drop_mult=0.1, qrnn=False)
    learn.metrics = [accuracy_fwd]
    learn.fit_one_cycle(2, 5e-3)
    assert learn.validate()[1] > 0.3


def test_bwdlm_lstm_can_be_trained():
    manual_seed()
    path, df_trn, df_val = prep_human_numbers()
    data = TextLMDataBunch.from_df(path, df_trn, df_val, tokenizer=Tokenizer(BaseTokenizer),
                                   lm_type = contrib_data.LanguageModelType.BwdLM)

    learn = language_model_learner(data, emb_sz=100, nl=1, drop_mult=0.1, qrnn=False)
    learn.fit_one_cycle(2, 5e-3)
    assert learn.validate()[1] > 0.3
