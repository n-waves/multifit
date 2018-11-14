from fastai.callbacks import *
from fastai.basic_data import *
from fastai.datasets import untar_data
from fastai_contrib.models import get_bilm
from fastai.text.learner import *


def bilm_learner(data:DataBunch, bptt:int=70, emb_sz:int=400, nh:int=1150, nl:int=3, pad_token:int=1,
                  drop_mult:float=1., tie_weights:bool=True, bias:bool=True, qrnn:bool=False, pretrained_model=None,
                  pretrained_fnames:OptStrTuple=None, **kwargs) -> 'LanguageLearner':
    "Create a `Learner` with a language model."
    dps = default_dropout['language'] * drop_mult
    vocab_size = len(data.vocab.itos)
    model = get_bilm(vocab_size, emb_sz, nh, nl, pad_token, input_p=dps[0], output_p=dps[1],
                weight_p=dps[2], embed_p=dps[3], hidden_p=dps[4], tie_weights=tie_weights, bias=bias, qrnn=qrnn)
    learn = LanguageLearner(data, model, bptt, split_func=bilm_split, **kwargs)
    if pretrained_model is not None:
        model_path = untar_data(pretrained_model, data=False)
        fnames = [list(model_path.glob(f'*.{ext}'))[0] for ext in ['pth', 'pkl']]
        learn.load_pretrained(*fnames)
        learn.freeze()
    if pretrained_fnames is not None:
        fnames = [learn.path/learn.model_dir/f'{fn}.{ext}' for fn,ext in zip(pretrained_fnames, ['pth', 'pkl'])]
        learn.load_pretrained(*fnames)
        learn.freeze()
    return learn


def bilm_split(model:nn.Module) -> List[nn.Module]:
    "Split a RNN `model` in groups for differential learning rates."

    return [f+b for f,b in zip(lm_split(model.fwd_lm),lm_split(model.bwd_lm))]
