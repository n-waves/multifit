from fastai import GradientClipping, accuracy
from fastai.callbacks import *
from fastai.basic_data import *
from fastai.datasets import untar_data
from fastai_contrib.models import get_bilm, get_rnn_classifier, get_birnn_classifier
from fastai.text.learner import *

#region New code

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

def bilm_text_classifier_learner(data: DataBunch, bptt: int = 70, max_len: int = 70 * 20, emb_sz: int = 400,
                            nh: int = 1150, nl: int = 3,
                            lin_ftrs: Collection[int] = None, ps: Collection[float] = None, pad_token: int = 1,
                            drop_mult: float = 1., qrnn: bool = False, **kwargs) -> 'TextClassifierLearner':
    "Create a RNN classifier."
    dps = default_dropout['classifier'] * drop_mult
    if lin_ftrs is None: lin_ftrs = [50]
    if ps is None:  ps = [0.1]
    ds = data.train_ds
    vocab_size, n_class = len(data.vocab.itos), data.c
    layers = [emb_sz * 3] + lin_ftrs + [n_class]
    ps = [dps[4]] + ps
    model = get_birnn_classifier(bptt, max_len, n_class, vocab_size, emb_sz, nh, nl, pad_token,
                               layers, ps, input_p=dps[0], weight_p=dps[1], embed_p=dps[2], hidden_p=dps[3],
                               qrnn=qrnn)
    learn = RNNLearner(data, model, bptt, split_func=birnn_classifier_split, **kwargs)
    return learn

def bilm_split(model:nn.Module) -> List[nn.Module]:
    "Split a RNN `model` in groups for differential learning rates."

    return [f+b for f,b in zip(lm_split(model.fwd_lm),lm_split(model.bwd_lm))]

def birnn_classifier_split(model:nn.Module) -> List[nn.Module]:
    "Split a RNN `model` in groups for differential learning rates."
    f_rnn,b_rnn = model[0].fwd_lm,model[0].bwd_lm
    groups = [[f_rnn.encoder, f_rnn.encoder_dp,b_rnn.encoder, b_rnn.encoder_dp]]
    groups += [a for a in zip(f_rnn.rnns, f_rnn.hidden_dps, b_rnn.rnns, b_rnn.hidden_dps, )]
    groups.append([model[1]])
    return groups

def accuracy_fwd(input, targs):
    return accuracy(input[...,0], targs[...,0])

def accuracy_bwd(input, targs):
    return accuracy(input[...,1], targs[...,1])


#endregion
#region Modified fastai code

def convert_weights(wgts:Weights, stoi_wgts:Dict[str,int], itos_new:Collection[str]) -> Weights:
    "Convert the model weights to go with a new vocabulary."
    if 'fwd_lm.0.encoder.weight' in wgts: #todo share embedding matrix computation
        wgts = convert_weights_with_prefix(wgts, stoi_wgts, itos_new, prefix='fwd_lm.')
        return convert_weights_with_prefix(wgts, stoi_wgts, itos_new, prefix='bwd_lm.')
    else:
        return convert_weights_with_prefix(wgts, stoi_wgts, itos_new, prefix='')

def convert_weights_with_prefix(wgts:Weights, stoi_wgts:Dict[str,int], itos_new:Collection[str], prefix='') -> Weights:
    "Convert the model weights to go with a new vocabulary."
    dec_bias, enc_wgts = wgts[prefix+'1.decoder.bias'], wgts[prefix+'0.encoder.weight']
    bias_m, wgts_m = dec_bias.mean(0), enc_wgts.mean(0)
    new_w = enc_wgts.new_zeros((len(itos_new),enc_wgts.size(1))).zero_()
    new_b = dec_bias.new_zeros((len(itos_new),)).zero_()
    for i,w in enumerate(itos_new):
        r = stoi_wgts[w] if w in stoi_wgts else -1
        new_w[i] = enc_wgts[r] if r>=0 else wgts_m
        new_b[i] = dec_bias[r] if r>=0 else bias_m
    wgts[prefix+'0.encoder.weight'] = new_w
    wgts[prefix+'0.encoder_dp.emb.weight'] = new_w.clone()
    wgts[prefix+'1.decoder.weight'] = new_w.clone()
    wgts[prefix+'1.decoder.bias'] = new_b
    return wgts

#endregion
#region Replace code in fastai

import fastai.text.learner
fastai.text.learner.convert_weights = convert_weights

#endregion