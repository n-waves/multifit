from fastai import GradientClipping, accuracy
from fastai.callbacks import *
from fastai.basic_data import *
from fastai.datasets import untar_data
from fastai_contrib.models import get_bilm, get_rnn_classifier, get_birnn_classifier
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

# learner extensions
class RNNLearner(Learner):
    "Basic class for a Learner in RNN."
    def __init__(self, data:DataBunch, model:nn.Module, bptt:int=70, split_func:OptSplitFunc=None, clip:float=None,
                 adjust:bool=False, alpha:float=2., beta:float=1., **kwargs):
        super().__init__(data, model, **kwargs)
        self.callbacks.append(RNNTrainer(self, bptt, alpha=alpha, beta=beta, adjust=adjust))
        if clip: self.callback_fns.append(partial(GradientClipping, clip=clip))
        if split_func: self.split(split_func)
        self.metrics = [accuracy]

    def model_path(self, name:str):
        return self.path/self.model_dir/f'{name}.pth'

    def _get_encoder(self):
        return self.model.encoder if hasattr(self.model, 'encoder') else self.model[0]

    def save_encoder(self, name:str):
        "Save the encoder to `name` inside the model directory."
        torch.save(self._get_encoder().state_dict(), self.model_path(name))

    def load_encoder(self, name:str):
        "Load the encoder `name` from the model directory."
        self._get_encoder().load_state_dict(torch.load(self.model_path(name)))
        self.freeze()

    def load_pretrained(self, wgts_fname:str, itos_fname:str):
        "Load a pretrained model and adapts it to the data vocabulary."
        old_itos = pickle.load(open(itos_fname, 'rb'))
        old_stoi = {v:k for k,v in enumerate(old_itos)}
        wgts = torch.load(wgts_fname, map_location=lambda storage, loc: storage)
        wgts = convert_weights(wgts, old_stoi, self.data.train_ds.vocab.itos)
        self.model.load_state_dict(wgts)

    def get_preds(self, ds_type:DatasetType=DatasetType.Valid, with_loss:bool=False, n_batch:Optional[int]=None, pbar:Optional[PBar]=None,
                  ordered:bool=False) -> List[Tensor]:
        "Return predictions and targets on the valid, train, or test set, depending on `ds_type`."
        self.model.reset()
        preds = super().get_preds(ds_type=ds_type, with_loss=with_loss, n_batch=n_batch, pbar=pbar)
        if ordered and hasattr(self.dl(ds_type), 'sampler'):
            sampler = [i for i in self.dl(ds_type).sampler]
            reverse_sampler = np.argsort(sampler)
            preds[0] = preds[0][reverse_sampler,:] if preds[0].dim() > 1 else preds[0][reverse_sampler]
            preds[1] = preds[1][reverse_sampler,:] if preds[1].dim() > 1 else preds[1][reverse_sampler]
        return(preds)


def accuracy_fwd(input, targs):
    return accuracy(input[...,0], targs[...,0])


def accuracy_bwd(input, targs):
    return accuracy(input[...,1], targs[...,1])