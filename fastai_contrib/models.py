from fastai.torch_core import *
from fastai.layers import *
from fastai.text.models import *

class BiLMModel(nn.Module):

    def __init__(self, fwd_lm:nn.Module, bwd_lm:nn.Module):
        super().__init__()
        self.fwd_lm = fwd_lm
        self.bwd_lm = bwd_lm

    def forward(self, input):
        sl, bs, tracks = input.size()

        decoded = []
        raw_outputs = []
        outputs = []

        fwd_o = self.fwd_lm(input[..., 0])
        bwd_o = self.bwd_lm(input[..., 1])

        return torch.stack([fwd_o[0], bwd_o[0]], dim=2), (fwd_o[1]+bwd_o[1]), (fwd_o[2] + bwd_o[2])

    def reset(self):
        "Reset the hidden states of underlaying lms."
        self.fwd_lm.reset()
        self.bwd_lm.reset()

def get_bilm(vocab_sz:int, emb_sz:int, n_hid:int, n_layers:int, pad_token:int, tie_weights:bool=True,
                       qrnn:bool=False, bias:bool=True, bidir:bool=False, output_p:float=0.4, hidden_p:float=0.2, input_p:float=0.6,
                       embed_p:float=0.1, weight_p:float=0.5)->nn.Module:
    "Create a two AWD-LSTM one for each direction "
    fwd_rnn_enc = RNNCore(vocab_sz, emb_sz, n_hid=n_hid, n_layers=n_layers, pad_token=pad_token, qrnn=qrnn, bidir=bidir,
                          hidden_p=hidden_p, input_p=input_p, embed_p=embed_p, weight_p=weight_p)
    bwd_rnn_enc = RNNCore(vocab_sz, emb_sz, n_hid=n_hid, n_layers=n_layers, pad_token=pad_token, qrnn=qrnn, bidir=bidir,
                          hidden_p=hidden_p, input_p=input_p, embed_p=embed_p, weight_p=weight_p)
    enc = None
    if tie_weights:
        enc = fwd_rnn_enc.encoder
        fwd_rnn_enc.encoder.weight = enc.weight
        bwd_rnn_enc.encoder.weight = enc.weight

    return BiLMModel(
        fwd_lm=SequentialRNN(fwd_rnn_enc, LinearDecoder(vocab_sz, emb_sz, output_p, tie_encoder=enc, bias=bias)),
        bwd_lm=SequentialRNN(bwd_rnn_enc, LinearDecoder(vocab_sz, emb_sz, output_p, tie_encoder=enc, bias=bias)))