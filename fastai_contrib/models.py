from fastai.torch_core import *
from fastai.layers import *
from fastai.text.models import *


class BiLMCore(nn.Module):
    """
    AWD-LSTM/QRNN inspired by https://arxiv.org/abs/1708.02182.
    Inspired by https://github.com/allenai/allennlp/blob/master/allennlp/models/bidirectional_lm.py#L65
    """
    initrange=0.1

    def __init__(self, vocab_sz:int, emb_sz:int, n_hid:int, n_layers:int, pad_token:int, bidir:bool=False,
                 hidden_p:float=0.2, input_p:float=0.6, embed_p:float=0.1, weight_p:float=0.5, qrnn:bool=False):

        super().__init__()
        self.bs,self.qrnn,self.ndir = 1, qrnn,(2 if bidir else 1)
        self.emb_sz,self.n_hid,self.n_layers = emb_sz,n_hid,n_layers
        # embeddings are shared between forward and backward LMs
        self.encoder = nn.Embedding(vocab_sz, emb_sz, padding_idx=pad_token)
        self.encoder_dp = EmbeddingDropout(self.encoder, embed_p)
        if self.qrnn:
            #Using QRNN requires cupy: https://github.com/cupy/cupy
            from fastai.text.qrnn.qrnn import QRNNLayer

            def create_qrnn_layers():
                return [QRNNLayer(emb_sz if l == 0 else n_hid, (n_hid if l != n_layers - 1 else emb_sz)//self.ndir,
                        save_prev_x=True, zoneout=0, window=2 if l == 0 else 1, output_gate=True,
                        use_cuda=torch.cuda.is_available()) for l in range(n_layers)]
            self.forward_rnns = create_qrnn_layers()
            self.backward_rnns = create_qrnn_layers()
            for rnn in self.forward_rnns + self.backward_rnns:
                rnn.linear = WeightDropout(rnn.linear, weight_p, layer_names=['weight'])
        else:
            def create_lstm_layers():
                return [nn.LSTM(emb_sz if l == 0 else n_hid, (n_hid if l != n_layers - 1 else emb_sz)//self.ndir,
                                1, bidirectional=False) for l in range(n_layers)]
            self.forward_rnns = [WeightDropout(rnn, weight_p) for rnn in create_lstm_layers()]
            self.backward_rnns = [WeightDropout(rnn, weight_p) for rnn in create_lstm_layers()]
        self.forward_rnns = torch.nn.ModuleList(self.forward_rnns)
        self.backward_rnns = torch.nn.ModuleList(self.backward_rnns)
        self.encoder.weight.data.uniform_(-self.initrange, self.initrange)
        self.input_dp = RNNDropout(input_p)
        self.hidden_dps = nn.ModuleList([RNNDropout(hidden_p) for l in range(n_layers)])

    def forward(self, input:LongTensor)->Tuple[Tensor,Tensor]:
        sl,bs = input.size()
        if bs!=self.bs:
            self.bs=bs
            self.reset()
        raw_output = self.input_dp(self.encoder_dp(input))

        # TODO get reverse input and compute backward representation
        new_hidden,raw_outputs,outputs = [],[],[]
        for l, (rnn,hid_dp) in enumerate(zip(self.forward_rnns, self.hidden_dps)):
            raw_output, new_h = rnn(raw_output, self.hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != self.n_layers - 1: raw_output = hid_dp(raw_output)
            outputs.append(raw_output)
        self.hidden = to_detach(new_hidden)
        return raw_outputs, outputs

    def _one_hidden(self, l:int)->Tensor:
        "Return one hidden state."
        nh = (self.n_hid if l != self.n_layers - 1 else self.emb_sz)//self.ndir
        return self.weights.new(self.ndir, self.bs, nh).zero_()

    def reset(self):
        "Reset the hidden states."
        [r.reset() for r in self.forward_rnns if hasattr(r, 'reset')]
        [r.reset() for r in self.backward_rnns if hasattr(r, 'reset')]
        self.weights = next(self.parameters()).data
        if self.qrnn: self.hidden = [self._one_hidden(l) for l in range(self.n_layers)]
        else: self.hidden = [(self._one_hidden(l), self._one_hidden(l)) for l in range(self.n_layers)]


def get_bilm(vocab_sz:int, emb_sz:int, n_hid:int, n_layers:int, pad_token:int, tie_weights:bool=True,
                       qrnn:bool=False, bias:bool=True, bidir:bool=False, output_p:float=0.4, hidden_p:float=0.2, input_p:float=0.6,
                       embed_p:float=0.1, weight_p:float=0.5)->nn.Module:
    "Create a full AWD-LSTM."
    rnn_enc = BiLMCore(vocab_sz, emb_sz, n_hid=n_hid, n_layers=n_layers, pad_token=pad_token, qrnn=qrnn, bidir=bidir,
                 hidden_p=hidden_p, input_p=input_p, embed_p=embed_p, weight_p=weight_p)
    enc = rnn_enc.encoder if tie_weights else None
    return SequentialRNN(rnn_enc, LinearDecoder(vocab_sz, emb_sz, output_p, tie_encoder=enc, bias=bias))
