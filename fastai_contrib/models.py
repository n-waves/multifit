from fastai.torch_core import *
from fastai.layers import *
from fastai.text.models import *

#region New code

class BiLMModel(nn.Module):

    def __init__(self, fwd_lm:nn.Module, bwd_lm:nn.Module):
        super().__init__()
        self.fwd_lm = fwd_lm
        self.bwd_lm = bwd_lm

    def __getitem__(self, idx):
        return BiLMModel(self.fwd_lm[idx], self.bwd_lm[idx])

    def __len__(self):
        return len(self.fwd_lm)

    def stack(self, fwd_o, bwd_o):
        if is_listy(fwd_o):
            return [self.stack(f, b) for f,b in zip(fwd_o,bwd_o)]
        else:
            return torch.stack([fwd_o, bwd_o], dim=len(fwd_o.shape))

    def forward(self, input):
        if len(input.shape) == 3: # sl, bs, tracks
            f = input[..., 0]
            b = input[..., 1]
        elif len(input.shape) == 2: # sl, bs - support during classification mode
            f = input
            b = torch.flip(input, [0])
        else:
            raise AttributeError(f"Inorrect size of input, {input.shape}")
        fwd_o = self.fwd_lm(f)
        bwd_o = self.bwd_lm(b)

        return self.stack(fwd_o, bwd_o)

    def reset(self):
        "Reset the hidden states of underlaying lms."
        self.fwd_lm.reset()
        self.bwd_lm.reset()


class BiPoolingLinearClassifier(nn.Module):
    "Create a linear classifier with pooling."

    def __init__(self, layers:Collection[int], drops:Collection[float]):
        super().__init__()
        mod_layers = []
        activs = [nn.ReLU(inplace=True)] * (len(layers) - 2) + [None]
        for n_in,n_out,p,actn in zip(layers[:-1],layers[1:], drops, activs):
            mod_layers += bn_drop_lin(n_in, n_out, p=p, actn=actn)
        self.layers = nn.Sequential(*mod_layers)

    def pool(self, x:Tensor, bs:int, is_max:bool):
        "Pool the tensor along the seq_len dimension."
        f = F.adaptive_max_pool1d if is_max else F.adaptive_avg_pool1d
        return f(x.permute(1,2,0), (1,)).view(bs,-1)

    def forward(self, input:Tuple[Tensor,Tensor])->Tuple[Tensor,Tensor,Tensor]:
        raw_outputs, outputs = input
        output = outputs[-1]
        if len(output.size()) == 3:
            sl,bs,_ = output.size()
            avgpool = self.pool(output, bs, False)
            mxpool = self.pool(output, bs, True)
            x = torch.cat([output[-1], mxpool, avgpool], 1)
            x = self.layers(x)
            return x, raw_outputs, outputs
        elif len(output.size()) == 4:
            sl, bs, em_sz, passes = output.size()

            f_avgpool = self.pool(output[...,0], bs, False)
            f_mxpool = self.pool(output[...,0], bs, True)
            b_avgpool = self.pool(output[..., 1], bs, False)
            b_mxpool = self.pool(output[..., 1], bs, True)
            x = torch.cat([output[-1][..., 0], f_mxpool, f_avgpool,
                           output[-1][..., 1], b_mxpool, b_avgpool,], 1)
            x = self.layers(x)
            return x, raw_outputs, outputs


class AvgPoolingLinearClassifier(nn.Module):
    "Create a linear classifier with pooling."

    def __init__(self, layers:Collection[int], drops:Collection[float]):
        super().__init__()
        mod_layers = []
        activs = [nn.ReLU(inplace=True)] * (len(layers) - 2) + [None]
        for n_in,n_out,p,actn in zip(layers[:-1],layers[1:], drops, activs):
            mod_layers += bn_drop_lin(n_in, n_out, p=p, actn=actn)
        self.layers = nn.Sequential(*mod_layers)

    def pool(self, x:Tensor, bs:int, is_max:bool):
        "Pool the tensor along the seq_len dimension."
        f = F.adaptive_max_pool1d if is_max else F.adaptive_avg_pool1d
        return f(x.permute(1,2,0), (1,)).view(bs,-1)

    def forward(self, input:Tuple[Tensor,Tensor])->Tuple[Tensor,Tensor,Tensor]:
        raw_outputs, outputs = input
        output = outputs[-1]
        if len(output.size()) == 3:
            sl,bs,_ = output.size()
            avgpool = self.pool(output, bs, False)
            mxpool = self.pool(output, bs, True)
            x = torch.cat([output[-1], mxpool, avgpool], 1)
            x = self.layers(x)
            return x, raw_outputs, outputs
        elif len(output.size()) == 4:
            sl, bs, em_sz, passes = output.size()

            avgpool = (self.pool(output[...,0], bs, False) + self.pool(output[..., 1], bs, False))/2
            mxpool = (self.pool(output[...,0], bs, True) +self.pool(output[..., 1], bs, True))/2
            x = torch.cat([(output[-1][..., 0]+output[-1][..., 1])/2, mxpool, avgpool], 1)
            x = self.layers(x)
            return x, raw_outputs, outputs


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

def get_birnn_classifier(bptt:int, max_seq:int, n_class:int, vocab_sz:int, emb_sz:int, n_hid:int, n_layers:int,
                       pad_token:int, layers:Collection[int], drops:Collection[float], bidir:bool=False, qrnn:bool=False,
                       hidden_p:float=0.2, input_p:float=0.6, embed_p:float=0.1, weight_p:float=0.5)->nn.Module:
    "Create a RNN classifier model."
    fwd_rnn_enc = MultiBatchRNNCore(bptt, max_seq, vocab_sz, emb_sz, n_hid, n_layers, pad_token=pad_token, bidir=bidir,
                      qrnn=qrnn, hidden_p=hidden_p, input_p=input_p, embed_p=embed_p, weight_p=weight_p)
    bwd_rnn_enc = MultiBatchRNNCore(bptt, max_seq, vocab_sz, emb_sz, n_hid, n_layers, pad_token=pad_token, bidir=bidir,
                                qrnn=qrnn, hidden_p=hidden_p, input_p=input_p, embed_p=embed_p, weight_p=weight_p)

    model = SequentialRNN(BiLMModel(fwd_rnn_enc, bwd_rnn_enc), AvgPoolingLinearClassifier(layers, drops))
    model.reset()
    return model

#endregion