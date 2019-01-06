from fastai.torch_core import *
from fastai.layers import *
from fastai.text.models import *

#region New code

class BiLMModel(nn.Module):

    def __init__(self, fwd_lm:nn.Module, bwd_lm:nn.Module, squash_bs_sl=False):
        super().__init__()
        self.fwd_lm = fwd_lm
        self.bwd_lm = bwd_lm
        self.squash_bs_sl = squash_bs_sl

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
            b = torch.flip(input, [1]) # todo test if we are duplicating the backward pass correctly
        else:
            raise AttributeError(f"Inorrect size of input, {input.shape}")
        fwd_o = self.fwd_lm(f)
        bwd_o = self.bwd_lm(b)

        outs = self.stack(fwd_o, bwd_o)
        if self.squash_bs_sl:
            o = outs[0]
            o = o.view(o.shape[0]*o.shape[1],o.shape[2],o.shape[3])
            outs[0] = o
        return outs

    def reset(self):
        "Reset the hidden states of underlaying lms."
        self.fwd_lm.reset()
        self.bwd_lm.reset()

class MultiBatchBiLMModel(BiLMModel):
    "Create a RNNCore module that can process a full sentence."

    def __init__(self, bptt:int, max_seq:int, *args, **kwargs):
        self.max_seq,self.bptt = max_seq,bptt
        super().__init__(*args, **kwargs)

    def concat(self, arrs:Collection[Tensor])->Tensor:
        "Concatenate the `arrs` along the batch dimension."
        return [torch.cat([l[si] for l in arrs], dim=1) for si in range_of(arrs[0])]

    def forward(self, input:LongTensor)->Tuple[Tensor,Tensor]:
        bs,sl = input.size()
        self.reset()
        raw_outputs, outputs = [],[]
        for i in range(0, sl, self.bptt):
            r, o = super().forward(input[:,i: min(i+self.bptt, sl)])
            if i>(sl-self.max_seq):
                raw_outputs.append(r)
                outputs.append(o)
        return self.concat(raw_outputs), self.concat(outputs)

class BiAttentionPoolingClassifier(nn.Module):
    r" [WIP] BiLM Pooling with self attention"

    def __init__(self, layers:Collection[int], drops:Collection[float]):
        super().__init__()
        mod_layers = []
        activs = [nn.ReLU(inplace=True)] * (len(layers) - 2) + [None]
        for n_in,n_out,p,actn in zip(layers[:-1],layers[1:], drops, activs):
            mod_layers += bn_drop_lin(n_in, n_out, p=p, actn=actn)
        self.self_attn = MultiHeadAttention(n_head=8, d_model=1, d_k=64, d_v=64, dropout=0.1)
        self.layers = nn.Sequential(*mod_layers)

    def pool(self, x:Tensor, bs:int, is_max:bool):
        "Pool the tensor along the seq_len dimension."
        f = F.adaptive_max_pool1d if is_max else F.adaptive_avg_pool1d
        return f(x.permute(1,2,0), (1,)).view(bs,-1)
    
    def forward(self, input:Tuple[Tensor,Tensor])->Tuple[Tensor,Tensor,Tensor]:
        raw_outputs, outputs = input
        output = outputs[-1]
        assert len(output.size()) == 4, 'Expected input dimension 4'
        sl, bs, em_sz, passes = output.size()

        f_avgpool = self.pool(output[...,0], bs, False)
        f_mxpool = self.pool(output[...,0], bs, True)
        b_avgpool = self.pool(output[..., 1], bs, False)
        b_mxpool = self.pool(output[..., 1], bs, True)
        x = torch.cat([output[-1][..., 0], f_mxpool, f_avgpool,
                        output[-1][..., 1], b_mxpool, b_avgpool,], 1)
        x = x.unsqueeze(-1)
        x, _ = self.self_attn(x, x, x)
        
        x = self.layers(x)
        return x, raw_outputs, outputs

class ScaledDotProductAttention(nn.Module):
    r""" 
    Scaled Dot-Product Attention
    based on: https://github.com/jadore801120/attention-is-all-you-need-pytorch
    """

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn

class MultiHeadAttention(nn.Module):
    r""" 
    Multi-Head Attention module
    based on: https://github.com/jadore801120/attention-is-all-you-need-pytorch
    """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()
        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        x, attn = self.attention(q, k, v)

        x = x.view(n_head, sz_b, len_q, d_v)
        x = x.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        x = self.dropout(self.fc(x))
        x = self.layer_norm(x + residual)

        return x, attn    

class BiPoolingLinearClassifier(PoolingLinearClassifier):
    "Create a linear classifier with pooling."

    def forward(self, input:Tuple[Tensor,Tensor])->Tuple[Tensor,Tensor,Tensor]:
        raw_outputs, outputs = input
        output = outputs[-1]
        if len(output.size()) == 3:
            return super().forward(input)
        elif len(output.size()) == 4:
            bs, sl, em_sz, passes = output.size()

            f_avgpool = self.pool(output[...,0], bs, False)
            f_mxpool = self.pool(output[...,0], bs, True)
            b_avgpool = self.pool(output[..., 1], bs, False)
            b_mxpool = self.pool(output[..., 1], bs, True)
            x = torch.cat([output[:,-1,..., 0], f_mxpool, f_avgpool,
                           output[:,-1,..., 1], b_mxpool, b_avgpool,], 1)
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
        bwd_lm=SequentialRNN(bwd_rnn_enc, LinearDecoder(vocab_sz, emb_sz, output_p, tie_encoder=enc, bias=bias)),
        squash_bs_sl=True)

def get_birnn_classifier(bptt:int, max_seq:int, n_class:int, vocab_sz:int, emb_sz:int, n_hid:int, n_layers:int,
                       pad_token:int, layers:Collection[int], drops:Collection[float], bidir:bool=False, qrnn:bool=False,
                       hidden_p:float=0.2, input_p:float=0.6, embed_p:float=0.1, weight_p:float=0.5, bicls_head:str='BiPoolingLinearClassifier')->nn.Module:
    "Create a RNN classifier model."
    fwd_rnn_enc = MultiBatchRNNCore(bptt, max_seq, vocab_sz, emb_sz, n_hid, n_layers, pad_token=pad_token, bidir=bidir,
                      qrnn=qrnn, hidden_p=hidden_p, input_p=input_p, embed_p=embed_p, weight_p=weight_p)
    bwd_rnn_enc = MultiBatchRNNCore(bptt, max_seq, vocab_sz, emb_sz, n_hid, n_layers, pad_token=pad_token, bidir=bidir,
                                qrnn=qrnn, hidden_p=hidden_p, input_p=input_p, embed_p=embed_p, weight_p=weight_p)

    head = BiPoolingLinearClassifier
    if bicls_head == 'BiPoolingLinearClassifier': head = BiPoolingLinearClassifier
    elif bicls_head == 'AvgPoolingLinearClassifier': head = AvgPoolingLinearClassifier
    elif bicls_head == 'BiAttentionPoolingClassifier': head = BiAttentionPoolingClassifier

    model = SequentialRNN(BiLMModel(fwd_rnn_enc, bwd_rnn_enc), head(layers, drops))
    model.reset()
    return model

#endregion
