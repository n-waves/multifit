from dataclasses import dataclass
from ulmfit.train_clas import LMHyperParams
from fastai.text import TextLMDataBunch, TextClasDataBunch
from fastai.basic_train import LearnerCallback
from fastai.torch_core import PBar, Rank0Tensor
from torch import nn, Tensor

from typing import List, Collection, Any
from pathlib import Path
import pandas as pd
import fire
import random

@dataclass
class ParallelAlignmentCallback(LearnerCallback):
    "A `LearnerCallback` that adds parallel alignment between sentences."

    data_src:TextClasDataBunch
    data_tgt:TextClasDataBunch
    alpha:float=0.1

    def __post_init__(self):
        self.bs = self.data_src.bs
        self.loss = nn.CosineEmbeddingLoss(margin=0.5)
        self.ones = torch.cat((torch.ones(self.bs), -torch.ones(self.bs)))

    def pool(self, x:Tensor, bs:int, is_max:bool):
        "Pool the tensor along the seq_len dimension."
        f = F.adaptive_max_pool1d if is_max else F.adaptive_avg_pool1d
        return f(x.transpose(1,2), (1,)).view(bs,-1)

    def get_representation(batch):
        last_output = self.learn.model(batch)
        output = last_output[1][-1]
        bs,sl,_ = output.size()
        avgpool = self.pool(output, bs, False)
        mxpool = self.pool(output, bs, True)
        return torch.cat([output[:,-1], mxpool, avgpool], 1)

    def on_train_begin(self, pbar:PBar, metrics_names:Collection[str], **kwargs:Any)->None:
        self.counter = 0

    def on_backward_begin(self, last_loss:Rank0Tensor, last_input:Tensor, **kwargs):
        "Adjust the loss by adding similarity of parallel sentences"
        src_rep = self.get_representation(data_src.train_ds[self.counter])
        tgt_rep = self.get_representation(data_tgt.train_ds[self.counter])

        offset = -random.randrange(1, self.bs)

        src_rep = torch.cat((src_rep, src_rep))
        tgt_rep = torch.cat((tgt_rep, tgt_rep[range(offset, self.bs + offset)]))

        parallel_loss = self.alpha * self.loss(src_rep, tgt_rep, self.y)
        
        self.counter += 1
        self.counter %= len(data_src.train_ds)
        return last_loss + parallel_loss


@dataclass
class XLingualLMHyperParams(LMHyperParams):
    
    parallel_data_path: str=None
    parallel_data_bs: int=32
    src_lang: str=None
    tgt_lang: str=None

    def create_lm_learner(self, data_lm, dps=None, **kwargs):
        learner = super().create_lm_learner(data_lm, dps, **kwargs)
        if self.parallel_data_path is not None:
            src_trn_df = pd.read_csv(self.parallel_data_path / self.src_lang / 'train.csv', header=None)
            tgt_trn_df = pd.read_csv(self.parallel_data_path / self.tgt_lang / 'train.csv', header=None)
            bs = self.parallel_data_bs
            data_src = TextClasDataBunch.from_df(path=self.cache_dir, train_df=src_trn_df, lm_type=self.lm_type, bs=bs)
            data_tgt = TextClasDataBunch.from_df(path=self.cache_dir, train_df=tgt_trn_df, lm_type=self.lm_type, bs=bs)
            learner.callback_fns = [
                partial(ParallelAlignmentCallback, data_src=data_src, data_tgt=data_tgt)
            ] + learner.callback_fns

    def load_wiki_data(self, bs=70):
        trn_path = self.dataset_path / f'{self.lang}.wiki.train.tokens'
        val_path = self.dataset_path / f'{self.lang}.wiki.valid.tokens'
        tst_path = self.dataset_path / f'{self.lang}.wiki.test.tokens'
        for path_ in [trn_path, val_path, tst_path]:
            assert path_.exists(), f'Error: {path_} does not exist.'

        args = self.tokenzier_to_fastai_args(trn_data_loading_func=self.load_train_text, add_moses=False)
        try:
            data_lm = TextLMDataBunch.load(self.cache_dir, '.', lm_type=self.lm_type, bs=bs)
            print("Tokenized data loaded")
        except FileNotFoundError:
            print("Running tokenization")
            data_lm = TextLMDataBunch.from_df(path=self.cache_dir, train_df=read_wiki_articles(trn_path),
                                              valid_df=read_wiki_articles(val_path),
                                              classes=None, lm_type=self.lm_type, max_vocab=self.max_vocab,
                                              bs=bs, text_cols='texts', **args)
            data_lm.save('.')

        itos, stoi, trn_path = data_lm.vocab.itos, data_lm.vocab.stoi, data_lm.path
        print('Size of vocabulary:', len(itos))
        print('First 20 words in vocab:', data_lm.vocab.itos[:20])
        return data_lm

if __name__ == '__main__':
    fire.Fire(XLingualLMHyperParams)


# python -m ulmfit.XLingualLMHyperParams --dataset-path data/wiki/wikitext-103 --bidir=True --qrnn=True --nl=4 --tokenizer=sp --name 'nl4' --bs 120 --cuda-id 0 - train 10 --drop-mult=0 --bs 40

