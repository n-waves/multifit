"NLP data loading pipeline. Supports csv, folders, and preprocessed data."
from fastai.text import *
from fastai.torch_core import *
from fastai.text.transform import *
from fastai.basic_data import *
from fastai.data_block import *

#region Modified fastai classes

LanguageModelType=Enum('LanguageModelType', 'FwdLM BwdLM BiLM')

class LanguageModelLoader(): # copy of the original LanguageModelLoader
    "Create a dataloader with bptt slightly changing."
    def __init__(self, dataset:LabelList, bs:int=64, bptt:int=70,
                 lm_type:LanguageModelType=LanguageModelType.FwdLM, shuffle:bool=False,
                 max_len:int=25):
        self.dataset,self.bs,self.bptt,self.lm_type,self.shuffle = dataset,bs,bptt,lm_type,shuffle
        self.first,self.i,self.iter = True,0,0
        self.n = len(np.concatenate(dataset.x.items)) // self.bs
        self.max_len,self.num_workers = max_len,0

    def __iter__(self):
        if getattr(self.dataset, 'item', None) is not None:
            yield LongTensor(getattr(self.dataset, 'item')).unsqueeze(1),LongTensor([0])
        idx = np.random.permutation(len(self.dataset)) if self.shuffle else range(len(self.dataset))
        data = self.batchify(np.concatenate([np.array(self.dataset.x.items[i], dtype=np.int) for i in idx]))

        pos, itr = 0,0
        while pos < self.n-1 and itr<len(self):
            if self.first and pos == 0: self.first,seq_len = False,self.bptt + self.max_len
            else:
                bptt = self.bptt if np.random.random() < 0.95 else self.bptt / 2.
                seq_len = max(5, int(np.random.normal(bptt, 5)))
                seq_len = min(seq_len, self.bptt + self.max_len)
            res = self.get_batch(data, pos, seq_len)
            pos += seq_len
            itr += 1
            yield res

    def __len__(self) -> int: return int(math.ceil((self.n-1) / self.bptt)) # so that it is always at least 1
    def __getattr__(self,k:str)->Any: return getattr(self.dataset, k)

    @property
    def batch_size(self):
        return self.bs

    @batch_size.setter
    def batch_size(self, v):
        self.bs = v

    def batchify(self, data:np.ndarray) -> LongTensor:
        "Split the corpus `data` in batches."
        nb = data.shape[0] // self.bs
        data = np.array(data[:nb*self.bs]).reshape(self.bs, -1).T
        if self.lm_type == LanguageModelType.BwdLM: data=data[::-1].copy()
        elif self.lm_type == LanguageModelType.BiLM: data = np.stack([data, data[::-1].copy()], axis=2)
        return LongTensor(data)

    def get_batch(self, data:LongTensor,  i:int, seq_len:int) -> Tuple[LongTensor, LongTensor]:
        "Create a batch at `i` of a given `seq_len`."
        seq_len = min(seq_len, len(data) - 1 - i)
        x = data[i:i+seq_len]
        y = data[i+1:i+1+seq_len].contiguous() # x & y has 2 elements on the last dimension
        y = y.view(-1, 2) if self.lm_type == LanguageModelType.BiLM else y.view(-1)
        return x,y

#endregion
#region Replaces fastai classes

import fastai.text.data
fastai.text.data.LanguageModelLoader = LanguageModelLoader # Replace original LanguageModelLoader with new verion

#endregion