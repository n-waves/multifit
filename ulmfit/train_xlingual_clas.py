from dataclasses import dataclass
from ulmfit.train_clas import CLSHyperParams
from fastai.text import TextLMDataBunch, TextClasDataBunch

from typing import List
from pathlib import Path
import pandas as pd
import fire

@dataclass
class XLingualCLSHyperParams(CLSHyperParams):
    csv_name: str='train.csv'
    target_paths: List[str]=None
    parallel_data_path: str=None
    
    def __post_init__(self, *args, **kwargs):
        super().__post_init__(*args, **kwargs)
        self.target_paths = [] if self.target_paths is None else self.target_paths

    def load_cls_data(self, bs, force=False, use_test_for_validation=False, **kwargs):
        args = self.tokenzier_to_fastai_args(trn_data_loading_func=lambda: trn_df[1], add_moses=True)
        src_path = self.dataset_path
        csv_name = self.csv_name
        tgt_paths = [Path(tgt_path) for tgt_path in self.target_paths]
        mixed_csv = pd.read_csv(src_path / csv_name, header=None)
        for tgt_path in tgt_paths:
            mixed_csv = pd.concat([mixed_csv, pd.read_csv(tgt_path / csv_name, header=None)])

        xcvs_name = ('x_' + csv_name)
        mixed_csv.to_csv(src_path / xcvs_name, header=None, index=False)

        try:
            if force: raise FileNotFoundError("Forcing reloading of caches")
            data_lm = TextLMDataBunch.load(src_path, 'xlm', lm_type=self.lm_type, bs=bs)
            print(f"Tokenized data loaded, xlm.trn {len(data_lm.train_ds)}, xlm.val {len(data_lm.valid_ds)}")
        except FileNotFoundError:
            print(f"Running tokenization...")
            data_lm = TextLMDataBunch.from_csv(path=src_path, csv_name=xcvs_name, bs=bs, lm_type=self.lm_type, **kwargs, **args)
            print(f"Saving tokenized: cls.trn {len(data_lm.train_ds)}, cls.val {len(data_lm.valid_ds)}")
            data_lm.save('xlm')

        try:
            if force: raise FileNotFoundError("Forcing reloading of caches")
            data_cls = TextClasDataBunch.load(src_path, 'cls', bs=bs)
            print(f"Tokenized data loaded, cls.trn {len(data_cls.train_ds)}, cls.val {len(data_cls.valid_ds)}")
        except FileNotFoundError:
            args['vocab'] = data_lm.vocab  # make sure we use the same vocab for classifcation
            print(f"Running tokenization...")
            data_cls = TextClasDataBunch.from_csv(path=src_path, csv_name=csv_name, bs=bs, **kwargs, **args)
            
            print(f"Saving tokenized: cls.trn {len(data_cls.train_ds)}, cls.val {len(data_cls.valid_ds)}")
            data_cls.save('cls')

        print('Size of vocabulary:', len(data_lm.vocab.itos))
        print('First 20 words in vocab:', data_lm.vocab.itos[:20])
        return data_cls, data_lm

    def validate_cls(self, save_name='cls_last', bs=40):
        args = self.tokenzier_to_fastai_args(trn_data_loading_func=lambda: trn_df[1], add_moses=True)
        data_clas, data_lm = self.load_cls_data(bs, use_test_for_validation=True)
        data_eval = [
            TextClasDataBunch.from_csv(path=Path(tgt_path), csv_name=self.csv_name, **args)
            for tgt_path in self.target_paths
        ]

        for data in [data_clas] + data_eval:
            learn = self.create_cls_learner(data, drop_mult=0.1)
            learn.load(save_name)
            print(f"Loss and accuracy using ({save_name}) for dataset at {data.path}:", learn.validate())


if __name__ == '__main__':
    fire.Fire(XLingualCLSHyperParams)
