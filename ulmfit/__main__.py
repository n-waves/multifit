import gc
import os
import pprint
import shutil
from collections import OrderedDict
from functools import wraps

import fire
from .pretrain_lm import LMHyperParams
from .train_clas import CLSHyperParams
from pathlib import Path
from string import Template

class FireView:
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

def get_lang_from_dataset_path(ds):
    lang,*_ = ds.name.split("-")
    if len(lang) == 2:
        return lang
    return "en"

def get_dataset_path(p, dataset_template):
    ds = [x for x in p.parents if x.name == "models"][0].parent
    lang = get_lang_from_dataset_path(ds)
    for ds_path in ds.parent.glob(Template(dataset_template).substitute(lang=lang, ds_name=ds.name)):
        yield lang, ds_path

class ULMFiT:
    @wraps(LMHyperParams)
    def lm(self, dataset_path,  **changes):
        changes['dataset_path'] = dataset_path
        params = LMHyperParams(**changes)
        return FireView(train=params.train_lm)

    lm2 = LMHyperParams
    @wraps(CLSHyperParams)
    def cls(self, dataset_path, base_lm_path, **changes):
        params = CLSHyperParams.from_lm(dataset_path, base_lm_path, **changes)
        return FireView(train=params.train_cls, validate_cls=params.validate_cls)

    def eval(self, glob="mldoc/*-1/models/sp30k/lstm_nl4.m", dataset_template='${lang}-1', name="tmp-100", cuda_id=0, **trn_params):
        results = OrderedDict()
        for base_model in sorted(Path("data").glob(glob)):
            for lang, dataset_path in sorted(get_dataset_path(base_model, dataset_template)):
                params = CLSHyperParams.from_lm(dataset_path, base_model, lang=lang, name=name, cuda_id=cuda_id)
                key = str(params.model_dir.relative_to(Path.cwd()))
                if params.model_dir.exists():
                    print("Evaluating previously trained model")
                    results[key] = params.validate_cls()[1]
                else:
                    print("Training")
                    results[key] = params.train_cls(num_lm_epochs=0, **trn_params)[1]
                del params
                gc.collect()

        pprint.pprint(results)

    def remove_lm_saves(self):
        for lm_save in Path("data").glob("**/lm_*.pth"):
            num = lm_save.stem.split("_")[-1]
            if not num.isdigit():
                continue
            if int(num) not in [5, 10, 15]:
                print("rm ", lm_save)
                os.remove(lm_save)

# python -m ulmfit cls --dataset-path data/mldoc/de-1-laser  --base-lm-path data/mldoc/de-1/models/sp30k/lstm_nl4.m  --lang=de --name 'nl4' --cuda-id=1 - train 0 --bs 40 --num-cls-epochs=2

if __name__ == '__main__':
    fire.Fire(ULMFiT())
