import gc
import shutil
from functools import wraps

import fire
from .pretrain_lm import LMHyperParams
from .train_clas import CLSHyperParams
from pathlib import Path

class FireView:
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

def get_dataset_path(p):
    return [x for x in p.parents if x.name == "models"][0].parent

def get_lang_from_dataset_path(ds):
    lang,*_ = ds.name.split("-")
    if len(lang) == 2:
        return lang
    return "en"

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

    def eval(self, glob="mldoc/*-1/models/sp30k/lstm_nl4.m", name="tmp-100", cuda_id=0, **trn_params):
        results={}
        for base_model in Path("data").glob(glob):
            dataset_path = get_dataset_path(base_model)
            lang = get_lang_from_dataset_path(dataset_path)
            params = CLSHyperParams.from_lm(dataset_path, base_model, lang=lang, name=name, cuda_id=cuda_id)
            key = str(params.model_dir.relative_to(Path.cwd()))
            if params.model_dir.exists():
                print("Evaluating previously trained model")
                results[key] = params.validate_cls()[1]
            else:
                print("Training")
                results[key] = params.train_cls(num_lm_epochs=0, **trn_params)[1]
            params = None
            gc.collect()

        print(list(sorted(results.items())))

if __name__ == '__main__':
    fire.Fire(ULMFiT())