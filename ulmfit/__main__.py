import gc
import os
import re
import pprint
import tarfile
import shutil
from collections import OrderedDict
from functools import wraps
import numpy as np
import pandas as pd
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
    pattern = Template(dataset_template).substitute(lang=lang, ds_name=ds.name)
    print(f"Searching for {pattern}, {ds.parent}")
    for ds_path in ds.parent.glob(pattern):
        yield lang, ds_path

name_re = re.compile("(lstm|qrnn)_(.*)_(lmseed-)?.*\.m")
def folder_name_to_model_name(folder_name):
    return name_re.match(folder_name).group(2)

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

    @wraps(CLSHyperParams)
    def load_cls(self, model_path, **changes):
        params = CLSHyperParams.from_json(model_path, **changes)
        return FireView(train=params.train_cls, validate_cls=params.validate_cls)


    def eval_noise_resistance(self, lang="de", size=1, prefix_name="", model="sp15k/qrnn_nl4.m",
                              num_cls_epochs=8, bs=18, lr_sched="1cycle", label_smoothing_eps=0.0, **kwargs):
        results= []
        for noise in range(0, 80, 5):
            print("Noise: ", noise)
            d = self.eval(glob=f"mldoc/{lang}-1/models/{model}",
                          name=f"nl4_{prefix_name}{noise}",
                          noise=noise/100,
                          dataset_template='${lang}-'+str(size),
                          num_cls_epochs=num_cls_epochs,
                          bs=bs,
                          lr_sched=lr_sched,
                          label_smoothing_eps=label_smoothing_eps,
                          **kwargs)

    def tar(self, model_path):
        data_dir = (Path.cwd()/"data").resolve()
        params = CLSHyperParams.from_json(model_path)
        name = str(params.dataset_dir.resolve().relative_to(data_dir)).replace("/", "-")

        tar_name = f"models/{name}-{params.tokenizer_prefix}-{params.model_name}.tar"
        print("Storing model in", tar_name)
        with tarfile.open(tar_name, mode="w") as tar:
            for g in map(params.model_dir.glob, ['*_best.pth', 'info.json', '../spm.*', '../itos.*',]):
                for f in g:
                    dest = f.resolve().relative_to(data_dir.parent)
                    print("Adding", f, dest)
                    tar.add(f, dest)


    def poleval19_full(self, base, name=None, num_lm_epochs=6, **kwargs):
        clsbase = self.poleval19_init(base, num_lm_epochs=num_lm_epochs, **kwargs)
        self.poleval19_seeds(clsbase, seed_name='clsweightseed', **kwargs)
        self.poleval19_seeds(clsbase, seed_name='clstrainseed', **kwargs)

    def poleval19_init(self, base, name=None, lmseed=None, **kwargs):
        clstrainseed = clsweightseed = ftseed = lmseed = 0
        if "wiki" in base:
            lmtype = "wiki"
        elif "reddit" in base:
            lmtype = "reddit"
        else:
            raise AttributeError("unkown lm ty")

        if "seed0" in base:
            lmseed = 0
            print("Setting lmseed ", lmseed)
        elif "seed1" in base:
            lmseed = 1
            print("Setting lmseed ", lmseed)

        dataset_template=f"../hate/pl-10-{lmtype}"

        return self.poleval19_eval(glob=base,
                         name=name,
                         dataset_template=dataset_template,
                         lmseed=lmseed,
                         ftseed=ftseed,
                         clstrainseed=clstrainseed,
                         clsweightseed=clsweightseed,
                         **kwargs)


    def poleval19_seeds(self, base, seed_name='clsweightseed', model_num=10, **kwargs):
        name = folder_name_to_model_name(Path(base).name)
        for seed in range(0, model_num, 1):
            kwargs[seed_name] = seed
            print("Seed: ", seed_name, seed)
            self.poleval19_eval(glob=base, name=name, num_lm_epochs=0, **kwargs)

    def poleval19_eval(self, glob, name=None, num_lm_epochs=6, num_cls_epochs=8, bs=160, **kwargs):
        if name is None:
            name = f"ft{num_lm_epochs}_cl{num_cls_epochs}"
            print("Setting name to ", name)

        return self.eval(glob=glob,
                  name=name,
                  num_lm_epochs=num_lm_epochs,
                  num_cls_epochs=num_cls_epochs,
                  bs=bs,
                  lr_sched="1cycle",
                  **kwargs)

    def eval(self, glob="data/mldoc/*-1/models/sp30k/lstm_nl4.m", dataset_template='${ds_name}', name=None,
             num_lm_epochs=0, train=True, to_csv=None, return_df=False, label_smoothing_eps=0.0,
             lmseed=None, ftseed=None, clsweightseed=None, clstrainseed=None,
             skip_on_error=True, **trn_params):
        results = []
        model_args = {}
        last_model_dir = None
        if clsweightseed is not None:
            model_args["clsweightseed"] = clsweightseed
        if clstrainseed is not None:
            model_args['clstrainseed'] = clstrainseed
        if ftseed is not None:
            model_args['ftseed'] = ftseed
        if lmseed is not None:
            model_args['lmseed'] = lmseed
        data_dir = Path("data").absolute()
        glob=str(glob)
        if "data" not in glob and not glob.startswith("/"):
            glob = "data/"+glob
        for base_model in sorted(data_dir.parent.glob(glob)):
            print("Processing", base_model)
            for lang, dataset_path in sorted(get_dataset_path(base_model, dataset_template)):
                try:
                    _name = name
                    if name is None:
                        _name = folder_name_to_model_name(base_model.name)
                    params = CLSHyperParams.from_lm(dataset_path, base_model, lang=lang, name=_name, **model_args)
                    last_model_dir = params.model_dir.relative_to(data_dir.parent)
                    if (params.model_dir/"cls_best.pth").exists():
                        print("Evaluating previously trained model")
                        d = params.validate_cls(label_smoothing_eps=label_smoothing_eps, use_cache=True)
                    elif train:
                        print("Training")
                        d = params.train_cls(num_lm_epochs=num_lm_epochs, label_smoothing_eps=label_smoothing_eps, **trn_params)
                    else:
                        print("Skipping", (params.model_dir/"cls_best.pth"))
                        d  = None
                    if d is not None:
                        d['model_dir_parent'] = params.model_dir.relative_to(data_dir.parent).parent
                        d['model_name'] = params.model_name
                        np.save(params.model_dir / "results.npy", d)
                        results.append(d)
                    del params
                except Exception as e:
                    print("Error", e)
                    if not skip_on_error:
                        raise e
                gc.collect()
        df = pd.DataFrame.from_records(results)
        print(df)
        if to_csv is not None:
            print(f"Saving result to: {to_csv}")
            df.to_csv(to_csv)
        if return_df:
            return last_model_dir, df
        return last_model_dir

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
