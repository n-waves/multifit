import gc
import os
import pprint
import tarfile
import shutil
from collections import OrderedDict
from functools import wraps
import pandas as pd
import fire
from .pretrain_lm import LMHyperParams, json_save, json_load, np
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
    print(pattern)
    for ds_path in ds.parent.glob(pattern):
        yield lang, ds_path

class ULMFiT:
    @wraps(LMHyperParams)
    def lm(self, dataset_path,  **changes):
        changes['dataset_path'] = dataset_path
        params = LMHyperParams(**changes)
        return FireView(train=params.train_lm)

    lm2 = LMHyperParams
    @wraps(CLSHyperParams)
    def cls(self, dataset_path, base_lm_path=None, **changes):
        if base_lm_path is not None:
            params = CLSHyperParams.from_lm(dataset_path, base_lm_path, **changes)
        else:
            params = CLSHyperParams(dataset_path=dataset_path, **changes)
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
                          return_df=True,
                          **kwargs)
            val = d['tst_accuracy'][0]
            results.append((noise/100, val))
        df = pd.DataFrame(results, columns=["noise", "accuracy"])
        df.to_csv(f"noise_{lang}-{size}{prefix_name}.csv")
        print(df)

    def tar(self, model_path):
        data_dir = (Path.cwd()/"data").resolve()
        params = CLSHyperParams.from_json(model_path)
        name = str(params.dataset_dir.resolve().relative_to(data_dir)).replace("/", "-")

        tar_name = f"models/{name}-{params.tokenizer_prefix}-{params.model_name}.tar"
        print("Storing model in", tar_name)
        with tarfile.open(tar_name, mode="w") as tar:
            for g in map(params.model_dir.glob, ['*_best.pth', 'info.json', '../spm.*', '../itos.*',]):
                for f in g:
                    dest = f.resolve().relative_to(Path.cwd())
                    print("Adding", f, dest)
                    tar.add(f, dest)

    def eval(self, glob="mldoc/*-1/models/sp30k/lstm_nl4.m", dataset_template='${ds_name}', name=None,
             num_lm_epochs=0, cuda_id=0, train=True, to_csv=None, return_df=False, label_smoothing_eps=0.0,
             **trn_params):
        results = []


        def extract_agg(group):
            best = group.loc[group["val_accuracy"].idxmax()]["tst_accuracy"]
            best_name = group.loc[group["val_accuracy"].idxmax()]["n"]
            return pd.Series({'best': best* 100,
                              'max': group['tst_accuracy'].max()* 100,
                              'avg': group['tst_accuracy'].mean()* 100})
        def pivot_to_lang(df):
            df['ds'] = df['name'].str.extract(r'data/[a-z]*/([^/]*)/models')
            df['n'] = df['name'].str.extract(r'models/[^/]*/([^/]*).m')
            best = df.groupby('ds').apply(extract_agg)
            best = best.round(2)
            return best.T
        for base_model in sorted(Path("data").glob(glob)):
            print("Processing", base_model)
            for lang, dataset_path in sorted(get_dataset_path(base_model, dataset_template)):
                try:
                    _name = name
                    if name is None:
                        _name = base_model.name.replace(".m","").replace("lstm_","").replace("qrnn_","")
                    params = CLSHyperParams.from_lm(dataset_path, base_model, lang=lang, name=_name, cuda_id=cuda_id)
                    key = str(params.model_dir.relative_to(Path.cwd()))
                    if (params.model_dir / "results.npy").exists():
                        d = np.load(params.model_dir / "results.npy")
                        d = d.tolist() # magiacally convert to dict
                    elif (params.model_dir/"cls_best.pth").exists():
                        print("Evaluating previously trained model")
                        d = params.validate_cls(label_smoothing_eps=label_smoothing_eps)
                    elif train:
                        print("Training")
                        d = params.train_cls(num_lm_epochs=num_lm_epochs, label_smoothing_eps=label_smoothing_eps, **trn_params)
                    else:
                        print("Skipping", (params.model_dir/"cls_best.pth"))
                        d  = None
                    if d is not None:
                        d['name']=key
                        np.save(params.model_dir / "results.npy", d)
                        results.append(d)
                    del params
                except Exception as e:
                    print("Error", e)
                gc.collect()
        df = pd.DataFrame.from_records(results)
        print(df)
        print(pivot_to_lang(df))
        if to_csv is not None:
            print(f"Saving result to: {to_csv}")
            df.to_csv(to_csv)
        if return_df:
            return df

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
