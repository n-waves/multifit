from pathlib import Path

import torch
import dataclasses
from fastai.callbacks import CSVLogger, SaveModelCallback
from fastai.text import *

from multifit.datasets import ULMFiTDataset,ULMFiTTokenizer

CLS_BEST = 'cls_best'
LM_BEST = "lm_best"
ENC_BEST = "enc_best"


def detect_lang_from_dataset_path(dataset_path:Path):
    lang, *size = dataset_path.name.split('-')
    if lang == "wikitext":
        lang = "en"
    if len(lang) == 2:
        return lang
    return None


@dataclass
class Params:
    def replace_(self, _verbose_diff=False, **changes):
        for f in dataclasses.fields(self):
            if f.name in changes:
                v = changes[f.name]
                if f.type == Path and v is not None:
                    v = Path(v)
                orig = getattr(self, f.name)
                if orig != v and _verbose_diff:
                    print(f"{self.__class__.__name__} Replacing {f.name} '{orig}' with '{v}")
                setattr(self, f.name, v)
        return self


@dataclass
class ULMFiTArchitecture(Params):
    tokenizer_type: str = "f"
    max_vocab: int = 60000
    lang: str = None

    emb_sz: int = awd_lstm_lm_config['emb_sz']
    n_hid: int = awd_lstm_lm_config['n_hid']
    n_layers: int = awd_lstm_lm_config['n_layers']
    qrnn: bool = awd_lstm_lm_config['qrnn']

    def model_name(self, name=""):
        model_suffix = ''  # if self.lmseed is None else f'_lmseed-{self.lmseed}'
        model_prefix = 'qrnn' if self.qrnn else 'lstm'

        model_name = f"{model_prefix}_{name}{model_suffix}.m"
        return model_name

    def dataset_cache_suffix(self):
        tokenizer_prefix = f"{self.tokenizer_type}{self.max_vocab // 1000}k"
        return f'models/{tokenizer_prefix}'

    def dataset(self, dataset_path_or_object, tokenizer=None, **args):
        if hasattr(dataset_path_or_object, 'load_lm_databunch'):
            return dataset_path_or_object
        if dataset_path_or_object is None:
            return None
        ds_path = Path(dataset_path_or_object)
        cache_path = ds_path / self.dataset_cache_suffix()
        if tokenizer is not None:
            tokenizer.save(cache_path) # saving the tokenizer to the cache_path so that it can be reused later.
            # TODO add proper caching prefixed with tokenizer hash.
        tokenizer = self.new_tokenizer(cache_path)
        return ULMFiTDataset(dataset_path=ds_path, cache_path=cache_path, tokenizer=tokenizer, **args)

    def new_tokenizer(self, pretrained_path=None):
        "gets untrained tokenizer in that stores its data in tmp, use .save once trained"
        return ULMFiTTokenizer(arch=self, pretrained_path=pretrained_path)

def set_seed(seed, name):
    if seed is not None:
        print(f"Setting {name} seed to {seed}")
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)


def to_json_serializable(d):
    n = {}
    for k, v in d.items():
        if isinstance(v, dict):
            n[k] = to_json_serializable(v)
        elif isinstance(v, (float, int, str, list, tuple)):
            n[k] = v
        elif v is None:
            n[k] = v
        else:
            n[k] = str(v)
    return n


def rename_dict_keys(d, rename_func):
    for k in list(d.keys()):
        d[rename_func(k)] = d.pop(k)

def convert_old_models_keys_hook(state_dict, *_, **__):
    rename_dict_keys(state_dict, lambda k:
        k.replace('linear', 'layers.0.linear') if 'layers.0' not in k else k)

def convert_new_models_keys_hook(state_dict, *_, **__):
    rename_dict_keys(state_dict, lambda k: k.replace('layers.0.linear', 'linear'))

def patch_learner(learn):
    encoder = get_model(learn.model)[0]
    if hasattr(encoder, 'module'): encoder = encoder.module
    if hasattr(encoder.rnns[0], 'layers'):
        encoder._register_load_state_dict_pre_hook(convert_old_models_keys_hook)
        learn.model._register_load_state_dict_pre_hook(convert_old_models_keys_hook)
    else:
        encoder._register_load_state_dict_pre_hook(convert_new_models_keys_hook)
        learn.model._register_load_state_dict_pre_hook(convert_new_models_keys_hook)
    return learn

@dataclass
class ULMFiTTrainingCommand(Params):
    seed: int = 0
    name: str = None
    arch: ULMFiTArchitecture = field(repr=False, default=None)
    experiment_path: Path = None
    dataset_path: Path = None

    @property
    def model_name(self):
        return (self.name or self.arch.model_name()) + (
            "" if self.seed is None or self.seed == 0 or "seed" in self.name else f"seed{self.seed}")

    @property
    def info_json(self):
        return self.__class__.__name__.lower().replace("ulmfit", "") + ".json"

    def _set_dataset_(self, dataset_or_path, tokenizer):
        #TODO: refactor, this bit is unclear (set_dataset that does nothing when is None passed?)
        dataset_or_path = dataset_or_path or self.dataset_path or getattr(self, 'base', self).dataset_path
        dataset = self.arch.dataset(dataset_or_path, tokenizer=tokenizer)
        self.dataset_path = dataset.dataset_path
        return dataset

    @property
    def dataset(self):
        return self.arch.dataset(self.dataset_path, self.tokenizer)

    @property
    def tokenizer(self):
        if self.experiment_path is None:
            return None
        return ULMFiTTokenizer(arch=self.arch, pretrained_path=self.experiment_path)

    def save_paramters(self):
        params = dataclasses.asdict(self)
        base_exp_path = params.pop('base', {}).pop('experiment_path', None)
        params['base'] = base_exp_path
        exp_path = params.get('experiment_path', None)
        if exp_path:
            fn = self.info_json
            print("Saving dump to", exp_path / fn)
            json_str = json.dumps(to_json_serializable(params), indent=2)
            with (exp_path / fn).open("w") as f:
                f.write(json_str)
            return json_str

    def load_(self, experiment_path, tantetive=True, update_arch=True, silent=False):
        fn = experiment_path / self.info_json
        if not fn.exists():
            if not tantetive:
                warn(f"Unable to load experiment_path {experiment_path}")
            return False
        print(f"Loading {fn}")
        with fn.open('r') as f:
            d = json.load(f)
        base = d.pop('base', None)
        arch = d.pop('arch')
        if hasattr(self, 'base'):
            self.base.load_(Path(base), tantetive=True, update_arch=False, silent=silent)

        # compatiblity with older info.json formats where lang was not stored
        self.name = experiment_path.name                # V ./de-1/models/fsp15k/multifit_fp16 -> ./de-1
        dataset_path = Path(d.get('dataset_path', experiment_path.parent.parent.parent))
        d['dataset_path'] = dataset_path
        d['experiment_path'] = experiment_path
        arch['lang'] = arch.get('lang', None) or detect_lang_from_dataset_path(dataset_path)
        arch['tokenizer_type'] = arch.pop('tokenizer', arch.get('tokenizer_type', None))

        if update_arch:
            self.arch.replace_(_verbose_diff=not silent, **arch)
        self.replace_(_verbose_diff=not silent, **d)
        return arch


@dataclass
class ULMFiTPretraining(ULMFiTTrainingCommand):
    num_epochs: int = 10
    bs: int = 20
    bptt: int = 70
    drop_mult: float = 1.0
    dropout_values: dict = field(default_factory=dict)
    label_smoothing_eps: float = 0.0
    label_smoothing_eps_norm_by_classes: bool = True
    use_adam_08: bool = False
    true_wd: bool = True
    wd: bool = 0.01
    clip: float = None
    fp16: bool = False
    lr: float = 5e-3

    def get_learner(self, data_lm, **additional_trn_args):
        config = awd_lstm_lm_config.copy()
        config.update(emb_sz=self.arch.emb_sz, n_hid=self.arch.n_hid, n_layers=self.arch.n_layers, qrnn=self.arch.qrnn,
                      **self.dropout_values)

        trn_args = dict(drop_mult=self.drop_mult, true_wd=self.true_wd, wd=self.wd,
                        pretrained=False, clip=self.clip)
        trn_args.update(**additional_trn_args)
        print("Training args: ", trn_args, "config: ", config)
        learn = language_model_learner(data_lm,
                                       AWD_LSTM,
                                       config=config,
                                       model_dir=self.model_name,
                                       **trn_args)
        learn = patch_learner(learn)
        # compared to standard Adam, we set beta_1 to 0.8
        if self.use_adam_08:
            learn.opt_func = partial(optim.Adam, betas=(0.8, 0.99))

        learn.callback_fns += [partial(CSVLogger, filename=f"{learn.model_dir}/lm-history")]
        if self.label_smoothing_eps > 0.0:
            eps = self.label_smoothing_eps
            if self.label_smoothing_eps_norm_by_classes:
                eps = eps/ learn.data.c
            print("Using Label smoothing with eps = ", eps)
            learn.loss_func = FlattenedLoss(LabelSmoothingCrossEntropy, eps=eps)

        set_seed(self.seed, "LM training seed")
        if self.fp16:
            learn.to_fp16()
        return learn

    def _fit_schedule(self, learn):
        print("Training lm from random weights")
        learn.unfreeze()
        learn.fit_one_cycle(self.num_epochs, self.lr, (0.8, 0.7))

    def train_(self, dataset_or_path, tokenizer=None, **train_config):
        if self.arch.lang is None:
            lang = detect_lang_from_dataset_path(Path(dataset_or_path))
            if lang is None:
                warn("Unable to detect language from dataset path assuming English, use replace_(lang='??') change it.")
                lang = 'en'
            self.arch.lang = lang
        self.replace_(**train_config, _strict=True)
        set_seed(self.seed, "LM weights seed")
        if tokenizer is None:
            if hasattr(self, 'base'):
                tokenizer = self.base.tokenizer
            else:
                tokenizer = self.arch.new_tokenizer()

        dataset = self._set_dataset_(dataset_or_path, tokenizer)
        learn = self.get_learner(data_lm=dataset.load_lm_databunch(bs=self.bs, bptt=self.bptt))
        experiment_path = learn.path / learn.model_dir
        print("Experiment", experiment_path)
        if self.num_epochs > 0:
            self._fit_schedule(learn)

        self.experiment_path = experiment_path
        tokenizer.save(self.experiment_path, learn=learn)
        learn.to_fp32()
        learn.save_encoder(ENC_BEST)
        learn.save(LM_BEST, with_opt=False)
        learn.destroy()
        self.save_paramters()
        print("Language model saved to", self.experiment_path)

    def validate(self):
        raise NotImplementedError("The validation on the language model is not implemented.")

    @property
    def model_fnames(self):
        if self.experiment_path:
            model_path = self.experiment_path.absolute()
            cache_path = (model_path if (model_path / "itos.pkl").exists() else model_path.parent)
            return [model_path / LM_BEST,  cache_path /'itos']
        return None

    @property
    def encoder_fname(self):
        if self.experiment_path:
            return (self.experiment_path / ENC_BEST).absolute()
        return None


@dataclass
class ULMFiTFinetuning(ULMFiTPretraining):
    base: ULMFiTPretraining = field(repr=False, default=None)

    def __post_init__(self):
        self.lr = 1e-3

    def get_learner(self, data_lm, **additional_trn_args):
        pretrained_fnames = None if self.base is None else self.base.model_fnames
        # data_lm.lang is added after dataloading
        if pretrained_fnames is None and data_lm.lang != 'en':
            warn(f"You are using fastai english langauge model for {data_lm.lang}, you might be better off with just random weights.")
        learn = super().get_learner(data_lm, **additional_trn_args)
        # we don't use pretrained_fnames param so that we can add load_state_dict_hook
        if pretrained_fnames is not None:
            print("Loading pretrained weights: ", pretrained_fnames)
            fnames = [learn.path / learn.model_dir / f'{fn}.{ext}' for fn, ext in zip(pretrained_fnames, ['pth', 'pkl'])]
            learn.load_pretrained(*fnames)
            learn.freeze()
        return learn

    def _fit_schedule(self, learn):
        if self.base is not None and self.base.model_fnames:
            print("Fitting using 2 cycle fit schedule")
            learn.freeze_to(-1)
            learn.fit_one_cycle(1, self.lr * 10, moms=(0.8, 0.7))
            learn.unfreeze()
            learn.fit_one_cycle(self.num_epochs, self.lr, moms=(0.8, 0.7))
        else:
            super()._fit_schedule(learn)

@dataclass
class ULMFiTClassifier(ULMFiTTrainingCommand):
    bs: int = 20
    num_epochs: int = 10
    drop_mult: float = 0.5
    dropout_values: dict = field(default_factory=dict)
    wd: float = 0.01
    clip: float = None
    label_smoothing_eps: float = 0.0
    label_smoothing_eps_norm_by_classes: bool = False
    weighted_cross_entropy: tuple = None
    early_stopping: str = 'accuracy'
    fit_schedule: str = '1cycle'
    base: ULMFiTFinetuning = field(repr=False, default=None)
    random_init: bool = False
    seed: int = 0
    bptt: int = 70
    fp16: bool = False
    arch: ULMFiTArchitecture = None

    def get_learner(self, data_clas, eval_only=False, **additional_trn_args):
        assert self.weighted_cross_entropy is None or self.label_smoothing_eps == 0, "Label smoohting not implemented with weighted_cross_entropy"
        if self.weighted_cross_entropy is not None:
            loss_func = CrossEntropyFlat(weight=torch.tensor(self.weighted_cross_entropy, dtype=torch.float32).cuda())
        elif self.label_smoothing_eps > 0.0:
            eps = self.label_smoothing_eps
            if self.label_smoothing_eps_norm_by_classes:
                eps = eps / data_clas.c
            print("Using Label smoothing with eps = ", eps)
            loss_func = FlattenedLoss(LabelSmoothingCrossEntropy, eps=eps)
        else:
            loss_func = None

        set_seed(self.seed, "Classifier weights seed")
        config = awd_lstm_clas_config.copy()
        config.update(emb_sz=self.arch.emb_sz, n_hid=self.arch.n_hid, n_layers=self.arch.n_layers, qrnn=self.arch.qrnn,
                      **self.dropout_values)

        trn_args = dict(drop_mult=self.drop_mult, wd=self.wd, pretrained=False, bptt=self.bptt,
                        loss_func=loss_func, clip=self.clip)
        if hasattr(Learner, 'silent'):
            trn_args.update(silent=eval_only)

        trn_args.update(**additional_trn_args)
        print("Training args: ", trn_args, "config: ", config)
        learn = text_classifier_learner(data_clas,
                                        AWD_LSTM,
                                        config=config,
                                        model_dir=self.model_name,
                                        **trn_args)
        learn = patch_learner(learn)
        if self.base.encoder_fname and not self.random_init:
            print("Loading pretrained model", self.base.encoder_fname)
            learn.load_encoder(self.base.encoder_fname)
            learn.freeze()
        else:
            warn("No pretrained encoder")

        set_seed(self.seed, "Classifier training seed")
        if not eval_only:
            learn.callback_fns += [partial(CSVLogger, filename=f"{learn.model_dir}/cls-history")]
            if self.early_stopping:
                learn.callback_fns += [partial(SaveModelCallback, every='improvement',
                                               name='cls_best_tmp',
                                               monitor=self.early_stopping)]
        if self.fp16:
            learn.to_fp16()
        return learn

    def train_(self, dataset_or_path=None, **train_config):
        self.replace_(**train_config, _strict=True)

        base_tokenizer = self.base.tokenizer
        dataset = self._set_dataset_(dataset_or_path, base_tokenizer)
        data_clas = dataset.load_clas_databunch(bs=self.bs)
        learn = self.get_learner(data_clas=data_clas)
        print(f"Training: {learn.path / learn.model_dir}")
        learn.unfreeze()
        self._fit_schedule(learn)

        self.experiment_path = learn.path / learn.model_dir
        base_tokenizer.save(self.experiment_path, learn=learn)
        learn.to_fp32()
        learn.save(CLS_BEST, with_opt=False)
        print("Classifier model saved to", self.experiment_path)
        self.save_paramters()
        learn.destroy()
        return

    def _validate(self, learn, ds_type):
        ds_name = ds_type.name.lower()
        print(f"Model: {self.name}, ds_name: {ds_name}")
        results_dict = dict(zip(
            [f'{ds_name} loss'] + [f"{ds_name} {getattr(m, '__name__', m.__class__.__name__)}" for m in learn.metrics],
            map(float, learn.validate(learn.data.dl(ds_type)))))
        results_dict['name'] = self.name
        return results_dict

    def validate(self, *splits, data_cls=None, save_name=CLS_BEST, use_cache=True, save_preds=False):
        """Validates
            splits - Dataset Types to validate on default DatasetType.Test, DatasetType.Valid, DatasetType.Train
        """
        if len(splits) == 0:
            splits = [DatasetType.Test, DatasetType.Valid, DatasetType.Train]
        cache_file = (self.experiment_path / f'results{"" if save_name == CLS_BEST else "-" + save_name}.json')
        if use_cache and cache_file.exists():
            with cache_file.open("r") as fp:
                return json.load(fp)

        if data_cls is None:
            data_cls = self.dataset.load_clas_databunch(bs=self.bs)

        learn = self.get_learner(data_cls, eval_only=True)
        # avg = 'binary' if learn.data.c == 2 else 'macro'
        # FBeta(beta=1.0, average=avg), Precision(average=avg), Recall(average=avg),
        learn.metrics = [accuracy]
        print(f"Loading model {save_name}")
        learn.load(save_name)
        if save_preds:
            probs, targets = learn.get_preds(ordered=True, ds_type=DatasetType.Test, activ=partial(F.softmax, dim=-1))
            np.save(str(self.experiment_path / f"preds-on-test.npy"), probs.cpu().numpy())

        results_dict = {}
        for split in splits:
            results_dict.update(self._validate(learn, split))
        print(results_dict)
        with cache_file.open("w") as fp:
            json.dump(results_dict, fp)
        return results_dict

    def _fit_schedule(self, learn):
        getattr(self, '_fit_schedule_' + self.fit_schedule)(learn)

    def _fit_schedule_1cycle(self, learn):
        learn.unfreeze()
        learn.fit_one_cycle(self.num_epochs, slice(1e-2 / (2.6 ** 4), 2e-2), moms=(0.8, 0.7))

    def _fit_schedule_layered(self, learn):
        learn.freeze_to(-1)
        learn.fit_one_cycle(1, 2e-2, moms=(0.8, 0.7))
        if self.num_epochs > 1:
            learn.freeze_to(-2)
            learn.fit_one_cycle(1, slice(1e-2 / (2.6 ** 4), 1e-2), moms=(0.8, 0.7))
            learn.freeze_to(-3)
            learn.fit_one_cycle(1, slice(5e-3 / (2.6 ** 4), 5e-3), moms=(0.8, 0.7))
            learn.unfreeze()
        if self.num_epochs > 5:
            learn.fit_one_cycle(self.num_epochs - 4, slice(1e-3 / (2.6 ** 4), 1e-3), moms=(0.8, 0.7))

    def _fit_schedule_2cycle(self, learn):
        learn.freeze_to(-1)
        learn.fit_one_cycle(1, 2e-2, moms=(0.8, 0.7))
        learn.unfreeze()
        if self.num_epochs > 1:
            learn.fit_one_cycle(self.num_epochs - 1, slice(1e-2 / (2.6 ** 4), 1e-2), moms=(0.8, 0.7))

    def _fit_schedule_reverse_2cycle(self, learn):
        learn.unfreeze()
        for g in learn.layer_groups[-1:]:
            for l in g:
                if not learn.train_bn or not isinstance(l, bn_types): requires_grad(l, False)
        learn.create_opt(defaults.lr)
        learn.fit_one_cycle(self.num_epochs, slice(1e-2 / (2.6 ** 4), 2e-2), moms=(0.8, 0.7))
        learn.unfreeze()
        learn.fit_one_cycle(self.num_epochs, slice(1e-3 / (2.6 ** 4), 2e-3), moms=(0.8, 0.7))

    def _fit_schedule_false_wd(self, learn):
        learn.true_wd = False
        learn.fit_one_cycle(1, 5e-2, moms=(0.8, 0.7), wd=1e-7)
        if self.num_epochs > 1:
            learn.freeze_to(-2)
            learn.fit_one_cycle(1, slice(5e-2 / (2.6 ** 4), 5e-2), moms=(0.8, 0.7), wd=1e-7)
            learn.freeze_to(-3)
            learn.fit_one_cycle(1, slice(5e-4 / (2.6 ** 4), 5e-4), moms=(0.8, 0.7), wd=1e-7)
            learn.unfreeze()
            if self.num_epochs > 5:
                learn.fit_one_cycle(self.num_epochs - 4, slice(1e-2 / (2.6 ** 4), 1e-2), moms=(0.8, 0.7), wd=1e-7)


def path_if_model_exists(path, weights_name):
    """Return path to model if it exists"""
    model_path = path / (weights_name + ".pth")
    return path if model_path.exists() else None


@dataclass
class ULMFiT:
    arch: ULMFiTArchitecture = None
    pretrain_lm: ULMFiTPretraining = None
    finetune_lm: ULMFiTFinetuning = None
    classifier: ULMFiTClassifier = None

    def __post_init__(self):
        self.arch = ULMFiTArchitecture()
        self.pretrain_lm = ULMFiTPretraining(arch=self.arch)
        self.finetune_lm = ULMFiTFinetuning(arch=self.arch, base=self.pretrain_lm)
        self.classifier = ULMFiTClassifier(arch=self.arch, base=self.finetune_lm)

    def load_(self, experiment_path:Path, silent=False):
        success = (self.classifier.load_(experiment_path, silent=silent) or
                   self.finetune_lm.load_(experiment_path, silent=silent) or
                   self.pretrain_lm.load_(experiment_path, silent=silent) or
                   self.load_legacy_(experiment_path, silent=silent))
        if not success:
            warn(f'Unable to load experiment {experiment_path}')
        return self

    def load_legacy_(self, experiment_path, silent=True):
        if not (experiment_path / "info.json").exists():
            return False
        with (experiment_path / "info.json").open('r') as f:
            d = json.load(f)
        dataset_path = d.pop('dataset_path', "")
        d['n_hid'] = d['nh']
        d['n_layers'] = d['nl']
        d['lang'] = detect_lang_from_dataset_path(Path(dataset_path))
        if "wiki" in str(dataset_path):
            self.arch.replace_(**d)
            self.pretrain_lm.replace_(**d)
            self.pretrain_lm.experiment_path = path_if_model_exists(experiment_path, LM_BEST)
            self.pretrain_lm.dataset_path = dataset_path if dataset_path in str(experiment_path) else None
        else:
            self.replace_(**d)
            self.finetune_lm.experiment_path = path_if_model_exists(experiment_path, ENC_BEST)
            self.finetune_lm.dataset_path = dataset_path if dataset_path in str(experiment_path) else None
            self.classifier.experiment_path = path_if_model_exists(experiment_path, CLS_BEST)
            self.classifier.dataset_path = dataset_path if dataset_path in str(experiment_path) else None
        return True

    def replace_(self, **kwargs):
        self.arch.replace_(**kwargs)
        self.pretrain_lm.replace_(**kwargs)
        self.finetune_lm.replace_(**kwargs)
        self.classifier.replace_(**kwargs)
        return self

    def pprint(self):
        print(f"""ULMFiT(
    {self.arch},
    {self.pretrain_lm},
    {self.finetune_lm},
    {self.classifier},
)""")

    def from_pretrained_(self, name, repo="n-waves/multifit-models"):
        name = name.rstrip(".tgz")  # incase someone put's tgz name the name
        url = f"https://github.com/{repo}/releases/download/{name}/{name}.tgz"
        path = untar_data(url.rstrip(".tgz"), data=False)  # untar_data adds .tgz
        return self.load_(path)


def from_pretrained(name):
    #TODO: Detect name and load configuration
    from . import configurations
    return configurations.multifit_paper_version().from_pretrained_(name)
