import dataclasses
from fastai.callbacks import CSVLogger, SaveModelCallback
from fastai.text import *
import torch

from ulmfit.datasets import ULMFiTDataset
from pathlib import Path

CLS_BEST = 'cls_best'
LM_BEST = "lm_best"
ENC_BEST = "enc_best"


@dataclass
class Params:
    def replace_(self, **changes):
        for f in dataclasses.fields(self):
            if f.name in changes:
                v = changes[f.name]
                if f.type == Path and v is not None:
                    v = Path(v)
                setattr(self, f.name, v)
        return self


@dataclass
class ULMFITArchitecture(Params):
    tokenizer: str = "f"
    max_vocab: int = 60000

    emb_sz: int = awd_lstm_lm_config['emb_sz']
    n_hid: int = awd_lstm_lm_config['n_hid']
    n_layers: int = awd_lstm_lm_config['n_layers']
    qrnn: bool = awd_lstm_lm_config['qrnn']

    def model_name(self, name):
        model_suffix = ''  # if self.lmseed is None else f'_lmseed-{self.lmseed}'
        model_prefix = 'qrnn' if self.qrnn else 'lstm'

        model_name = f"{model_prefix}_{name}{model_suffix}.m"
        return model_name

    def dataset_cache_suffix(self):
        tokenizer_prefix = f"{self.tokenizer}{self.max_vocab // 1000}k"
        return f'models/{tokenizer_prefix}'

    def dataset(self, dataset_path_or_object, **args):
        if isinstance(dataset_path_or_object, Dataset):
            return dataset_path_or_object
        return ULMFiTDataset(dataset_path=Path(dataset_path_or_object), tokenizer=self.tokenizer, max_vocab=self.max_vocab, **args)


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


@dataclass
class ULMFiTTrainingCommand(Params):
    seed: int = 0
    name: str = None
    arch: ULMFITArchitecture = field(repr=False, default=None)
    experiment_path: Path = None
    dataset_path: Path = None

    @property
    def model_name(self):
        return (self.name or self.arch.model_name()) + (
            "" if self.seed == 0 or "seed" in self.name else f"seed{self.seed}")

    @property
    def info_json(self):
        return self.__class__.__name__.lower().replace("ulmfit", "") + ".json"


    def _set_dataset_(self, dataset_or_path):
        dataset_or_path = self.arch.dataset(dataset_or_path or self.dataset_path or getattr(self, 'base', self).dataset_path)
        self.dataset_path = dataset_or_path.dataset_path
        return dataset_or_path

    @property
    def dataset(self):
        return self.arch.dataset(self.dataset_path)

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

    def load_(self, experiment_path, tantetive=True):
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
        self.arch.replace_(**arch)
        self.replace_(**d)
        if base is not None:
            other_arch = getattr(self, 'base').load_(Path(base), tantetive=True)
            if other_arch and other_arch != arch:
                warn(f"Architecuture does not match {arch}, {other_arch}")
        self.name = experiment_path.name
        dataset_path = experiment_path.parent.parent.parent  # data/mldoc/de-1/models/fsp15k/multfit_fp16 -> data/mldoc/de-1
        self.dataset_path = Path(dataset_path)
        self.experiment_path = Path(experiment_path)
        return arch


@dataclass
class ULMFiTPretraining(ULMFiTTrainingCommand):
    num_epochs: int = 10
    bs: int = 20
    bptt: int = 70
    drop_mult: float = 1.0
    label_smoothing_eps: float = 0.0
    use_adam_08: bool = False
    true_wd: bool = True
    wd: bool = 0.1
    fp16: bool = False
    lr: float = 5e-3

    def _learner(self, dataset, **additional_trn_args):
        config = awd_lstm_lm_config.copy()
        config.update(emb_sz=self.arch.emb_sz, n_hid=self.arch.n_hid, n_layers=self.arch.n_layers, qrnn=self.arch.qrnn)

        trn_args = dict(drop_mult=self.drop_mult, true_wd=self.true_wd, wd=self.wd,
                        pretrained=False)
        trn_args.update(**additional_trn_args)
        print("Training args: ", trn_args, "config: ", config)
        data_lm = dataset.load_lm_databunch(bs=self.bs, bptt=self.bptt)
        learn = language_model_learner(data_lm,
                                       AWD_LSTM,
                                       config=config,
                                       model_dir=self.model_name,
                                       **trn_args)

        # compared to standard Adam, we set beta_1 to 0.8
        if self.use_adam_08:
            learn.opt_func = partial(optim.Adam, betas=(0.8, 0.99))

        learn.callback_fns += [partial(CSVLogger, filename=f"{learn.model_dir}/lm-history")]
        if self.label_smoothing_eps > 0.0:
            learn.loss_func = FlattenedLoss(LabelSmoothingCrossEntropy, eps=self.label_smoothing_eps)

        set_seed(self.seed, "LM training seed")
        if self.fp16:
            learn.to_fp16()
        return learn

    def _fit_schedule(self, learn):
        print("Training lm from random weights")
        learn.unfreeze()
        learn.fit_one_cycle(self.num_epochs, self.lr, (0.8, 0.7))

    def train_(self, dataset_or_path=None, **train_config):
        dataset = self._set_dataset_(dataset_or_path)
        self.replace_(**train_config, _strict=True)
        set_seed(self.seed, "LM weights seed")
        if hasattr(self, 'base'):
            dataset.use_base_model_subword_vocabulary(self.base.experiment_path)
        learn = self._learner(dataset)
        experiment_path = learn.path / learn.model_dir
        print("Experiment", experiment_path)
        if self.num_epochs > 0:
            self._fit_schedule(learn)
        self.experiment_path = experiment_path
        learn.to_fp32()
        learn.save_encoder(ENC_BEST)
        learn.save(LM_BEST, with_opt=False)
        learn.destroy()
        print("Language model saved to", self.experiment_path)
        self.save_paramters()

    def validate(self):
        raise NotImplementedError("The validation on the language model is not implemented.")

    @property
    def model_fnames(self):
        if self.experiment_path:
            return [self.experiment_path.absolute() / LM_BEST, self.experiment_path.parent.absolute() / 'itos']
        return None

    @property
    def encoder_fname(self):
        if self.experiment_path:
            return (self.experiment_path / ENC_BEST).absolute()
        return None


@dataclass
class ULMFiTFinetuining(ULMFiTPretraining):
    base: ULMFiTPretraining = field(repr=False, default=None)
    pretrained: bool = True

    def __post_init__(self):
        self.lr = 1e-3

    def _learner(self, dataset, **additional_trn_args):
        pretrained_fnames = None if self.base is None else self.base.model_fnames
        if self.pretrained and pretrained_fnames is None and dataset.lang != 'en':
            warn(
                "You are using fastai english langauge model for {data_lm.lang}, you might be better off with just random weights.")
        return super()._learner(dataset, pretrained=self.pretrained, pretrained_fnames=pretrained_fnames,
                                **additional_trn_args)

    def _fit_schedule(self, learn):
        if self.pretrained:
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
    wd: float = 0.1
    label_smoothing_eps: float = 0.0
    weighted_cross_entropy: tuple = None
    early_stopping: str = 'accuracy'
    fit_schedule: str = '1cycle'
    base: ULMFiTFinetuining = field(repr=False, default=None)
    random_init: bool = False
    seed: int = 0
    bptt: int = 70
    fp16: bool = False
    arch: ULMFITArchitecture = None

    def _learner(self, dataset, eval_only=False, **additional_trn_args):
        assert self.weighted_cross_entropy is None or self.label_smoothing_eps == 0, "Label smoohting not implemented with weighted_cross_entropy"
        if self.weighted_cross_entropy is not None:
            loss_func = CrossEntropyFlat(weight=torch.tensor(self.weighted_cross_entropy, dtype=torch.float32).cuda())
        elif self.label_smoothing_eps > 0.0:
            loss_func = FlattenedLoss(LabelSmoothingCrossEntropy, eps=self.label_smoothing_eps)
        else:
            loss_func = None

        set_seed(self.seed, "Classifier weights seed")
        data_clas, data_tst = dataset.load_clas_databunch(bs=self.bs)
        config = awd_lstm_clas_config.copy()
        config.update(emb_sz=self.arch.emb_sz, n_hid=self.arch.n_hid, n_layers=self.arch.n_layers, qrnn=self.arch.qrnn)

        trn_args = dict(drop_mult=self.drop_mult, wd=self.wd, pretrained=False, bptt=self.bptt,
                        loss_func=loss_func)

        trn_args.update(**additional_trn_args)
        print("Training args: ", trn_args, "config: ", config)
        learn = text_classifier_learner(data_clas,
                                        AWD_LSTM,
                                        config=config,
                                        model_dir=self.model_name,
                                        silent=eval_only,
                                        **trn_args)
        learn.data.test_dl = data_tst.valid_dl
        if self.base and not self.random_init:
            print("Loading pretrained model", self.base.encoder_fname)
            learn.load_encoder(self.base.encoder_fname)
            learn.freeze()

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
        dataset = self._set_dataset_(dataset_or_path)
        self.replace_(**train_config, _strict=True)
        dataset.use_base_model_subword_vocabulary(self.base.experiment_path)
        learn = self._learner(dataset)

        self._fit_schedule(learn)

        self.experiment_path = learn.path / learn.model_dir
        learn.to_fp32()
        learn.save(CLS_BEST, with_opt=False)
        print("Classifier model saved to", self.experiment_path)
        self.save_paramters()
        learn.destroy()

    def _validate(self, learn, ds_type):
        ds_name = ds_type.name.lower()
        print(f"Model: {self.name}, ds_name: {ds_name}")
        results_dict = dict(zip(
            [f'{ds_name} loss'] + [f"{ds_name} {getattr(m, '__name__', m.__class__.__name__)}" for m in learn.metrics],
            map(float, learn.validate(learn.data.dl(ds_type)))))
        results_dict['name'] = self.name
        return results_dict

    def validate(self, save_name=CLS_BEST, use_cache=True):
        dataset = self.dataset
        cache_file = (self.experiment_path / f'results{"" if save_name == CLS_BEST else "-" + save_name}.json')
        if use_cache and cache_file.exists():
            with cache_file.open("r") as fp:
                return json.load(fp)

        learn = self._learner(dataset, eval_only=True)
        avg = 'binary' if learn.data.c == 2 else 'macro'
        learn.metrics = [FBeta(beta=1.0, average=avg), Precision(average=avg), Recall(average=avg), accuracy]
        print(f"Loading model {save_name}")
        learn.load(save_name)

        probs, targets = learn.get_preds(ordered=True, ds_type=DatasetType.Test, activ=partial(F.softmax, dim=-1))
        np.save(str(self.experiment_path / f"preds-on-test.npy"), probs.cpu().numpy())

        results_dict = self._validate(learn, DatasetType.Test)
        results_dict.update(self._validate(learn, DatasetType.Valid))
        results_dict.update(self._validate(learn, DatasetType.Train))
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
    arch: ULMFITArchitecture = None
    pretrain_lm: ULMFiTPretraining = None
    finetuine_lm: ULMFiTFinetuining = None
    classifier: ULMFiTClassifier = None

    def __post_init__(self):
        self.arch = ULMFITArchitecture()
        self.pretrain_lm = ULMFiTPretraining(arch=self.arch)
        self.finetuine_lm = ULMFiTFinetuining(arch=self.arch, base=self.pretrain_lm)
        self.classifier = ULMFiTClassifier(arch=self.arch, base=self.finetuine_lm)

    def load_(self, experiment_path:Path):
        success = (self.classifier.load_(experiment_path) or
                   self.finetuine_lm.load_(experiment_path) or
                   self.pretrain_lm.load_(experiment_path) or
                   self.load_legacy_(experiment_path))
        if not success:
            warn('Unable to load experiment')
        return self

    def load_legacy_(self, experiment_path):
        if not (experiment_path / "info.json").exists():
            return False
        with (experiment_path / "info.json").open('r') as f:
            d = json.load(f)
        d.pop('dataset_path', None)
        d['n_hid'] = d['nh']
        d['n_layers'] = d['nl']
        self.replace_(**d)
        if "wiki" in str(experiment_path):
            self.pretrain_lm.experiment_path = path_if_model_exists(experiment_path, LM_BEST)
            self.pretrain_lm.dataset_path = experiment_path.parent.parent.parent
        else:
            self.finetuine_lm.experiment_path = path_if_model_exists(experiment_path, ENC_BEST)
            self.finetuine_lm.dataset_path = experiment_path.parent.parent.parent
            self.classifier.experiment_path = path_if_model_exists(experiment_path, CLS_BEST)
            self.classifier.dataset_path = experiment_path.parent.parent.parent
        return True

    def replace_(self, **kwargs):
        self.arch.replace_(**kwargs)
        self.pretrain_lm.replace_(**kwargs)
        self.finetuine_lm.replace_(**kwargs)
        self.classifier.replace_(**kwargs)
        return self

    def pprint(self):
        print(f"""ULMFiT(
    {self.arch},
    {self.pretrain_lm},
    {self.finetuine_lm},
    {self.classifier},
)""")
