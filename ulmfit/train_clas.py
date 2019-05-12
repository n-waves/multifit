"""
Train a classifier on top of a language model trained with `pretrain_lm.py`.
Optionally fine-tune LM before.
"""

from fastai.callbacks import CSVLogger, SaveModelCallback
from fastai.text import *

from fastai_contrib.utils import PAD_TOKEN_ID

import fire

from ulmfit.pretrain_lm import LMHyperParams, ENC_BEST

from sklearn.metrics import f1_score as f1s, precision_score, recall_score

@dataclass
class CLSHyperParams(LMHyperParams):
    # dir_path -> data/imdb/
    use_test_for_validation: bool=False
    ftseed: int = None
    clsweightseed: int = None
    clstrainseed: int = None

    bicls_head:str = 'BiPoolingLinearClassifier'

    def __post_init__(self, *args, **kwargs):
        super().__post_init__(*args, **kwargs)
        self.dataset_dir=self.dataset_path

    @property
    def model_suffix(self):
        s1 = '' if self.lmseed is None else f'lmseed-{self.lmseed}'
        s2 = '' if self.ftseed is None else f'ftseed-{self.ftseed}'
        s3 = '' if self.clsweightseed is None else f'clsweightseed-{self.clsweightseed}'
        s4 = '' if self.clstrainseed is None else f'clstrainseed-{self.clstrainseed}'
        s = '-'.join([x for x in [s1, s2, s3, s4] if x != ''])
        if s != '':
            return '_'+s
        return ''

    @property
    def need_fine_tune_lm(self): return not (self.model_dir/f"enc_best.pth").exists()

    def lr_schedule_layered(self, learn, num_cls_epochs):
        learn.freeze_to(-1)
        learn.fit_one_cycle(1, 2e-2, moms=(0.8, 0.7))
        if num_cls_epochs > 1:
            learn.freeze_to(-2)
            learn.fit_one_cycle(1, slice(1e-2 / (2.6 ** 4), 1e-2), moms=(0.8, 0.7))
            learn.freeze_to(-3)
            learn.fit_one_cycle(1, slice(5e-3 / (2.6 ** 4), 5e-3), moms=(0.8, 0.7))
            learn.unfreeze()
            learn.fit_one_cycle(num_cls_epochs, slice(1e-3 / (2.6 ** 4), 1e-3), moms=(0.8, 0.7))

    def lr_schedule_2cycle(self, learn, num_cls_epochs):
        print("2cycle training schedule")
        learn.freeze_to(-1)
        learn.fit_one_cycle(1, 2e-2, moms=(0.8, 0.7))
        learn.unfreeze()
        if num_cls_epochs > 1:
            learn.fit_one_cycle(num_cls_epochs -1, slice(1e-2 / (2.6 ** 4), 1e-2), moms=(0.8, 0.7))

    def lr_schedule_1cycle(self, learn, num_cls_epochs):
        print("Single training schedule")
        learn.unfreeze()
        learn.fit_one_cycle(num_cls_epochs, slice(1e-2 / (2.6 ** 4), 2e-2), moms=(0.8, 0.7))

    def lr_schedule_false_wd(self, learn, num_cls_epochs):
        learn.true_wd = False
        print("Starting classifier training")
        learn.fit_one_cycle(1, 5e-2, moms=(0.8, 0.7), wd=1e-7)
        if num_cls_epochs > 1:
            learn.freeze_to(-2)
            learn.fit_one_cycle(1, slice(5e-2 / (2.6 ** 4), 5e-2), moms=(0.8, 0.7), wd=1e-7)
            learn.freeze_to(-3)
            learn.fit_one_cycle(1, slice(5e-4 / (2.6 ** 4), 5e-4), moms=(0.8, 0.7), wd=1e-7)
            learn.unfreeze()
            if num_cls_epochs > 5:
                learn.fit_one_cycle(num_cls_epochs-4, slice(1e-2 / (2.6 ** 4), 1e-2), moms=(0.8, 0.7), wd=1e-7)

    def get_metrics(self, init=False):
        f1_score = FBeta(beta=1.0)
        precision = Precision()
        recall = Recall()
        metrics = [f1_score, precision, recall]

        # TODO: fix this in fast.ai
        if init:
            for metric in metrics: metric.on_train_begin()
        metrics.append(accuracy)
        return metrics

    def output_metrics(self, results):
        print(f"F1 score bin: {results[1].item()}")
        print(f"Loss: {results[0]}")
        print(f"Precision: {results[2].item()}")
        print(f"Recall: {results[3].item()}")
        print(f"Accuracy: {results[4].item()}")


    def train_cls(self, num_lm_epochs, unfreeze=True, num_cls_frozen_epochs=1, bs=40, drop_mul_lm=0.3, drop_mul_cls=0.5,
                  use_test_for_validation=False, num_cls_epochs=2, limit=None, noise=0.0, cls_max_len=20*70, lr_sched='layered',
                  label_smoothing_eps=0.0, random_init=False, dump_preds=None):
        assert use_test_for_validation == False, "use_test_for_validation=True is not supported"
        self.model_dir.mkdir(exist_ok=True, parents=True)

        if not unfreeze:
           num_cls_epochs = 1

        data_clas, data_lm, data_tst = self.load_cls_data(bs, limit=limit, noise=noise)

        if self.need_fine_tune_lm and not random_init:
            if not (self.model_dir/(ENC_BEST+".pth")).exists():
                self.train_lm(num_lm_epochs, data_lm=data_lm, drop_mult=drop_mul_lm, label_smoothing_eps=label_smoothing_eps)
            else:
                print("Language model already exist, skipping finetuning")
        loss_func = CrossEntropyFlat(weight=torch.FloatTensor([0.5,30]).cuda())

        self.set_seed(self.clsweightseed, "classifier weights")

        learn = self.create_cls_learner(data_clas, drop_mult=drop_mul_cls, max_len=cls_max_len,
                                        label_smoothing_eps=label_smoothing_eps, random_init=random_init,
                                        metrics=self.get_metrics(),
                                        loss_func=loss_func)

        if not random_init:
            try:
                learn.load('cls_best')
                print("Loading last classifier")
            except FileNotFoundError:
                learn.load_encoder(ENC_BEST)
        else:
            print("Starting classifier from random weights")


        self.set_seed(self.clstrainseed, "classifier train")

        if hasattr(self, 'lr_schedule_'+lr_sched):
            learn.true_wd = True
            getattr(self, 'lr_schedule_'+lr_sched)(learn, num_cls_epochs)
        else:
            raise ValueError(f"Wrong lr_sched: {lr_sched}")

        print(f"Saving models at {learn.path / learn.model_dir}")
        learn.save('cls_last', with_opt=False)
        #learn.save('cls_best', with_opt=False) # we don't use early stopping for the time being
        del learn
        return self.validate_cls('cls_best', bs=bs, data_tst=data_tst, learn=None)

    def validate_cls(self, save_name='cls_best', bs=40, data_tst=None, learn=None, dump_preds=None, mode="test"):
        if data_tst is None:
            data_clas , _, data_tst = self.load_cls_data(bs)
        if learn is None:
            learn = self.create_cls_learner(data_tst, drop_mult=0.3, metrics=self.get_metrics(True))
            learn.unfreeze()
        learn.load(save_name)
        probs, targets = learn.get_preds(ordered=True)
        preds = np.argmax(probs.cpu().numpy(), axis=1)
        if dump_preds:
            with open(dump_preds, 'w') as f:
                f.write('\n'.join([str(x) for x in preds]))
        results = learn.validate(data_tst.valid_dl if mode == "test" else data_clas.valid_dl)
        print(f"Model: {self.name}")
        print(f"Validation on: {mode}")
        self.output_metrics(results)

        return list(map(float, results))

    def create_cls_learner(self, data_clas, dps=None, label_smoothing_eps=0.0, random_init=False, **kwargs):
        assert self.bidir == False, "bidirectional model is not yet supported"
        config = dict(emb_sz=self.emb_sz, n_hid=self.nh, n_layers=self.nl, pad_token=PAD_TOKEN_ID, qrnn=self.qrnn)
        config.update(dps or self.dps)
        trn_args=dict(bptt=self.bptt, clip=self.clip)
        trn_args.update(kwargs)
        learn = text_classifier_learner(data_clas, AWD_LSTM, config=config,
            pretrained=False, path=self.model_dir.parent, model_dir=self.model_dir.name, **trn_args)

        if self.pretrained_model is not None and not random_init:
            print("Loading pretrained model", self.pretrained_model)
            model_path = untar_data(self.pretrained_model, data=False)
            fnames = [list(model_path.glob(f'*.{ext}'))[0] for ext in ['pth', 'pkl']]
            learn.load_pretrained(*fnames, strict=False)
            learn.freeze()

        learn.callback_fns += [partial(CSVLogger, filename=f"{learn.model_dir}/cls-history"),
                               partial(SaveModelCallback, every='improvement', name='cls_best', monitor="f_beta")
                               ]
        if label_smoothing_eps > 0.0:
            learn.loss_func = FlattenedLoss(LabelSmoothingCrossEntropy, eps=label_smoothing_eps)
        return learn

    def load_cls_data(self, bs, **kwargs):
        self.model_dir.mkdir(exist_ok=True, parents=True)
        add_trn_to_lm = True
        lang = self.lang
        use_moses = False #True
        if 'xnli' in str(self.dataset_dir):
            NotImplementedError("Support for Xnli is not implemented yet")
        if 'imdb' in self.dataset_dir.name:
            lang=''
            add_trn_to_lm = True
        if 'mldoc' in str(self.dataset_dir):
            add_trn_to_lm = False  # False as trn_df is contained in unsup already
            lang = self.lang

        data = self.load_data(lang=lang,
                              add_trn_to_lm=add_trn_to_lm,
                              use_moses=use_moses,
                              **kwargs)
        return self.databunches(bs, **data)

    def load_data(self, lang='', **kwargs):
        prefix = '' if lang == '' else lang+'.'
        trn_df = pd.read_csv(self.dataset_path / f'{prefix}train.csv', header=None)
        tst_df = pd.read_csv(self.dataset_path / f'{prefix}test.csv', header=None)
        val_fn = self.dataset_path / f'{prefix}dev.csv'
        if val_fn.exists():
            print("Loading validation", val_fn)
            val_df = pd.read_csv(val_fn, header=None)
        else:
            val_df = None

        unsup_df = pd.read_csv(self.dataset_path / f'{prefix}unsup.csv', header=None)

        if val_df is None:
            print("Validation set not found using 10% of trn")
            val_len = max(int(len(trn_df) * 0.1), 2)
            trn_len = len(trn_df) - val_len
            trn_df, val_df = trn_df[:trn_len], trn_df[trn_len:]
        kwargs.update(dict(trn_df=trn_df, val_df=val_df, tst_df=tst_df, unsup_df=unsup_df))
        return kwargs

    def add_noise(self, trn_df, noise):
        count = len(trn_df)
        labels = trn_df[0].unique()
        assert np.issubdtype(labels.dtype, np.integer), "noise only works on numerical numbers"
        modulo = labels.max() + 1
        idx_to_distrub = np.random.permutation(count)[:int(count * noise)]
        trn_df.loc[idx_to_distrub, [0]] = (np.random.randint(1, modulo - 1, size=len(idx_to_distrub)) +
                                           trn_df.loc[idx_to_distrub][0]) % modulo
        print(f"Added noise to {len(idx_to_distrub)} examples, only {(count - len(idx_to_distrub)) / count} have correct labels")
        return trn_df

    def databunches(self, bs, trn_df, val_df, tst_df, unsup_df, add_trn_to_lm=True, use_moses=False, force=False, limit=None, noise=0.0):
        lm_trn_df = pd.concat([unsup_df, val_df, tst_df] + ([trn_df] if add_trn_to_lm else []))
        val_len = max(int(len(lm_trn_df) * 0.1), 2)
        lm_trn_df = lm_trn_df[val_len:]
        lm_val_df = lm_trn_df[:val_len]

        cls_name="cls"
        if limit is not None:
            print("Limiting data set to:", limit)
            trn_df = trn_df[:limit]
            val_df = val_df[:limit]
            cls_name=f'{cls_name}limit{limit}'

        if noise > 0.0:
            trn_df = self.add_noise(trn_df, noise)
            val_df = self.add_noise(val_df, noise)
            cls_name = f'{cls_name}noise{noise}tv'

        args = self.tokenizer_to_fastai_args(sp_data_func=lambda: trn_df[1], use_moses=use_moses)
        args['text_cols'] = list(trn_df.columns.values)[1:]
        args['mark_fields'] = True
        lm_suffix = self.bptt if self.bptt != 70 else ""
        data_lm = self.lm_databunch(f'lm{lm_suffix}', train_df=lm_trn_df, valid_df=lm_val_df, bs=bs, force=force, bptt=self.bptt, **args)
        args['vocab'] = data_lm.vocab
        data_cls = self.cls_databunch(cls_name, train_df=trn_df, valid_df=val_df, bs=bs, force=force, **args)
        data_tst = self.cls_databunch('tst', train_df=val_df, valid_df=tst_df, bs=bs, force=force, **args) # Hack to load test dataset with labels

        print('Size of vocabulary:', len(data_lm.vocab.itos))
        print('First 20 words in vocab:', data_lm.vocab.itos[:20])
        return data_cls, data_lm, data_tst

    def cls_databunch(self, name, *args, **kwargs):
        return self.databunch(name, bunch_class=TextClasDataBunch, *args, **kwargs)

if __name__ == '__main__':
    fire.Fire(CLSHyperParams)

##

