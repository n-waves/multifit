"""
Train a classifier on top of a language model trained with `pretrain_lm.py`.
Optionally fine-tune LM before.
"""

from fastai.callbacks import CSVLogger
from fastai.text import *

from fastai_contrib.utils import PAD_TOKEN_ID

import fire

from ulmfit.pretrain_lm import LMHyperParams, ENC_BEST

class OneHotAcc(Callback):
    def on_epoch_begin(self, **kwargs):
        self.correct, self.total = 0, 0

    def on_batch_end(self, last_output, last_target, last_input, **kwargs):
        preds = last_output.argmax(1)
        target = last_target.argmax(1) if len(last_target.shape)>1 else last_target

        self.correct += (preds == target).float().sum()
        self.total += last_target.size(0)

    def on_epoch_end(self, **kwargs):
        self.metric = self.correct/self.total


class OneHotCrossEntropyWithSmoothLabelling:
    "Same as `func`, but flattens input and target."
    def __init__(self, eps=0.1):
        self.eps = eps

    def __repr__(self): return "OneHotCrossEntropyWithSmoothLabelling"

    def __call__(self, output:Tensor, target:Tensor, **kwargs)->Rank0Tensor:
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        loss = -log_preds.sum(dim=-1).mean() * self.eps / c #(c-1)
        #loss -= (1-self.eps*c/(c-1)) * (log_preds * target).sum(dim=-1).mean()
        loss -= (1 - self.eps) * (log_preds * target).sum(dim=-1).mean()
        return loss

class OneHotCrossEntropy:
    def __repr__(self): return "OneHotCrossEntropy"

    def __call__(self, output:Tensor, target:Tensor, **kwargs)->Rank0Tensor:
        log_preds = F.log_softmax(output, dim=-1)
        loss = - (log_preds * target).sum(dim=-1).mean()
        return loss



class CLSHyperParams(LMHyperParams):
    # dir_path -> data/imdb/
    use_test_for_validation=False

    bicls_head:str = 'BiPoolingLinearClassifier'

    def __post_init__(self, *args, **kwargs):
        super().__post_init__(*args, **kwargs)
        self.dataset_dir=self.dataset_path

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

    def train_cls(self, num_lm_epochs, unfreeze=True, num_cls_frozen_epochs=1, bs=40, drop_mul_lm=0.3, drop_mul_cls=0.5,
                  use_test_for_validation=False, num_cls_epochs=2, limit=None, noise=0.0, cls_max_len=20*70, lr_sched='layered',
                  label_smoothing_eps=0.0, label_cols=0, text_cols=1):
        assert use_test_for_validation == False, "use_test_for_validation=True is not supported"
        self.model_dir.mkdir(exist_ok=True, parents=True)

        if not unfreeze:
           num_cls_epochs = 1

        data_clas, data_lm, data_tst = self.load_cls_data(bs, limit=limit, noise=noise, label_cols=label_cols, text_cols=text_cols)

        if self.need_fine_tune_lm: self.train_lm(num_lm_epochs, data_lm=data_lm, drop_mult=drop_mul_lm, label_smoothing_eps=label_smoothing_eps)
        metrics = [OneHotAcc()] if isinstance(label_cols, list) and len(label_cols) > 1 else None
        learn = self.create_cls_learner(data_clas, drop_mult=drop_mul_cls, max_len=cls_max_len, label_smoothing_eps=label_smoothing_eps, metrics=metrics)

        try:
            learn.load('cls_best')
            print("Loading last classifier")
        except FileNotFoundError:
            learn.load_encoder(ENC_BEST)

        if isinstance(label_cols, list) and len(label_cols) > 1:
            if label_smoothing_eps > 0.0:
                learn.loss_func = OneHotCrossEntropyWithSmoothLabelling(label_smoothing_eps)
            else:
                learn.loss_func = OneHotCrossEntropy()
        if hasattr(self, 'lr_schedule_'+lr_sched):
            learn.true_wd = True
            getattr(self, 'lr_schedule_'+lr_sched)(learn, num_cls_epochs)
        else:
            raise ValueError(f"Wrong lr_sched: {lr_sched}")

        print(f"Saving models at {learn.path / learn.model_dir}")
        learn.save('cls_last', with_opt=False)
        learn.save('cls_best', with_opt=False) # we don't use early stopping for the time being
        del learn
        return self.validate_cls('cls_best', bs=bs, data_tst=data_tst, learn=None, label_cols=label_cols, text_cols=text_cols)

    def validate_cls(self, save_name='cls_best', bs=40, data_tst=None, learn=None, label_cols=0, text_cols=1):
        if data_tst is None:
            _, _, data_tst = self.load_cls_data(bs, label_cols=label_cols, text_cols=text_cols)
        if learn is None:
            metrics = [OneHotAcc()] #if isinstance(label_cols, list) and len(label_cols) > 1 else None
            learn = self.create_cls_learner(data_tst, drop_mult=0.3,metrics=metrics)
            learn.unfreeze()
        learn.load(save_name)
        if isinstance(label_cols, list) and len(label_cols) > 1:
            learn.loss_func = OneHotCrossEntropy()
        print(data_tst.one_batch())
        results = learn.validate(data_tst.valid_dl)
        print(f"Loss and accuracy using ({save_name}):", results)
        return list(map(float, results))

    def create_cls_learner(self, data_clas, dps=None, label_smoothing_eps=0.0, **kwargs):
        assert self.bidir == False, "bidirectional model is not yet supported"
        config = dict(emb_sz=self.emb_sz, n_hid=self.nh, n_layers=self.nl, pad_token=PAD_TOKEN_ID, qrnn=self.qrnn)
        config.update(dps or self.dps)
        trn_args=dict(bptt=self.bptt, clip=self.clip)
        trn_args.update(kwargs)
        learn = text_classifier_learner(data_clas, AWD_LSTM, config=config,
            pretrained=False, path=self.model_dir.parent, model_dir=self.model_dir.name, **trn_args)

        if self.pretrained_model is not None:
            print("Loading pretrained model")
            model_path = untar_data(self.pretrained_model, data=False)
            fnames = [list(model_path.glob(f'*.{ext}'))[0] for ext in ['pth', 'pkl']]
            learn.load_pretrained(*fnames, strict=False)
            learn.freeze()

        learn.callback_fns += [partial(CSVLogger, filename=f"{learn.model_dir}/cls-history"),
                               #partial(SaveModelCallback, every='improvement', name='cls_best') disabled due to memory issues
                               ]
        if label_smoothing_eps > 0.0:
            learn.loss_func = FlattenedLoss(LabelSmoothingCrossEntropy, eps=label_smoothing_eps)
        return learn

    def load_cls_data(self, bs, **kwargs):
        self.model_dir.mkdir(exist_ok=True, parents=True)
        add_trn_to_lm = True
        lang = self.lang
        use_moses = True
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

    def merge_cols(self, df, label_cols, text_cols):
        if len(text_cols) < 2:
            return df
        print(f"merging: {label_cols} {text_cols}")
        text_col = text_cols[0]
        ndf = df[label_cols+[text_col]].copy().fillna(" ")
        for i in text_cols[1:]:
            ndf[text_col] += ("\n" + FLD + "\n") + df[i].fillna(" ")

        assert ndf[text_col].isna().sum().sum() == 0, f"You have NaN values in column(s) of your dataframe, please fix it."
        print(ndf.head())
        return ndf

    def load_data(self, lang='', label_cols=0, text_cols=1, **kwargs):
        if isinstance(label_cols, int): label_cols = [label_cols]
        if isinstance(text_cols, int): text_cols = [text_cols]

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
        for nm, df in zip(['trn_df', 'val_df', 'tst_df', 'unsup_df'], [trn_df, val_df, tst_df, unsup_df]):
            for lbl_col in label_cols:
                assert np.issubdtype(df[lbl_col].dtype, np.number), f'Column {lbl_col} of {nm} should be numeric'
            for txt_col in text_cols:
                assert not np.issubdtype(df[txt_col], np.number), f'Column {lbl_col} of {nm} should not be numeric, perhaps'
            if set(label_cols + text_cols) != set(df.columns):
                print(f"WARN Columns {set(df.columns) - set(label_cols + text_cols)} of {nm} are not used")
        trn_df = self.merge_cols(trn_df, label_cols, text_cols)
        val_df = self.merge_cols(val_df, label_cols, text_cols)
        tst_df = self.merge_cols(tst_df, label_cols, text_cols)
        unsup_df = self.merge_cols(unsup_df, label_cols, text_cols)
        kwargs.update(dict(trn_df=trn_df, val_df=val_df, tst_df=tst_df, unsup_df=unsup_df))
        kwargs['label_cols'] = label_cols
        kwargs['text_merged_col'] = text_cols[0]
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

    def databunches(self, bs, trn_df, val_df, tst_df, unsup_df, add_trn_to_lm=True, use_moses=False, force=False, limit=None, noise=0.0, label_cols=0, text_merged_col=1):
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

        args = self.tokenizer_to_fastai_args(sp_data_func=lambda: trn_df[text_merged_col], use_moses=use_moses)
        lm_suffix = self.bptt if self.bptt != 70 else ""
        data_lm = self.lm_databunch(f'lm{lm_suffix}', train_df=lm_trn_df, valid_df=lm_val_df, bs=bs, force=force, bptt=self.bptt, label_cols=label_cols, text_cols=text_merged_col, **args)
        args['vocab'] = data_lm.vocab
        data_cls = self.cls_databunch(cls_name, train_df=trn_df, valid_df=val_df, bs=bs, force=force, label_cols=label_cols, text_cols=text_merged_col, **args)
        data_tst = self.cls_databunch('tst', train_df=val_df, valid_df=tst_df, bs=bs, force=force, label_cols=label_cols, text_cols=text_merged_col, **args) # Hack to load test dataset with labels

        print('Size of vocabulary:', len(data_lm.vocab.itos))
        print('First 20 words in vocab:', data_lm.vocab.itos[:20])
        return data_cls, data_lm, data_tst

    def cls_databunch(self, name, *args, **kwargs):
        return self.databunch(name, bunch_class=TextClasDataBunch, *args, **kwargs)

if __name__ == '__main__':
    fire.Fire(CLSHyperParams)

##

