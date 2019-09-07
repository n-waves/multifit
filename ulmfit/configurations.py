from .training import *

def multifit1552_fp32(bs=64):
    self = ULMFiT()
    self.replace_(
        label_smoothing_eps=0.0,
        true_wd=True,
        wd=0.1,
        seed=0,
        fp16=False,
        bs=bs,
        use_adam_08=False,
        early_stopping=None,
        name=multifit1552_fp32.__name__
    )
    self.arch.replace_(
        tokenizer='fsp',
        max_vocab=15000,
        qrnn=True,
        n_layers=4,
        n_hid=1552
    )
    self.pretrain_lm.replace_(num_epochs=10, drop_mult=0.5, lr=(1e-2 * bs / 48))
    self.finetuine_lm.replace_(num_epochs=10, drop_mult=1.0, lr=(1e-3 * bs / 48))
    self.classifier.replace_(num_epochs=8, drop_mult=0.5, bs=20, label_smoothing_eps=0.1)
    return self

multifit_fp32 = multifit1552_fp32

def multifit1552_fp16():
    return multifit1552_fp32(bs=128).replace_(fp16=True, name=multifit1552_fp16.__name__)

multifit_fp16 = multifit1552_fp16

def multifit_lstm():
    return multifit1552_fp32(bs=128).replace_(qrnn=False, n_hid=1552, name=multifit_lstm.__name__)

def multifit_paper_version():
    self = multifit1552_fp32()
    self.replace_(
        seed=None,
        name="multifit_paper_version"
    )
    self.arch.replace_(
        tokenizer='sp', # sentence piece model that prefixes each control token with space token
        n_hid=1550
    )
    self.pretrain_lm.replace_(drop_mult=0.0, lr=5e-3, use_adam_08=True, true_wd=False, wd=1e-7, bs=50,)
    self.finetuine_lm.replace_(drop_mult=0.3, lr=1e-3, num_epochs=20, true_wd=False, wd=1e-7, bs=20)
    self.classifier.replace_(early_stopping='accuracy', bs=20)
    return self

def ulmfit_orig():
    self = multifit_paper_version()
    self.replace_(
        seed=None,
        name="ulfmfit"
    )
    self.arch.replace_(
        tokenizer='f',
        max_vocab=60000,
        qrnn=False,
        n_layers=3,
        n_hid=1150
    )
    return self
