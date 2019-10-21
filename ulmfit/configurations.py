import inspect
from .training import *
__all__ = [
    'multifit_paper_version',
    'multifit1552_fp32', 'multifit_fp32',
    'multifit_fp32_nl3',

    'multifit1552_fp16','multifit_fp16',
    'multifit_fp16_nl3',
    'multifit1552_fp16_nl3_large',

    'multifit_lstm',
    'multifit1152_lstm_nl3',
    'multifit1152_lstm_nl3_fp16_large',
]

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
        name=_use_caller_name()
    )
    self.arch.replace_(
        tokenizer_type='fsp',
        max_vocab=15000,
        qrnn=True,
        n_layers=4,
        n_hid=1552
    )
    self.pretrain_lm.replace_(num_epochs=10, drop_mult=0.5, lr=(1e-2 * bs / 48))
    self.finetune_lm.replace_(num_epochs=10, drop_mult=1.0, lr=(1e-3 * bs / 48))
    self.classifier.replace_(num_epochs=8, drop_mult=0.5, bs=20, label_smoothing_eps=0.1)
    return self

multifit_fp32 = multifit1552_fp32

def multifit_fp32_nl3():
    return multifit1552_fp32().replace_(n_layers=3, name=_use_caller_name())

# FP16

def multifit1552_fp16():
    return multifit1552_fp32(bs=128).replace_(fp16=True, name=_use_caller_name())

def multifit1552_fp16_nl3_large():
    return multifit1552_fp32(bs=448).replace_(fp16=True, n_layers=3, num_epochs=20, name=_use_caller_name())

multifit_fp16 = multifit1552_fp16

def multifit_lstm():
    return multifit1552_fp32(bs=128).replace_(qrnn=False, n_hid=1552, name=_use_caller_name())

def multifit1152_lstm_nl3(bs=128):
    return multifit1552_fp32(bs).replace_(qrnn=False, n_hid=1152,  n_layers=3, name=_use_caller_name())

def multifit1152_lstm_nl3_fp16_large():
    return multifit1152_lstm_nl3(bs=448).replace_(fp16=True, num_epochs=20, name=_use_caller_name())

def multifit_fp16_nl3():
    return multifit1552_fp16().replace_(n_layers=3, name=_use_caller_name())

def multifit_paper_version():
    self = multifit1552_fp32()
    self.replace_(
        seed=None,
        name=_use_caller_name()
    )
    self.arch.replace_(
        n_hid=1550
        tokenizer_type='sp',
    )
    self.pretrain_lm.replace_(drop_mult=0.0, lr=5e-3, use_adam_08=True, true_wd=False, wd=1e-7, bs=50,)
    self.finetune_lm.replace_(drop_mult=0.3, lr=1e-3, num_epochs=20, true_wd=False, wd=1e-7, bs=20)
    self.classifier.replace_(early_stopping='accuracy', bs=20)
    return self

def ulmfit_orig():
    self = multifit_paper_version()
    self.replace_(
        seed=None,
        name=_use_caller_name()
    )
    self.arch.replace_(
        tokenizer_type='f',
        max_vocab=60000,
        qrnn=False,
        n_layers=3,
        n_hid=1150
    )
    return self


def _use_caller_name():
    return inspect.stack()[1].function