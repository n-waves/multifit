
#
```
 export CUDA_VISIBLE_DEVICES=0
LANG=zh
python -m ulmfit lm --dataset-path data/wiki/${LANG}-100 --tokenizer='sp' --nl 4 --name 'nl4' --max-vocab 15000 --lang ${LANG} --qrnn=True - train 10 --bs=50 --drop_mult=0

Wiki text was split to 103929 articles
Wiki text was split to 113 articles
Running tokenization lm...
Data lm, trn: 103929, val: 113
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁的', '▁。', '▁年', '▁、', '▁在', '▁一', '▁是', '▁中', '▁有']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
/home/pczapla/anaconda3/envs/fastaiv1/lib/python3.7/site-packages/torch/utils/cpp_extension.py:152: UserWarning:

                               !! WARNING !!

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
Your compiler (c++) may be ABI-incompatible with PyTorch!
Please use a compiler that is ABI-compatible with GCC 4.9 and above.
See https://gcc.gnu.org/onlinedocs/libstdc++/manual/abi.html.

See https://gist.github.com/goldsborough/d466f43e8ffc948ff92de7486c5216d6
for instructions on how to install GCC 4.9 or higher.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                              !! WARNING !!

  warnings.warn(ABI_INCOMPATIBILITY_WARNING.format(compiler))
Training lm from random weights
epoch     train_loss  valid_loss  accuracy
1         2.489521    2.734049    0.482433
2         2.427567    2.662464    0.488089
3         2.415744    2.613971    0.494118
4         2.334062    2.560180    0.501209
5         2.343723    2.503271    0.507307
6         2.260171    2.444533    0.516768
7         2.198721    2.367407    0.526631
8         2.161857    2.308182    0.535856
9         2.142125    2.252678    0.544535
10        2.087831    2.234440    0.548529
Total time: 11:01:47
data/wiki/zh-100/models/sp15k
Saving info data/wiki/zh-100/models/sp15k/qrnn_nl4.m/info.json
```

## MLDoc
```bash
export CUDA_VISIBLE_DEVICES=0
LANG=zh
python -m ulmfit cls --dataset-path data/mldoc/${LANG}-1  --base-lm-path data/wiki/${LANG}-100/models/sp15k/qrnn_nl4.m  --lang=${LANG} --name 'nl4' - train 20 --bs 20 --num-cls-epochs=8

Training lm from:  [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/wiki/zh-100/models/sp15k/qrnn_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/wiki/zh-100/models/sp15k/qrnn_nl4.m/../itos')]
epoch     train_loss  valid_loss  accuracy
1         2.723684    2.148748    0.571206
Total time: 02:13
epoch     train_loss  valid_loss  accuracy
1         2.157829    1.937637    0.601026
2         1.898958    1.712967    0.637379
3         1.722818    1.547745    0.664276
4         1.570266    1.427551    0.682546
5         1.503477    1.344690    0.696379
6         1.434701    1.289549    0.704813
7         1.425267    1.217570    0.717714
8         1.373606    1.174655    0.725217
9         1.297397    1.116406    0.735997
10        1.211259    1.062999    0.745848
11        1.248108    1.024482    0.754134
12        1.198918    0.980273    0.762664
13        1.121848    0.937985    0.771961
14        1.111386    0.898821    0.780796
15        1.120596    0.866009    0.787908
16        1.056925    0.836998    0.794833
17        1.020636    0.816387    0.799694
18        1.002068    0.802623    0.802859
19        0.998480    0.796877    0.804212
20        0.959919    0.794685    0.804594
Total time: 1:02:57
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp15k/qrnn_nl4.m/info.json
Starting classifier training
epoch     train_loss  valid_loss  accuracy
1         0.666322    0.433893    0.855000
Total time: 00:08
epoch     train_loss  valid_loss  accuracy
1         0.448371    0.317440    0.889000
Total time: 00:09
epoch     train_loss  valid_loss  accuracy
1         0.336693    0.309876    0.900000
Total time: 00:10
epoch     train_loss  valid_loss  accuracy
1         0.266735    0.302003    0.903000
2         0.222821    0.294501    0.905000
3         0.207295    0.293751    0.908000
4         0.179668    0.296945    0.911000
5         0.153803    0.293158    0.911000
Traceback (most recent call last):
  File "/home/pczapla/anaconda3/envs/fastaiv1/lib/python3.7/runpy.py", line 193, in _run_module_as_main
    "__main__", mod_spec)
  File "/home/pczapla/anaconda3/envs/fastaiv1/lib/python3.7/runpy.py", line 85, in _run_code
    exec(code, run_globals)
  File "/home/pczapla/workspace/ulmfit-multilingual/ulmfit/__main__.py", line 73, in <module>
    fire.Fire(ULMFiT())
  File "/home/pczapla/anaconda3/envs/fastaiv1/lib/python3.7/site-packages/fire/core.py", line 127, in Fire
    component_trace = _Fire(component, args, context, name)
  File "/home/pczapla/anaconda3/envs/fastaiv1/lib/python3.7/site-packages/fire/core.py", line 366, in _Fire
    component, remaining_args)
  File "/home/pczapla/anaconda3/envs/fastaiv1/lib/python3.7/site-packages/fire/core.py", line 542, in _CallCallable
    result = fn(*varargs, **kwargs)
  File "/home/pczapla/workspace/ulmfit-multilingual/ulmfit/train_clas.py", line 54, in train_cls
    learn.fit_one_cycle(num_cls_epochs, slice(1e-3 / (2.6 ** 4), 1e-3), moms=(0.8, 0.7))
  File "/home/pczapla/workspace/_oss/fastai/fastai/fastai/train.py", line 22, in fit_one_cycle
    learn.fit(cyc_len, max_lr, wd=wd, callbacks=callbacks)
  File "/home/pczapla/workspace/_oss/fastai/fastai/fastai/basic_train.py", line 178, in fit
    callbacks=self.callbacks+callbacks)
  File "/home/pczapla/workspace/_oss/fastai/fastai/fastai/utils/mem.py", line 77, in wrapper
    return func(*args, **kwargs)
  File "/home/pczapla/workspace/_oss/fastai/fastai/fastai/basic_train.py", line 90, in fit
    loss = loss_batch(model, xb, yb, loss_func, opt, cb_handler)
  File "/home/pczapla/workspace/_oss/fastai/fastai/fastai/basic_train.py", line 20, in loss_batch
    out = model(*xb)
  File "/home/pczapla/anaconda3/envs/fastaiv1/lib/python3.7/site-packages/torch/nn/modules/module.py", line 477, in __call__
    result = self.forward(*input, **kwargs)
  File "/home/pczapla/anaconda3/envs/fastaiv1/lib/python3.7/site-packages/torch/nn/modules/container.py", line 92, in forward
    input = module(input)
  File "/home/pczapla/anaconda3/envs/fastaiv1/lib/python3.7/site-packages/torch/nn/modules/module.py", line 477, in __call__
    result = self.forward(*input, **kwargs)
  File "/home/pczapla/workspace/_oss/fastai/fastai/fastai/text/learner.py", line 235, in forward
    return self.concat(raw_outputs), self.concat(outputs)
  File "/home/pczapla/workspace/_oss/fastai/fastai/fastai/text/learner.py", line 221, in concat
    return [torch.cat([l[si] for l in arrs], dim=1) for si in range_of(arrs[0])]
  File "/home/pczapla/workspace/_oss/fastai/fastai/fastai/text/learner.py", line 221, in <listcomp>
    return [torch.cat([l[si] for l in arrs], dim=1) for si in range_of(arrs[0])]
RuntimeError: CUDA error: out of memory
```

## Fixed sentence piece 
LANG=zh
python -m ulmfit lm --dataset-path data/wiki/${LANG}-100 --tokenizer='sp' --nl 4 --name 'nl4' --max-vocab 15000 --lang ${LANG} --qrnn=True "--tokenizer-mod=-fix" --lmseed=1 - train 10 --bs=50 --drop_mult=0
