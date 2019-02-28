# Results on books dataset

|                         |   de  |   fr  |
|-------------------------|-------|-------|
| laser zero shot from en | 84.15 | 83.90 |
| with ULMFIT QRNN sp15k  | 89.60 | 87.84 |

## Laser results

|      |    en  |   de  |   fr |
|------|--------|-------|------|
| en:  |  84.55 | 84.15 | 83.90|
| de:  |  82.60 | 85.20 | 83.05|
| fr:  |  77.20 | 82.95 | 84.85|

## ULMFiT improvment
```
data/cls/de-books-laser-en1/models/sp15k/qrnn_nl4.m: 0.8960000276565552
data/cls/fr-books-laser-en1/models/sp15k/qrnn_nl4.m: 0.8784999847412109
```

### Execution log
```
python -m ulmfit eval --glob="cls/*-books/models/sp15k/qrnn_nl4.m" --name nl4 --dataset-template='${lang}-books-laser-en1' --num-lm-epochs=0  --num-cls-epochs=8  --bs=18 --lr_sched=1cycle --label-smoothing-eps=0.1
Processing data/cls/de-books/models/sp15k/qrnn_nl4.m
de-books-laser-en1
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/cls/de-books-laser-en1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/cls/de-books-laser-en1/models/sp15k/qrnn_nl4.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/cls/de-books-laser-en1/de.dev.csv
Running tokenization lm...
Data lm, trn: 152523, val: 16947
Running tokenization cls...
Data cls, trn: 1800, val: 200
Running tokenization tst...
Data tst, trn: 200, val: 2000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁.', '▁,', '▁der', '▁die', 'en', '▁und', 's', '▁in', 'er', "▁&'"]
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
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
Loading pretrained model
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/cls/de-books-laser-en1/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/cls/de-books-laser-en1/models/sp15k/qrnn_nl4.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.541837    0.487596    0.870000
2         0.498016    0.490300    0.885000
3         0.442417    0.479205    0.875000
4         0.395640    0.528897    0.855000
5         0.369408    0.521830    0.855000
6         0.361129    0.481892    0.880000
7         0.351095    0.481634    0.885000
8         0.343147    0.481654    0.880000
Total time: 02:35
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/cls/de-books-laser-en1/models/sp15k/qrnn_nl4.m
Loss and accuracy using (cls_best): [0.29498395, tensor(0.8960)]
Processing data/cls/en-books/models/sp15k/qrnn_nl4.m
en-books-laser-en1
Processing data/cls/fr-books/models/sp15k/qrnn_nl4.m
fr-books-laser-en1
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/cls/fr-books-laser-en1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/cls/fr-books-laser-en1/models/sp15k/qrnn_nl4.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/cls/fr-books-laser-en1/fr.dev.csv
Running tokenization lm...
Data lm, trn: 33183, val: 3687
Running tokenization cls...
Data cls, trn: 1800, val: 200
Running tokenization tst...
Data tst, trn: 200, val: 2000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', 's', '▁.', "'", '▁la', '▁le', '▁et', '▁l', '▁à']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/cls/fr-books-laser-en1/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/cls/fr-books-laser-en1/models/sp15k/qrnn_nl4.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.551931    0.532421    0.835000
2         0.509913    0.524893    0.880000
3         0.433385    0.502657    0.860000
4         0.397314    0.487201    0.880000
5         0.365447    0.467523    0.885000
6         0.356587    0.520736    0.855000
7         0.353801    0.487093    0.875000
8         0.343812    0.484453    0.880000
Total time: 01:45
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/cls/fr-books-laser-en1/models/sp15k/qrnn_nl4.m
Loss and accuracy using (cls_best): [0.32666296, tensor(0.8785)]
Processing data/cls/ja-books/models/sp15k/qrnn_nl4.m
ja-books-laser-en1
OrderedDict([('data/cls/de-books-laser-en1/models/sp15k/qrnn_nl4.m',
              0.8960000276565552),
             ('data/cls/fr-books-laser-en1/models/sp15k/qrnn_nl4.m',
              0.8784999847412109)])
data/cls/de-books-laser-en1/models/sp15k/qrnn_nl4.m: 0.8960000276565552
data/cls/fr-books-laser-en1/models/sp15k/qrnn_nl4.m: 0.8784999847412109
```

