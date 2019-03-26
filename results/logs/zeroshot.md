## Laser Perforamnce

Accuracy matrix:
### 1k

| Train |   en  |   de  |   es  |   fr  |   it  |   ru  |   zh |
|-------|-------|-------|-------|-------|-------|-------|-------|
| en:   | 91.48 | 87.65 | 75.48 | 84.00 | 71.18 | 66.58 | 76.65 |
| de:   | 78.23 | 93.50 | 81.40 | 81.50 | 74.53 | 64.58 | 73.20 |
| es:   | 71.62 | 84.00 | 93.73 | 78.90 | 73.38 | 53.33 | 55.83 |
| fr:   | 81.30 | 88.75 | 80.12 | 90.85 | 72.58 | 67.35 | 79.40 |
| it:   | 74.33 | 83.53 | 80.58 | 79.78 | 84.48 | 66.45 | 63.35 |
| ru:   | 72.38 | 81.65 | 65.73 | 71.30 | 63.33 | 85.45 | 59.58 |
| zh:   | 74.98 | 81.35 | 72.20 | 73.28 | 70.08 | 66.23 | 88.30 |

### 10k
 
| Train |   en  |   de  |   es  |   fr  |   it  |   ru  |   zh |
|-------|-------|-------|-------|-------|-------|-------|-------|
|  en:  | 92.70 | 87.43 | 77.38 | 78.70 | 72.53 | 67.70 | 75.18 |
|  de:  | 81.60 | 95.40 | 83.50 | 82.85 | 76.60 | 68.80 | 73.12 |
|  es:  | 73.48 | 87.13 | 94.40 | 81.63 | 76.70 | 58.65 | 72.98 |
|  fr:  | 85.08 | 91.65 | 81.05 | 93.65 | 75.08 | 70.73 | 76.33 |
|  it:  | 76.75 | 86.68 | 82.55 | 82.65 | 87.80 | 65.90 | 73.35 |
|  ru:  | 75.23 | 81.88 | 66.83 | 68.60 | 67.38 | 87.00 | 62.68 |
|  zh:  | 76.05 | 82.05 | 68.40 | 77.38 | 68.45 | 66.88 | 90.38 |



### laser 1k 
#### Building dataset
```bash
for SRC_LANG in en de fr; do                                                                                                                                         ✘ 130
    for LANG in en de es fr it ru zh; do
        echo $LANG from $SRC_LANG
        python ../../source/classify.py embed10000/mldoc.${SRC_LANG}-${SRC_LANG}.h5 ~/workspace/ulmfit-multilingual/data/mldoc/${LANG}-10 --suffix=10 | grep Test:
    done
done
```

```
for SRC_LANG in en de fr; do                                                                                                                                         ✘ 130
    for LANG in en de es fr it ru zh; do
        echo $LANG from $SRC_LANG
        python ../../source/classify.py embed1000/mldoc.${SRC_LANG}-${SRC_LANG}.h5 ~/workspace/ulmfit-multilingual/data/mldoc/${LANG}-1 | grep Test:
    done
done

en from en
 | Test: 91.48% | classes: 23.77 24.90 26.25 25.07
de from en
 | Test: 87.65% | classes: 21.98 24.45 27.65 25.93
es from en
 | Test: 75.48% | classes: 21.60 15.82 22.10 40.48
fr from en
 | Test: 84.00% | classes: 23.18 29.12 27.90 19.80
it from en
 | Test: 71.18% | classes: 23.65 22.88 25.68 27.80
ru from en
 | Test: 66.58% | classes: 29.48 13.78 34.52 22.23
zh from en
 | Test: 76.65% | classes: 30.25 31.30 13.93 24.52
en from de
 | Test: 78.23% | classes: 31.80 17.73 30.15 20.32
de from de
 | Test: 93.50% | classes: 24.45 25.45 26.00 24.10
es from de
 | Test: 81.40% | classes: 24.15 25.77 20.12 29.95
fr from de
 | Test: 81.50% | classes: 25.52 29.45 27.45 17.57
it from de
 | Test: 74.53% | classes: 24.70 27.25 22.43 25.62
ru from de
 | Test: 64.58% | classes: 45.62  9.12 26.73 18.52
zh from de
 | Test: 73.20% | classes: 31.20 43.38  7.60 17.82
en from fr
 | Test: 81.30% | classes: 28.95 18.02 24.98 28.05
de from fr
 | Test: 88.75% | classes: 24.00 23.75 24.85 27.40
es from fr
 | Test: 80.12% | classes: 24.50 14.82 18.40 42.27
fr from fr
 | Test: 90.85% | classes: 24.50 24.75 24.68 26.07
it from fr
 | Test: 72.58% | classes: 25.45 24.10 17.50 32.95
ru from fr
 | Test: 67.35% | classes: 47.15 13.62 16.68 22.55
zh from fr
 | Test: 79.40% | classes: 33.60 31.12  9.07 26.20
```
#### Fixed label smoohting

```
                                                 name  tst_accuracy  tst_loss  val_accuracy  val_loss
0   data/mldoc/de-1-laser-en1/models/sp15k/qrnn_nl...       0.91625  0.295122         0.922  0.256934
1   data/mldoc/de-1-laser-en1/models/sp15k/qrnn_nl4.m       0.91325  0.375667         0.910  0.357531
2   data/mldoc/es-1-laser-en1/models/sp15k/qrnn_nl...       0.76725  1.266754         0.872  0.482154
3   data/mldoc/es-1-laser-en1/models/sp15k/qrnn_nl...       0.78425  1.213045         0.876  0.478503
4   data/mldoc/es-1-laser-en1/models/sp15k/qrnn_nl4.m       0.79100  0.828614         0.878  0.439474
5   data/mldoc/fr-1-laser-en1/models/sp15k/qrnn_nl...       0.87125  0.406994         0.877  0.358970
6   data/mldoc/fr-1-laser-en1/models/sp15k/qrnn_nl4.m       0.89425  0.384984         0.888  0.405739
7   data/mldoc/it-1-laser-en1/models/sp15k/qrnn_nl...       0.75850  1.014424         0.815  0.579695
8   data/mldoc/it-1-laser-en1/models/sp15k/qrnn_nl4.m       0.76025  0.808285         0.818  0.555738
9   data/mldoc/ru-1-laser-en1/models/sp15k/qrnn_nl...       0.67925  1.588188         0.841  0.541397
10  data/mldoc/ru-1-laser-en1/models/sp15k/qrnn_nl4.m       0.68125  1.069047         0.838  0.536006
11  data/mldoc/zh-1-laser-en1/models/sp15k/qrnn_nl...       0.81450  0.624627         0.815  0.643505
12  data/mldoc/zh-1-laser-en1/models/sp15k/qrnn_nl4.m       0.82475  0.582873         0.820  0.570754
ds    de-1-laser-en1  es-1-laser-en1  fr-1-laser-en1  it-1-laser-en1  ru-1-laser-en1  zh-1-laser-en1
best           91.62           79.10           89.42           76.02           67.93           82.48
max            91.62           79.10           89.42           76.02           68.12           82.48
avg            91.48           78.08           88.27           75.94           68.02           81.96
Saving result to: laser-en1-results.csv
```
#### JA 
```bash
                                                name  tst_accuracy  tst_loss  val_accuracy  val_loss
0  data/mldoc/ja-1-laser-en1/models/sp15k/qrnn_nl...       0.68500  1.000403         0.722  0.847306
1  data/mldoc/ja-1-laser-en1/models/sp15k/qrnn_tls.m       0.65625  1.421663         0.788  0.749413
2  data/mldoc/ja-1-laser-en1/models/sp15k/qrnn_tl...       0.68125  1.126172         0.786  0.650364
3  data/mldoc/ja-1-laser-en1/models/sp15k/qrnn_tl...       0.68625  1.229256         0.784  0.674508
4  data/mldoc/ja-1-laser-en1/models/sp15k/qrnn_tl...       0.69575  1.110130         0.800  0.644522
5  data/mldoc/ja-1-laser-en1/models/sp15k/qrnn_tl...       0.69650  1.082014         0.767  0.672668
ds    ja-1-laser-e
best         69.57
max          69.65
avg          68.35
```
##### ES labels from laser-EN10k
```
python -m ulmfit eval --glob="mldoc/es-1/models/sp15k/qrnn_nl4.m" --dataset_template='${lang}-1-laser-en1' --name nl4 --num-cls-epochs=8 --label-smoothing-eps=0.1 --lr_sched=1cycle --bs=18
Processing data/mldoc/es-1/models/sp15k/qrnn_nl4.m
es-1-laser-en1
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1-laser-en1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1-laser-en1/models/sp15k/qrnn_nl4.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1-laser-en1/es.dev.csv
Running tokenization lm...
Data lm, trn: 13013, val: 1445
Running tokenization cls...
Data cls, trn: 1000, val: 1000
Running tokenization tst...
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁la', '▁.', 's', '▁en', '▁el', '▁y', '▁a', '▁que']
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
Bptt 70
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1-laser-en1/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1-laser-en1/models/sp15k/qrnn_nl4.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.755567    0.673722    0.822000
2         0.641650    0.623495    0.854000
3         0.526631    0.626180    0.857000
4         0.442634    0.789478    0.837000
5         0.341413    0.623467    0.859000
6         0.254876    0.599503    0.875000
7         0.202478    0.573548    0.870000
8         0.179376    0.564544    0.870000
Total time: 01:56
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1-laser-en1/models/sp15k/qrnn_nl4.m
Loss and accuracy using (cls_best): [1.0270673, tensor(0.7983)] [0.4585123, tensor(0.8700)]
                                                name  tst_accuracy  tst_loss  val_accuracy  val_loss
0  data/mldoc/es-1-laser-en1/models/sp15k/qrnn_nl4.m       0.79825  1.027067          0.87  0.458512
ds    es-1-laser-en1
best           79.83
max            79.83
avg            79.83
```
#### ULMFit zershot on laser-en1k 4 epochs
##### missing RU
```
 python -m ulmfit eval --glob="mldoc/*-1/models/sp15k/qrnn_nl4.m" --dataset_template='${lang}-1-laser-en1' --name nl4 --num-cls-epochs=4 --label-smoothing-eps=0.1 --lr_sched=1cycle --bs=18
Processing data/mldoc/de-1/models/sp15k/qrnn_nl4.m
de-1-laser-en1
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1-laser-en1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1-laser-en1/models/sp15k/qrnn_nl4.m
Evaluating previously trained model
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1-laser-en1/de.dev.csv
Data lm, trn: 13500, val: 1500
Data cls, trn: 1000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁.', '▁,', '▁der', '▁die', 'en', '▁und', 's', '▁in', 'er', "▁&'"]
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
Loss and accuracy using (cls_best): [0.3181207, tensor(0.9133)]
Processing data/mldoc/en-1/models/sp15k/qrnn_nl4.m
en-1-laser-en1
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/en-1-laser-en1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/en-1-laser-en1/models/sp15k/qrnn_nl4.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/en-1-laser-en1/en.dev.csv
Running tokenization lm...
Data lm, trn: 13500, val: 1500
Running tokenization cls...
Data cls, trn: 1000, val: 1000
Running tokenization tst...
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁the', '▁,', 's', '▁.', '▁of', '▁and', '▁in', '▁to', '▁a', 'ed']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
Bptt 70
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/en-1-laser-en1/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/en-1-laser-en1/models/sp15k/qrnn_nl4.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.850595    0.667213    0.919000
2         0.698336    0.648402    0.927000
3         0.588279    0.600404    0.936000
4         0.529861    0.581380    0.937000
Total time: 01:10
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/en-1-laser-en1/models/sp15k/qrnn_nl4.m
Loss and accuracy using (cls_best): [0.21109423, tensor(0.9490)]
Processing data/mldoc/es-1/models/sp15k/qrnn_nl4.m
es-1-laser-en1
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1-laser-en1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1-laser-en1/models/sp15k/qrnn_nl4.m
Evaluating previously trained model
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1-laser-en1/es.dev.csv
Data lm, trn: 13013, val: 1445
Data cls, trn: 1000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁la', '▁.', 's', '▁en', '▁el', '▁y', '▁a', '▁que']
Loss and accuracy using (cls_best): [0.7726423, tensor(0.7910)]
Processing data/mldoc/fr-1/models/sp15k/qrnn_nl4.m
fr-1-laser-en1
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1-laser-en1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1-laser-en1/models/sp15k/qrnn_nl4.m
Evaluating previously trained model
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1-laser-en1/fr.dev.csv
Data lm, trn: 13500, val: 1500
Data cls, trn: 1000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', 's', '▁.', "'", '▁la', '▁le', '▁et', '▁l', '▁à']
Loss and accuracy using (cls_best): [0.3214729, tensor(0.8942)]
Processing data/mldoc/it-1/models/sp15k/qrnn_nl4.m
it-1-laser-en1
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1-laser-en1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1-laser-en1/models/sp15k/qrnn_nl4.m
Evaluating previously trained model
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1-laser-en1/it.dev.csv
Data lm, trn: 13500, val: 1500
Data cls, trn: 1000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁.', '▁di', '▁e', "▁&'", "'", '▁il', '▁la', '▁in', 'e']
Loss and accuracy using (cls_best): [0.75963426, tensor(0.7602)]
Processing data/mldoc/ja-1/models/sp15k/qrnn_nl4.m
ja-1-laser-en1
Processing data/mldoc/ru-1/models/sp15k/qrnn_nl4.m
ru-1-laser-en1
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1-laser-en1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1-laser-en1/models/sp15k/qrnn_nl4.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1-laser-en1/ru.dev.csv
Running tokenization lm...
Data lm, trn: 9195, val: 1021
Running tokenization cls...
Data cls, trn: 1000, val: 1000
Running tokenization tst...
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁.', '▁в', 'а', 'и', 'е', '▁и', 'й', '▁на', 'х']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
Bptt 70
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1-laser-en1/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1-laser-en1/models/sp15k/qrnn_nl4.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.053947    0.859605    0.785000
2         0.882559    0.793324    0.836000
3         0.713290    0.812535    0.834000
4         0.620544    0.801801    0.837000
Total time: 01:35
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1-laser-en1/models/sp15k/qrnn_nl4.m
Loss and accuracy using (cls_best): [1.0211968, tensor(0.6820)]
Processing data/mldoc/zh-1/models/sp15k/qrnn_nl4.m
zh-1-laser-en1
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1-laser-en1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1-laser-en1/models/sp15k/qrnn_nl4.m
Evaluating previously trained model
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1-laser-en1/zh.dev.csv
Data lm, trn: 13500, val: 1500
Data cls, trn: 1000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁的', '▁。', '▁年', '▁、', '▁在', '▁一', '▁是', '▁中', '▁有']
Loss and accuracy using (cls_best): [0.52925, tensor(0.8248)]
OrderedDict([('data/mldoc/de-1-laser-en1/models/sp15k/qrnn_nl4.m',
              0.9132500290870667),
             ('data/mldoc/en-1-laser-en1/models/sp15k/qrnn_nl4.m',
              0.9490000009536743),
             ('data/mldoc/es-1-laser-en1/models/sp15k/qrnn_nl4.m',
              0.7910000085830688),
             ('data/mldoc/fr-1-laser-en1/models/sp15k/qrnn_nl4.m',
              0.8942499756813049),
             ('data/mldoc/it-1-laser-en1/models/sp15k/qrnn_nl4.m',
              0.7602499723434448),
             ('data/mldoc/ru-1-laser-en1/models/sp15k/qrnn_nl4.m',
              0.6819999814033508),
             ('data/mldoc/zh-1-laser-en1/models/sp15k/qrnn_nl4.m',
              0.8247500061988831)])
data/mldoc/de-1-laser-en1/models/sp15k/qrnn_nl4.m: 0.9132500290870667
data/mldoc/en-1-laser-en1/models/sp15k/qrnn_nl4.m: 0.9490000009536743
data/mldoc/es-1-laser-en1/models/sp15k/qrnn_nl4.m: 0.7910000085830688
data/mldoc/fr-1-laser-en1/models/sp15k/qrnn_nl4.m: 0.8942499756813049
data/mldoc/it-1-laser-en1/models/sp15k/qrnn_nl4.m: 0.7602499723434448
data/mldoc/ru-1-laser-en1/models/sp15k/qrnn_nl4.m: 0.6819999814033508
data/mldoc/zh-1-laser-en1/models/sp15k/qrnn_nl4.m: 0.8247500061988831
```
##### Other LAngs

```bash
python -m ulmfit eval --glob="mldoc/*-1/models/sp15k/qrnn_nl4.m" --dataset_template="{lang}-1*-laser-en1" --name nl4 --num-cls-epochs=4 --label-smoothing-eps=0.1 --lr_sched=1cycle --bs=18
 python -m ulmfit eval --glob="mldoc/*-1/models/sp15k/qrnn_nl4.m" --dataset_template='${lang}-1*-laser-en1' --name nl4 --num-cls-epochs=4 --label-smoothing-eps=0.1 --lr_sched=1cycle --bs=18
Processing data/mldoc/de-1/models/sp15k/qrnn_nl4.m
de-1*-laser-en1
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1-laser-en1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1-laser-en1/models/sp15k/qrnn_nl4.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1-laser-en1/de.dev.csv
Running tokenization lm...
Data lm, trn: 13500, val: 1500
Running tokenization cls...
Data cls, trn: 1000, val: 1000
Running tokenization tst...
Data tst, trn: 1000, val: 4000
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
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1-laser-en1/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1-laser-en1/models/sp15k/qrnn_nl4.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.858069    0.756738    0.853000
2         0.737577    0.687798    0.909000
3         0.611922    0.615140    0.919000
4         0.544274    0.608824    0.909000
Total time: 01:08
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1-laser-en1/models/sp15k/qrnn_nl4.m
Loss and accuracy using (cls_best): [0.31768194, tensor(0.9133)]
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10-laser-en1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10-laser-en1/models/sp15k/qrnn_nl4.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10-laser-en1/de.dev.csv
Running tokenization lm...
Data lm, trn: 13500, val: 1500
Running tokenization cls...
Data cls, trn: 10000, val: 1000
Running tokenization tst...
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁.', '▁,', '▁der', '▁die', 'en', '▁und', 's', '▁in', 'er', "▁&'"]
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10-laser-en1/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10-laser-en1/models/sp15k/qrnn_nl4.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.665453    0.600660    0.924000
2         0.624768    0.595788    0.922000
3         0.576251    0.580166    0.930000
4         0.520507    0.569867    0.930000
Total time: 09:38
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10-laser-en1/models/sp15k/qrnn_nl4.m
Loss and accuracy using (cls_best): [0.3278069, tensor(0.9190)]
Processing data/mldoc/en-1/models/sp15k/qrnn_nl4.m
en-1*-laser-en1
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/en-1-laser-en1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/en-1-laser-en1/models/sp15k/qrnn_nl4.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/en-1-laser-en1/en.dev.csv
Running tokenization lm...
Data lm, trn: 13500, val: 1500
Running tokenization cls...
Data cls, trn: 1000, val: 1000
Running tokenization tst...
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁the', '▁,', 's', '▁.', '▁of', '▁and', '▁in', '▁to', '▁a', 'ed']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/en-1-laser-en1/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/en-1-laser-en1/models/sp15k/qrnn_nl4.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.836131    0.760151    0.887000
2         0.691355    0.666257    0.905000
3         0.587016    0.603976    0.932000
4         0.529849    0.584768    0.943000
Total time: 01:09
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/en-1-laser-en1/models/sp15k/qrnn_nl4.m
Loss and accuracy using (cls_best): [0.20341124, tensor(0.9503)]
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/en-10-laser-en1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/en-10-laser-en1/models/sp15k/qrnn_nl4.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/en-10-laser-en1/en.dev.csv
Running tokenization lm...
Data lm, trn: 13500, val: 1500
Running tokenization cls...
Data cls, trn: 10000, val: 1000
Running tokenization tst...
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁the', '▁,', 's', '▁.', '▁of', '▁and', '▁in', '▁to', '▁a', 'ed']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/en-10-laser-en1/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/en-10-laser-en1/models/sp15k/qrnn_nl4.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.676697    0.611570    0.931000
2         0.647868    0.629156    0.916000
3         0.578947    0.546653    0.943000
4         0.539242    0.550864    0.949000
Total time: 10:25
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/en-10-laser-en1/models/sp15k/qrnn_nl4.m
Loss and accuracy using (cls_best): [0.2546631, tensor(0.9490)]
Processing data/mldoc/es-1/models/sp15k/qrnn_nl4.m
es-1*-laser-en1
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1-laser-en1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1-laser-en1/models/sp15k/qrnn_nl4.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1-laser-en1/es.dev.csv
Running tokenization lm...
Data lm, trn: 13013, val: 1445
Running tokenization cls...
Data cls, trn: 1000, val: 1000
Running tokenization tst...
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁la', '▁.', 's', '▁en', '▁el', '▁y', '▁a', '▁que']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1-laser-en1/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1-laser-en1/models/sp15k/qrnn_nl4.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.947175    0.774158    0.854000
2         0.827687    0.747528    0.859000
3         0.698278    0.726713    0.881000
4         0.603821    0.729449    0.878000
Total time: 00:58
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1-laser-en1/models/sp15k/qrnn_nl4.m
Loss and accuracy using (cls_best): [0.77420205, tensor(0.7893)]
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10-laser-en1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10-laser-en1/models/sp15k/qrnn_nl4.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10-laser-en1/es.dev.csv
Running tokenization lm...
Data lm, trn: 13013, val: 1445
Running tokenization cls...
Data cls, trn: 9458, val: 1000
Running tokenization tst...
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁la', '▁.', 's', '▁en', '▁el', '▁y', '▁a', '▁que']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10-laser-en1/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10-laser-en1/models/sp15k/qrnn_nl4.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.654948    0.968341    0.863000
2         0.634143    0.844979    0.873000
3         0.559031    0.719894    0.890000
4         0.524448    0.675569    0.887000
Total time: 05:36
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10-laser-en1/models/sp15k/qrnn_nl4.m
Loss and accuracy using (cls_best): [0.70765454, tensor(0.7880)]
Processing data/mldoc/fr-1/models/sp15k/qrnn_nl4.m
fr-1*-laser-en1
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1-laser-en1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1-laser-en1/models/sp15k/qrnn_nl4.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1-laser-en1/fr.dev.csv
Running tokenization lm...
Data lm, trn: 13500, val: 1500
Running tokenization cls...
Data cls, trn: 1000, val: 1000
Running tokenization tst...
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', 's', '▁.', "'", '▁la', '▁le', '▁et', '▁l', '▁à']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1-laser-en1/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1-laser-en1/models/sp15k/qrnn_nl4.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.882636    0.720057    0.851000
2         0.766720    0.790358    0.847000
3         0.661309    0.669355    0.877000
4         0.584607    0.676029    0.889000
Total time: 01:05
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1-laser-en1/models/sp15k/qrnn_nl4.m
Loss and accuracy using (cls_best): [0.32164142, tensor(0.8945)]
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-10-laser-en1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-10-laser-en1/models/sp15k/qrnn_nl4.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-10-laser-en1/fr.dev.csv
Running tokenization lm...
Data lm, trn: 13500, val: 1500
Running tokenization cls...
Data cls, trn: 10000, val: 1000
Running tokenization tst...
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', 's', '▁.', "'", '▁la', '▁le', '▁et', '▁l', '▁à']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-10-laser-en1/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-10-laser-en1/models/sp15k/qrnn_nl4.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.721444    0.714190    0.865000
2         0.684221    0.688866    0.879000
3         0.616377    0.624161    0.904000
4         0.576405    0.630186    0.904000
Total time: 09:37
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-10-laser-en1/models/sp15k/qrnn_nl4.m
Loss and accuracy using (cls_best): [0.39518934, tensor(0.8848)]
Processing data/mldoc/it-1/models/sp15k/qrnn_nl4.m
it-1*-laser-en1
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1-laser-en1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1-laser-en1/models/sp15k/qrnn_nl4.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1-laser-en1/it.dev.csv
Running tokenization lm...
Data lm, trn: 13500, val: 1500
Running tokenization cls...
Data cls, trn: 1000, val: 1000
Running tokenization tst...
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁.', '▁di', '▁e', "▁&'", "'", '▁il', '▁la', '▁in', 'e']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1-laser-en1/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1-laser-en1/models/sp15k/qrnn_nl4.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.963395    0.843734    0.777000
2         0.892776    0.890245    0.785000
3         0.755211    0.799284    0.815000
4         0.629971    0.796769    0.820000
Total time: 00:40
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1-laser-en1/models/sp15k/qrnn_nl4.m
Loss and accuracy using (cls_best): [0.76029295, tensor(0.7600)]
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-10-laser-en1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-10-laser-en1/models/sp15k/qrnn_nl4.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-10-laser-en1/it.dev.csv
Running tokenization lm...
Data lm, trn: 13500, val: 1500
Running tokenization cls...
Data cls, trn: 10000, val: 1000
Running tokenization tst...
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁.', '▁di', '▁e', "▁&'", "'", '▁il', '▁la', '▁in', 'e']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-10-laser-en1/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-10-laser-en1/models/sp15k/qrnn_nl4.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.828429    0.801399    0.820000
2         0.769261    0.786845    0.816000
3         0.717665    0.734139    0.845000
4         0.597664    0.741355    0.843000
Total time: 05:29
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-10-laser-en1/models/sp15k/qrnn_nl4.m
Loss and accuracy using (cls_best): [0.7915614, tensor(0.7605)]
Processing data/mldoc/ja-1/models/sp15k/qrnn_nl4.m
ja-1*-laser-en1
Processing data/mldoc/zh-1/models/sp15k/qrnn_nl4.m
zh-1*-laser-en1
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1-laser-en1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1-laser-en1/models/sp15k/qrnn_nl4.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1-laser-en1/zh.dev.csv
Running tokenization lm...
Data lm, trn: 13500, val: 1500
Running tokenization cls...
Data cls, trn: 1000, val: 1000
Running tokenization tst...
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁的', '▁。', '▁年', '▁、', '▁在', '▁一', '▁是', '▁中', '▁有']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1-laser-en1/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1-laser-en1/models/sp15k/qrnn_nl4.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.964628    1.016147    0.710000
2         0.836267    0.885741    0.789000
3         0.694938    0.789230    0.811000
4         0.594315    0.809480    0.811000
Total time: 01:08
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1-laser-en1/models/sp15k/qrnn_nl4.m
Loss and accuracy using (cls_best): [0.5295076, tensor(0.8245)]
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-10-laser-en1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-10-laser-en1/models/sp15k/qrnn_nl4.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-10-laser-en1/zh.dev.csv
Running tokenization lm...
Data lm, trn: 13500, val: 1500
Running tokenization cls...
Data cls, trn: 10000, val: 1000
Running tokenization tst...
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁的', '▁。', '▁年', '▁、', '▁在', '▁一', '▁是', '▁中', '▁有']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-10-laser-en1/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-10-laser-en1/models/sp15k/qrnn_nl4.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.823208    0.774504    0.821000
2         0.771960    0.769799    0.822000
3         0.682731    0.724021    0.847000
4         0.595054    0.744821    0.836000
Total time: 09:42
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-10-laser-en1/models/sp15k/qrnn_nl4.m
Loss and accuracy using (cls_best): [0.59163076, tensor(0.8127)]
OrderedDict([('data/mldoc/de-1-laser-en1/models/sp15k/qrnn_nl4.m',
              0.9132500290870667),
             ('data/mldoc/de-10-laser-en1/models/sp15k/qrnn_nl4.m',
              0.9190000295639038),
             ('data/mldoc/en-1-laser-en1/models/sp15k/qrnn_nl4.m',
              0.9502500295639038),
             ('data/mldoc/en-10-laser-en1/models/sp15k/qrnn_nl4.m',
              0.9490000009536743),
             ('data/mldoc/es-1-laser-en1/models/sp15k/qrnn_nl4.m',
              0.7892500162124634),
             ('data/mldoc/es-10-laser-en1/models/sp15k/qrnn_nl4.m',
              0.7879999876022339),
             ('data/mldoc/fr-1-laser-en1/models/sp15k/qrnn_nl4.m',
              0.8945000171661377),
             ('data/mldoc/fr-10-laser-en1/models/sp15k/qrnn_nl4.m',
              0.8847500085830688),
             ('data/mldoc/it-1-laser-en1/models/sp15k/qrnn_nl4.m',
              0.7599999904632568),
             ('data/mldoc/it-10-laser-en1/models/sp15k/qrnn_nl4.m',
              0.7605000138282776),
             ('data/mldoc/zh-1-laser-en1/models/sp15k/qrnn_nl4.m',
              0.8245000243186951),
             ('data/mldoc/zh-10-laser-en1/models/sp15k/qrnn_nl4.m',
              0.812749981880188)])
data/mldoc/de-1-laser-en1/models/sp15k/qrnn_nl4.m:  0.9132500290870667
data/mldoc/de-10-laser-en1/models/sp15k/qrnn_nl4.m: 0.9190000295639038
data/mldoc/en-1-laser-en1/models/sp15k/qrnn_nl4.m:  0.9502500295639038
data/mldoc/en-10-laser-en1/models/sp15k/qrnn_nl4.m: 0.9490000009536743
data/mldoc/es-1-laser-en1/models/sp15k/qrnn_nl4.m:  0.7892500162124634
data/mldoc/es-10-laser-en1/models/sp15k/qrnn_nl4.m: 0.7879999876022339
data/mldoc/fr-1-laser-en1/models/sp15k/qrnn_nl4.m:  0.8945000171661377
data/mldoc/fr-10-laser-en1/models/sp15k/qrnn_nl4.m: 0.8847500085830688
data/mldoc/it-1-laser-en1/models/sp15k/qrnn_nl4.m:  0.7599999904632568
data/mldoc/it-10-laser-en1/models/sp15k/qrnn_nl4.m: 0.7605000138282776
data/mldoc/zh-1-laser-en1/models/sp15k/qrnn_nl4.m:  0.8245000243186951
```


#### Evaluation of Laser 1k Performance 
```
python -m ulmfit eval --glob="mldoc/*-1/models/sp60k/lstm_nl4.m" --dataset_template="{}-laser-*" --name nl4 --cuda-id=0 
Max vocab: 60000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1-laser-fr/models/sp60k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1-laser-fr/models/sp60k/lstm_nl4.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1-laser-fr/zh.dev.csv
Running tokenization...
Saving tokenized: cls.trn 13500, cls.val 1500
Running tokenization...
Saving tokenized: cls.trn 1000, cls.val 1000
Size of vocabulary: 60000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁的', '▁。', '▁年', '▁、', '▁在', '▁一', '▁中', '▁是', '▁人']
Training args:  {'tie_weights': True, 'clip': 0.12, 'bptt': 70, 'pretrained_fnames': [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp60k/lstm_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp60k/lstm_nl4.m/../itos')], 'pretrained_model': None, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  [0.25 0.1  0.2  0.02 0.15]
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1-laser-fr/models/sp60k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1-laser-fr/models/sp60k/lstm_nl4.m/info.json
Starting classifier training
epoch     train_loss  valid_loss  accuracy
1         0.789124    0.620514    0.781000
epoch     train_loss  valid_loss  accuracy
1         0.621348    0.524669    0.828000
epoch     train_loss  valid_loss  accuracy
1         0.497774    0.467979    0.842000
epoch     train_loss  valid_loss  accuracy
1         0.445851    0.479755    0.833000
2         0.424097    0.468968    0.826000
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1-laser-fr/models/sp60k/lstm_nl4.m
Loss and accuracy using (cls_best): [0.53502685, tensor(0.8235)]
[('data/mldoc/zh-1-laser-fr/models/sp60k/lstm_nl4.m', 0.8234999775886536)]
python -m ulmfit eval --glob="mldoc/*-1/models/sp60k/lstm_nl4.m" --dataset_template="{}-laser-*" --name nl4 --cuda-id=0                                   ✘ 130
Max vocab: 60000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1-laser-fr/models/sp60k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1-laser-fr/models/sp60k/lstm_nl4.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1-laser-fr/zh.dev.csv
Running tokenization...
Saving tokenized: cls.trn 13500, cls.val 1500
Running tokenization...
Saving tokenized: cls.trn 1000, cls.val 1000
Size of vocabulary: 60000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁的', '▁。', '▁年', '▁、', '▁在', '▁一', '▁中', '▁是', '▁人']
Training args:  {'tie_weights': True, 'clip': 0.12, 'bptt': 70, 'pretrained_fnames': [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp60k/lstm_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp60k/lstm_nl4.m/../itos')], 'pretrained_model': None, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  [0.25 0.1  0.2  0.02 0.15]
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1-laser-fr/models/sp60k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1-laser-fr/models/sp60k/lstm_nl4.m/info.json
Starting classifier training
epoch     train_loss  valid_loss  accuracy
1         0.789124    0.620514    0.781000
epoch     train_loss  valid_loss  accuracy
1         0.621348    0.524669    0.828000
epoch     train_loss  valid_loss  accuracy
1         0.497774    0.467979    0.842000
epoch     train_loss  valid_loss  accuracy
1         0.445851    0.479755    0.833000
2         0.424097    0.468968    0.826000
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1-laser-fr/models/sp60k/lstm_nl4.m
Loss and accuracy using (cls_best): [0.53502685, tensor(0.8235)]
[('data/mldoc/zh-1-laser-fr/models/sp60k/lstm_nl4.m', 0.8234999775886536)]
(fastaiv1) pczapla@galatea ~/w/ulmfit-multilingual ❯❯❯ python -m ulmfit eval --glob="mldoc/*-1/models/sp30k/lstm_nl4.m" --dataset_template="{}-laser-*" --name nl4 --cuda-id=0
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1-laser-de/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1-laser-de/models/sp30k/lstm_nl4.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1-laser-de/it.dev.csv
Running tokenization...
Saving tokenized: cls.trn 13500, cls.val 1500
Running tokenization...
Saving tokenized: cls.trn 1000, cls.val 1000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁.', '▁di', "▁&'", "'", '▁e', '▁il', '▁la', 'e', '▁in']
Training args:  {'tie_weights': True, 'clip': 0.12, 'bptt': 70, 'pretrained_fnames': [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1/models/sp30k/lstm_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1/models/sp30k/lstm_nl4.m/../itos')], 'pretrained_model': None, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  [0.25 0.1  0.2  0.02 0.15]
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1-laser-de/models/sp30k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1-laser-de/models/sp30k/lstm_nl4.m/info.json
Starting classifier training
epoch     train_loss  valid_loss  accuracy
1         0.823176    0.588192    0.802000
epoch     train_loss  valid_loss  accuracy
1         0.654395    0.465622    0.846000
epoch     train_loss  valid_loss  accuracy
1         0.536948    0.453061    0.847000
epoch     train_loss  valid_loss  accuracy
1         0.488410    0.454361    0.845000
2         0.450684    0.448873    0.849000
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1-laser-de/models/sp30k/lstm_nl4.m
Loss and accuracy using (cls_best): [0.6332891, tensor(0.7875)]
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1-laser-fr/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1-laser-fr/models/sp30k/lstm_nl4.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1-laser-fr/de.dev.csv
Running tokenization...
Saving tokenized: cls.trn 13500, cls.val 1500
Running tokenization...
Saving tokenized: cls.trn 1000, cls.val 1000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁.', '▁,', '▁der', '▁die', '▁und', '▁in', 'en', "▁&'", 's', '-']
Training args:  {'tie_weights': True, 'clip': 0.12, 'bptt': 70, 'pretrained_fnames': [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1/models/sp30k/lstm_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1/models/sp30k/lstm_nl4.m/../itos')], 'pretrained_model': None, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  [0.25 0.1  0.2  0.02 0.15]
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1-laser-fr/models/sp30k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1-laser-fr/models/sp30k/lstm_nl4.m/info.json
Starting classifier training
epoch     train_loss  valid_loss  accuracy
1         0.566941    0.389549    0.882000
epoch     train_loss  valid_loss  accuracy
1         0.399470    0.302616    0.898000
epoch     train_loss  valid_loss  accuracy
1         0.349054    0.336955    0.900000
epoch     train_loss  valid_loss  accuracy
1         0.278230    0.333488    0.896000
2         0.275510    0.343370    0.899000
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1-laser-fr/models/sp30k/lstm_nl4.m
Loss and accuracy using (cls_best): [0.26227093, tensor(0.9222)]
Traceback (most recent call last):
  File "/home/pczapla/anaconda3/envs/fastaiv1/lib/python3.7/runpy.py", line 193, in _run_module_as_main
    "__main__", mod_spec)
  File "/home/pczapla/anaconda3/envs/fastaiv1/lib/python3.7/runpy.py", line 85, in _run_code
    exec(code, run_globals)
  File "/home/pczapla/workspace/ulmfit-multilingual/ulmfit/__main__.py", line 58, in <module>
    fire.Fire(ULMFiT())
  File "/home/pczapla/anaconda3/envs/fastaiv1/lib/python3.7/site-packages/fire/core.py", line 127, in Fire
    component_trace = _Fire(component, args, context, name)
  File "/home/pczapla/anaconda3/envs/fastaiv1/lib/python3.7/site-packages/fire/core.py", line 366, in _Fire
    component, remaining_args)
  File "/home/pczapla/anaconda3/envs/fastaiv1/lib/python3.7/site-packages/fire/core.py", line 542, in _CallCallable
    result = fn(*varargs, **kwargs)
  File "/home/pczapla/workspace/ulmfit-multilingual/ulmfit/__main__.py", line 41, in eval
    dataset_path = get_dataset_path(base_model, dataset_template)
  File "/home/pczapla/workspace/ulmfit-multilingual/ulmfit/__main__.py", line 17, in get_dataset_path
    return list(ds.parent.glob(dataset_template.format(ds.name)))[0]
IndexError: list index out of range
(fastaiv1) pczapla@galatea ~/w/ulmfit-multilingual ❯❯❯ python -m ulmfit eval --glob="mldoc/*-1/models/sp30k/lstm_nl4.m" --dataset_template="{}-laser-*" --name nl4 --cuda-id=0                                     ✘ 1
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1-laser-de/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1-laser-de/models/sp30k/lstm_nl4.m
Evaluating previously trained model
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1-laser-de/it.dev.csv
Tokenized data loaded, lm.trn 13500, lm.val 1500
Tokenized data loaded, cls.trn 1000, cls.val 1000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁.', '▁di', "▁&'", "'", '▁e', '▁il', '▁la', 'e', '▁in']
Loss and accuracy using (cls_last): [0.6332891, tensor(0.7875)]
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1-laser-fr/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1-laser-fr/models/sp30k/lstm_nl4.m
Evaluating previously trained model
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1-laser-fr/de.dev.csv
Tokenized data loaded, lm.trn 13500, lm.val 1500
Tokenized data loaded, cls.trn 1000, cls.val 1000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁.', '▁,', '▁der', '▁die', '▁und', '▁in', 'en', "▁&'", 's', '-']
Loss and accuracy using (cls_last): [0.26227093, tensor(0.9222)]
Skipping data/mldoc/ja-1/models/sp30k/lstm_nl4.m as template {}-laser-* was not found
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1-laser-fr/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1-laser-fr/models/sp30k/lstm_nl4.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1-laser-fr/zh.dev.csv
Running tokenization...
Saving tokenized: cls.trn 13500, cls.val 1500
Running tokenization...
Saving tokenized: cls.trn 1000, cls.val 1000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁的', '▁。', '▁年', '▁、', '▁在', '▁一', '▁中', '▁人', '▁是']
Training args:  {'tie_weights': True, 'clip': 0.12, 'bptt': 70, 'pretrained_fnames': [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp30k/lstm_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp30k/lstm_nl4.m/../itos')], 'pretrained_model': None, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  [0.25 0.1  0.2  0.02 0.15]
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1-laser-fr/models/sp30k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1-laser-fr/models/sp30k/lstm_nl4.m/info.json
Starting classifier training
epoch     train_loss  valid_loss  accuracy
1         0.745636    0.601781    0.812000
epoch     train_loss  valid_loss  accuracy
1         0.564749    0.435314    0.851000
epoch     train_loss  valid_loss  accuracy
1         0.485875    0.428803    0.850000
epoch     train_loss  valid_loss  accuracy
1         0.405431    0.439304    0.847000
2         0.418333    0.442639    0.845000
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1-laser-fr/models/sp30k/lstm_nl4.m
Loss and accuracy using (cls_best): [0.5289812, tensor(0.8465)]
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1-laser-en/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1-laser-en/models/sp30k/lstm_nl4.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1-laser-en/fr.dev.csv
Running tokenization...
Saving tokenized: cls.trn 13500, cls.val 1500
Running tokenization...
Saving tokenized: cls.trn 1000, cls.val 1000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁.', "'", 's', '▁la', '▁le', '▁et', '▁l', '▁à']
Training args:  {'tie_weights': True, 'clip': 0.12, 'bptt': 70, 'pretrained_fnames': [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/models/sp30k/lstm_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/models/sp30k/lstm_nl4.m/../itos')], 'pretrained_model': None, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  [0.25 0.1  0.2  0.02 0.15]
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1-laser-en/models/sp30k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1-laser-en/models/sp30k/lstm_nl4.m/info.json
Starting classifier training
epoch     train_loss  valid_loss  accuracy
1         0.669493    0.510190    0.852000
epoch     train_loss  valid_loss  accuracy
1         0.464863    0.349456    0.888000
epoch     train_loss  valid_loss  accuracy
1         0.396977    0.335358    0.879000
epoch     train_loss  valid_loss  accuracy
1         0.316100    0.326822    0.882000
2         0.292052    0.326660    0.874000
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1-laser-en/models/sp30k/lstm_nl4.m
Loss and accuracy using (cls_best): [0.3416499, tensor(0.8878)]
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1-laser-fr/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1-laser-fr/models/sp30k/lstm_nl4.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1-laser-fr/ru.dev.csv
Running tokenization...
Saving tokenized: cls.trn 9195, cls.val 1021
Running tokenization...
Saving tokenized: cls.trn 1000, cls.val 1000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁.', '▁в', 'а', '▁и', 'е', 'и', 'й', '▁на', 'х']
Training args:  {'tie_weights': True, 'clip': 0.12, 'bptt': 70, 'pretrained_fnames': [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp30k/lstm_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp30k/lstm_nl4.m/../itos')], 'pretrained_model': None, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  [0.25 0.1  0.2  0.02 0.15]
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1-laser-fr/models/sp30k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1-laser-fr/models/sp30k/lstm_nl4.m/info.json
Starting classifier training
epoch     train_loss  valid_loss  accuracy
1         0.906124    0.592495    0.797000
epoch     train_loss  valid_loss  accuracy
1         0.751562    0.440800    0.842000
epoch     train_loss  valid_loss  accuracy
1         0.631221    0.393381    0.860000
epoch     train_loss  valid_loss  accuracy
1         0.582251    0.376320    0.867000
2         0.543821    0.374095    0.860000
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1-laser-fr/models/sp30k/lstm_nl4.m
Loss and accuracy using (cls_best): [1.0429544, tensor(0.6833)]
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1-laser-de/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1-laser-de/models/sp30k/lstm_nl4.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1-laser-de/es.dev.csv
Running tokenization...
Saving tokenized: cls.trn 13013, cls.val 1445
Running tokenization...
Saving tokenized: cls.trn 1000, cls.val 1000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁la', '▁.', '▁en', '▁el', '▁y', 's', '▁a', '▁que']
Training args:  {'tie_weights': True, 'clip': 0.12, 'bptt': 70, 'pretrained_fnames': [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp30k/lstm_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp30k/lstm_nl4.m/../itos')], 'pretrained_model': None, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  [0.25 0.1  0.2  0.02 0.15]
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1-laser-de/models/sp30k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1-laser-de/models/sp30k/lstm_nl4.m/info.json
Starting classifier training
epoch     train_loss  valid_loss  accuracy
1         0.667080    0.471187    0.884000
epoch     train_loss  valid_loss  accuracy
1         0.553853    0.329840    0.904000
epoch     train_loss  valid_loss  accuracy
1         0.463647    0.309136    0.907000
epoch     train_loss  valid_loss  accuracy
1         0.396284    0.282263    0.911000
2         0.368159    0.287222    0.916000
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1-laser-de/models/sp30k/lstm_nl4.m
Loss and accuracy using (cls_best): [0.5038375, tensor(0.8550)]
[('data/mldoc/de-1-laser-fr/models/sp30k/lstm_nl4.m', 0.922249972820282), ('data/mldoc/es-1-laser-de/models/sp30k/lstm_nl4.m', 0.8550000190734863), ('data/mldoc/fr-1-laser-en/models/sp30k/lstm_nl4.m', 0.8877500295639038), ('data/mldoc/it-1-laser-de/models/sp30k/lstm_nl4.m', 0.7875000238418579), ('data/mldoc/ru-1-laser-fr/models/sp30k/lstm_nl4.m', 0.6832500100135803), ('data/mldoc/zh-1-laser-fr/models/sp30k/lstm_nl4.m', 0.8464999794960022)]
```
second run
```
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1-laser-de/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1-laser-de/models/sp30k/lstm_nl4.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1-laser-de/de.dev.csv
Running tokenization...
Saving tokenized: cls.trn 13500, cls.val 1500
Running tokenization...
Saving tokenized: cls.trn 1000, cls.val 1000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁.', '▁,', '▁der', '▁die', '▁und', '▁in', 'en', "▁&'", 's', '-']
Training args:  {'tie_weights': True, 'clip': 0.12, 'bptt': 70, 'pretrained_fnames': [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1/models/sp30k/lstm_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1/models/sp30k/lstm_nl4.m/../itos')], 'pretrained_model': None, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  [0.25 0.1  0.2  0.02 0.15]
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1-laser-de/models/sp30k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1-laser-de/models/sp30k/lstm_nl4.m/info.json
Starting classifier training
epoch     train_loss  valid_loss  accuracy
1         0.467292    0.243158    0.919000
epoch     train_loss  valid_loss  accuracy
1         0.270090    0.207252    0.941000
epoch     train_loss  valid_loss  accuracy
1         0.201597    0.219442    0.934000
epoch     train_loss  valid_loss  accuracy
1         0.193163    0.199092    0.943000
2         0.169631    0.199501    0.940000
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1-laser-de/models/sp30k/lstm_nl4.m
Loss and accuracy using (cls_best): [0.16265252, tensor(0.9545)]
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1-laser-en/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1-laser-en/models/sp30k/lstm_nl4.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1-laser-en/de.dev.csv
Running tokenization...
Saving tokenized: cls.trn 13500, cls.val 1500
Running tokenization...
Saving tokenized: cls.trn 1000, cls.val 1000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁.', '▁,', '▁der', '▁die', '▁und', '▁in', 'en', "▁&'", 's', '-']
Training args:  {'tie_weights': True, 'clip': 0.12, 'bptt': 70, 'pretrained_fnames': [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1/models/sp30k/lstm_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1/models/sp30k/lstm_nl4.m/../itos')], 'pretrained_model': None, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  [0.25 0.1  0.2  0.02 0.15]
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1-laser-en/models/sp30k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1-laser-en/models/sp30k/lstm_nl4.m/info.json
Starting classifier training
epoch     train_loss  valid_loss  accuracy
1         0.575276    0.419917    0.879000
epoch     train_loss  valid_loss  accuracy
1         0.475003    0.263138    0.909000
epoch     train_loss  valid_loss  accuracy
1         0.345987    0.260215    0.911000
epoch     train_loss  valid_loss  accuracy
1         0.305776    0.268171    0.906000
2         0.289134    0.267642    0.911000
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1-laser-en/models/sp30k/lstm_nl4.m
Loss and accuracy using (cls_best): [0.23464507, tensor(0.9295)]
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1-laser-fr/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1-laser-fr/models/sp30k/lstm_nl4.m
Evaluating previously trained model
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1-laser-fr/de.dev.csv
Tokenized data loaded, lm.trn 13500, lm.val 1500
Tokenized data loaded, cls.trn 1000, cls.val 1000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁.', '▁,', '▁der', '▁die', '▁und', '▁in', 'en', "▁&'", 's', '-']
Loss and accuracy using (cls_last): [0.26227093, tensor(0.9222)]
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1-laser-de/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1-laser-de/models/sp30k/lstm_nl4.m
Evaluating previously trained model
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1-laser-de/es.dev.csv
Tokenized data loaded, lm.trn 13013, lm.val 1445
Tokenized data loaded, cls.trn 1000, cls.val 1000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁la', '▁.', '▁en', '▁el', '▁y', 's', '▁a', '▁que']
Loss and accuracy using (cls_last): [0.5038375, tensor(0.8550)]
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1-laser-en/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1-laser-en/models/sp30k/lstm_nl4.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1-laser-en/es.dev.csv
Running tokenization...
Saving tokenized: cls.trn 13013, cls.val 1445
Running tokenization...
Saving tokenized: cls.trn 1000, cls.val 1000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁la', '▁.', '▁en', '▁el', '▁y', 's', '▁a', '▁que']
Training args:  {'tie_weights': True, 'clip': 0.12, 'bptt': 70, 'pretrained_fnames': [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp30k/lstm_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp30k/lstm_nl4.m/../itos')], 'pretrained_model': None, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  [0.25 0.1  0.2  0.02 0.15]
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1-laser-en/models/sp30k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1-laser-en/models/sp30k/lstm_nl4.m/info.json
Starting classifier training
epoch     train_loss  valid_loss  accuracy
1         0.784271    0.667321    0.741000
epoch     train_loss  valid_loss  accuracy
1         0.601108    0.471457    0.854000
epoch     train_loss  valid_loss  accuracy
1         0.489287    0.428631    0.854000
epoch     train_loss  valid_loss  accuracy
1         0.434144    0.413409    0.864000
2         0.443724    0.385349    0.869000
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1-laser-en/models/sp30k/lstm_nl4.m
Loss and accuracy using (cls_best): [0.82167965, tensor(0.8050)]
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1-laser-fr/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1-laser-fr/models/sp30k/lstm_nl4.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1-laser-fr/es.dev.csv
Running tokenization...
Saving tokenized: cls.trn 13013, cls.val 1445
Running tokenization...
Saving tokenized: cls.trn 1000, cls.val 1000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁la', '▁.', '▁en', '▁el', '▁y', 's', '▁a', '▁que']
Training args:  {'tie_weights': True, 'clip': 0.12, 'bptt': 70, 'pretrained_fnames': [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp30k/lstm_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp30k/lstm_nl4.m/../itos')], 'pretrained_model': None, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  [0.25 0.1  0.2  0.02 0.15]
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1-laser-fr/models/sp30k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1-laser-fr/models/sp30k/lstm_nl4.m/info.json
Starting classifier training
epoch     train_loss  valid_loss  accuracy
1         0.752788    0.615142    0.786000
epoch     train_loss  valid_loss  accuracy
1         0.566108    0.403893    0.870000
epoch     train_loss  valid_loss  accuracy
1         0.503008    0.468810    0.865000
epoch     train_loss  valid_loss  accuracy
1         0.413641    0.448900    0.873000
2         0.381155    0.413034    0.879000
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1-laser-fr/models/sp30k/lstm_nl4.m
Loss and accuracy using (cls_best): [0.7937071, tensor(0.8100)]
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1-laser-de/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1-laser-de/models/sp30k/lstm_nl4.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1-laser-de/fr.dev.csv
Running tokenization...
Saving tokenized: cls.trn 13500, cls.val 1500
Running tokenization...
Saving tokenized: cls.trn 1000, cls.val 1000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁.', "'", 's', '▁la', '▁le', '▁et', '▁l', '▁à']
Training args:  {'tie_weights': True, 'clip': 0.12, 'bptt': 70, 'pretrained_fnames': [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/models/sp30k/lstm_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/models/sp30k/lstm_nl4.m/../itos')], 'pretrained_model': None, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  [0.25 0.1  0.2  0.02 0.15]
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1-laser-de/models/sp30k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1-laser-de/models/sp30k/lstm_nl4.m/info.json
Starting classifier training
epoch     train_loss  valid_loss  accuracy
1         0.674638    0.524605    0.796000
epoch     train_loss  valid_loss  accuracy
1         0.493693    0.401442    0.851000
epoch     train_loss  valid_loss  accuracy
1         0.418525    0.394886    0.859000
epoch     train_loss  valid_loss  accuracy
1         0.343561    0.402565    0.862000
2         0.335855    0.418237    0.851000
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1-laser-de/models/sp30k/lstm_nl4.m
Loss and accuracy using (cls_best): [0.44778627, tensor(0.8737)]
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1-laser-en/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1-laser-en/models/sp30k/lstm_nl4.m
Evaluating previously trained model
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1-laser-en/fr.dev.csv
Tokenized data loaded, lm.trn 13500, lm.val 1500
Tokenized data loaded, cls.trn 1000, cls.val 1000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁.', "'", 's', '▁la', '▁le', '▁et', '▁l', '▁à']
Loss and accuracy using (cls_last): [0.3416499, tensor(0.8878)]
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1-laser-fr/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1-laser-fr/models/sp30k/lstm_nl4.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1-laser-fr/fr.dev.csv
Running tokenization...
Saving tokenized: cls.trn 13500, cls.val 1500
Running tokenization...
Saving tokenized: cls.trn 1000, cls.val 1000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁.', "'", 's', '▁la', '▁le', '▁et', '▁l', '▁à']
Training args:  {'tie_weights': True, 'clip': 0.12, 'bptt': 70, 'pretrained_fnames': [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/models/sp30k/lstm_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/models/sp30k/lstm_nl4.m/../itos')], 'pretrained_model': None, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  [0.25 0.1  0.2  0.02 0.15]
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1-laser-fr/models/sp30k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1-laser-fr/models/sp30k/lstm_nl4.m/info.json
Starting classifier training
epoch     train_loss  valid_loss  accuracy
1         0.477812    0.332947    0.894000
epoch     train_loss  valid_loss  accuracy
1         0.305868    0.201659    0.937000
epoch     train_loss  valid_loss  accuracy
1         0.208116    0.224481    0.931000
epoch     train_loss  valid_loss  accuracy
1         0.146847    0.214640    0.941000
2         0.129603    0.227498    0.929000
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1-laser-fr/models/sp30k/lstm_nl4.m
Loss and accuracy using (cls_best): [0.19940722, tensor(0.9358)]
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1-laser-de/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1-laser-de/models/sp30k/lstm_nl4.m
Evaluating previously trained model
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1-laser-de/it.dev.csv
Tokenized data loaded, lm.trn 13500, lm.val 1500
Tokenized data loaded, cls.trn 1000, cls.val 1000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁.', '▁di', "▁&'", "'", '▁e', '▁il', '▁la', 'e', '▁in']
Loss and accuracy using (cls_last): [0.6332891, tensor(0.7875)]
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1-laser-en/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1-laser-en/models/sp30k/lstm_nl4.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1-laser-en/it.dev.csv
Running tokenization...
Saving tokenized: cls.trn 13500, cls.val 1500
Running tokenization...
Saving tokenized: cls.trn 1000, cls.val 1000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁.', '▁di', "▁&'", "'", '▁e', '▁il', '▁la', 'e', '▁in']
Training args:  {'tie_weights': True, 'clip': 0.12, 'bptt': 70, 'pretrained_fnames': [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1/models/sp30k/lstm_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1/models/sp30k/lstm_nl4.m/../itos')], 'pretrained_model': None, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  [0.25 0.1  0.2  0.02 0.15]
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1-laser-en/models/sp30k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1-laser-en/models/sp30k/lstm_nl4.m/info.json
Starting classifier training
epoch     train_loss  valid_loss  accuracy
1         0.845312    0.660645    0.769000
epoch     train_loss  valid_loss  accuracy
1         0.699314    0.584146    0.786000
epoch     train_loss  valid_loss  accuracy
1         0.556744    0.531658    0.801000
epoch     train_loss  valid_loss  accuracy
1         0.503091    0.529716    0.805000
2         0.474142    0.520058    0.806000
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1-laser-en/models/sp30k/lstm_nl4.m
Loss and accuracy using (cls_best): [0.7639212, tensor(0.7620)]
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1-laser-fr/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1-laser-fr/models/sp30k/lstm_nl4.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1-laser-fr/it.dev.csv
Running tokenization...
Saving tokenized: cls.trn 13500, cls.val 1500
Running tokenization...
Saving tokenized: cls.trn 1000, cls.val 1000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁.', '▁di', "▁&'", "'", '▁e', '▁il', '▁la', 'e', '▁in']
Training args:  {'tie_weights': True, 'clip': 0.12, 'bptt': 70, 'pretrained_fnames': [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1/models/sp30k/lstm_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1/models/sp30k/lstm_nl4.m/../itos')], 'pretrained_model': None, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  [0.25 0.1  0.2  0.02 0.15]
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1-laser-fr/models/sp30k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1-laser-fr/models/sp30k/lstm_nl4.m/info.json
Starting classifier training
epoch     train_loss  valid_loss  accuracy
1         0.773207    0.568426    0.803000
epoch     train_loss  valid_loss  accuracy
1         0.570457    0.516704    0.821000
epoch     train_loss  valid_loss  accuracy
1         0.527280    0.460192    0.840000
epoch     train_loss  valid_loss  accuracy
1         0.469201    0.461563    0.841000
2         0.458892    0.443310    0.836000
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1-laser-fr/models/sp30k/lstm_nl4.m
Loss and accuracy using (cls_best): [0.80693215, tensor(0.7688)]
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1-laser-de/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1-laser-de/models/sp30k/lstm_nl4.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1-laser-de/ru.dev.csv
Running tokenization...
Saving tokenized: cls.trn 9195, cls.val 1021
Running tokenization...
Saving tokenized: cls.trn 1000, cls.val 1000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁.', '▁в', 'а', '▁и', 'е', 'и', 'й', '▁на', 'х']
Training args:  {'tie_weights': True, 'clip': 0.12, 'bptt': 70, 'pretrained_fnames': [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp30k/lstm_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp30k/lstm_nl4.m/../itos')], 'pretrained_model': None, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  [0.25 0.1  0.2  0.02 0.15]
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1-laser-de/models/sp30k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1-laser-de/models/sp30k/lstm_nl4.m/info.json
Starting classifier training
epoch     train_loss  valid_loss  accuracy
1         0.878202    0.549530    0.815000
epoch     train_loss  valid_loss  accuracy
1         0.747663    0.439798    0.860000
epoch     train_loss  valid_loss  accuracy
1         0.610381    0.391122    0.878000
epoch     train_loss  valid_loss  accuracy
1         0.563902    0.393633    0.880000
2         0.515117    0.403987    0.878000
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1-laser-de/models/sp30k/lstm_nl4.m
Loss and accuracy using (cls_best): [1.3181443, tensor(0.6695)]
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1-laser-en/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1-laser-en/models/sp30k/lstm_nl4.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1-laser-en/ru.dev.csv
Running tokenization...
Saving tokenized: cls.trn 9195, cls.val 1021
Running tokenization...
Saving tokenized: cls.trn 1000, cls.val 1000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁.', '▁в', 'а', '▁и', 'е', 'и', 'й', '▁на', 'х']
Training args:  {'tie_weights': True, 'clip': 0.12, 'bptt': 70, 'pretrained_fnames': [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp30k/lstm_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp30k/lstm_nl4.m/../itos')], 'pretrained_model': None, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  [0.25 0.1  0.2  0.02 0.15]
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1-laser-en/models/sp30k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1-laser-en/models/sp30k/lstm_nl4.m/info.json
Starting classifier training
epoch     train_loss  valid_loss  accuracy
1         0.897468    0.570228    0.801000
epoch     train_loss  valid_loss  accuracy
1         0.704874    0.560132    0.812000
epoch     train_loss  valid_loss  accuracy
1         0.595008    0.507041    0.816000
epoch     train_loss  valid_loss  accuracy
1         0.484754    0.479213    0.825000
2         0.454896    0.501114    0.824000
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1-laser-en/models/sp30k/lstm_nl4.m
Loss and accuracy using (cls_best): [1.1765001, tensor(0.7005)]
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1-laser-fr/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1-laser-fr/models/sp30k/lstm_nl4.m
Evaluating previously trained model
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1-laser-fr/ru.dev.csv
Tokenized data loaded, lm.trn 9195, lm.val 1021
Tokenized data loaded, cls.trn 1000, cls.val 1000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁.', '▁в', 'а', '▁и', 'е', 'и', 'й', '▁на', 'х']
Loss and accuracy using (cls_last): [1.0429544, tensor(0.6833)]
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1-laser-de/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1-laser-de/models/sp30k/lstm_nl4.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1-laser-de/zh.dev.csv
Running tokenization...
Saving tokenized: cls.trn 13500, cls.val 1500
Running tokenization...
Saving tokenized: cls.trn 1000, cls.val 1000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁的', '▁。', '▁年', '▁、', '▁在', '▁一', '▁中', '▁人', '▁是']
Training args:  {'tie_weights': True, 'clip': 0.12, 'bptt': 70, 'pretrained_fnames': [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp30k/lstm_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp30k/lstm_nl4.m/../itos')], 'pretrained_model': None, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  [0.25 0.1  0.2  0.02 0.15]
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1-laser-de/models/sp30k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1-laser-de/models/sp30k/lstm_nl4.m/info.json
Starting classifier training
epoch     train_loss  valid_loss  accuracy
1         0.759435    0.762386    0.707000
epoch     train_loss  valid_loss  accuracy
1         0.631534    0.591862    0.786000
epoch     train_loss  valid_loss  accuracy
1         0.534237    0.589429    0.801000
epoch     train_loss  valid_loss  accuracy
1         0.454291    0.589220    0.799000
2         0.446990    0.586956    0.804000
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1-laser-de/models/sp30k/lstm_nl4.m
Loss and accuracy using (cls_best): [0.8401224, tensor(0.7232)]
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1-laser-en/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1-laser-en/models/sp30k/lstm_nl4.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1-laser-en/zh.dev.csv
Running tokenization...
Saving tokenized: cls.trn 13500, cls.val 1500
Running tokenization...
Saving tokenized: cls.trn 1000, cls.val 1000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁的', '▁。', '▁年', '▁、', '▁在', '▁一', '▁中', '▁人', '▁是']
Training args:  {'tie_weights': True, 'clip': 0.12, 'bptt': 70, 'pretrained_fnames': [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp30k/lstm_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp30k/lstm_nl4.m/../itos')], 'pretrained_model': None, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  [0.25 0.1  0.2  0.02 0.15]
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1-laser-en/models/sp30k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1-laser-en/models/sp30k/lstm_nl4.m/info.json
Starting classifier training
epoch     train_loss  valid_loss  accuracy
1         0.827517    0.821250    0.712000
epoch     train_loss  valid_loss  accuracy
1         0.636761    0.656195    0.772000
epoch     train_loss  valid_loss  accuracy
1         0.582199    0.675501    0.769000
epoch     train_loss  valid_loss  accuracy
1         0.511542    0.634232    0.764000
2         0.508244    0.647197    0.771000
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1-laser-en/models/sp30k/lstm_nl4.m
Loss and accuracy using (cls_best): [0.5421255, tensor(0.8045)]
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1-laser-fr/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1-laser-fr/models/sp30k/lstm_nl4.m
Evaluating previously trained model
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1-laser-fr/zh.dev.csv
Tokenized data loaded, lm.trn 13500, lm.val 1500
Tokenized data loaded, cls.trn 1000, cls.val 1000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁的', '▁。', '▁年', '▁、', '▁在', '▁一', '▁中', '▁人', '▁是']
Loss and accuracy using (cls_last): [0.5289812, tensor(0.8465)]
OrderedDict([('data/mldoc/de-1-laser-de/models/sp30k/lstm_nl4.m',
              0.9545000195503235),
             ('data/mldoc/de-1-laser-en/models/sp30k/lstm_nl4.m',
              0.9294999837875366),
             ('data/mldoc/de-1-laser-fr/models/sp30k/lstm_nl4.m',
              0.922249972820282),
             ('data/mldoc/es-1-laser-de/models/sp30k/lstm_nl4.m',
              0.8550000190734863),
             ('data/mldoc/es-1-laser-en/models/sp30k/lstm_nl4.m',
              0.8050000071525574),
             ('data/mldoc/es-1-laser-fr/models/sp30k/lstm_nl4.m',
              0.8100000023841858),
             ('data/mldoc/fr-1-laser-de/models/sp30k/lstm_nl4.m',
              0.8737499713897705),
             ('data/mldoc/fr-1-laser-en/models/sp30k/lstm_nl4.m',
              0.8877500295639038),
             ('data/mldoc/fr-1-laser-fr/models/sp30k/lstm_nl4.m',
              0.9357500076293945),
             ('data/mldoc/it-1-laser-de/models/sp30k/lstm_nl4.m',
              0.7875000238418579),
             ('data/mldoc/it-1-laser-en/models/sp30k/lstm_nl4.m',
              0.7620000243186951),
             ('data/mldoc/it-1-laser-fr/models/sp30k/lstm_nl4.m',
              0.768750011920929),
             ('data/mldoc/ru-1-laser-de/models/sp30k/lstm_nl4.m',
              0.6694999933242798),
             ('data/mldoc/ru-1-laser-en/models/sp30k/lstm_nl4.m',
              0.7005000114440918),
             ('data/mldoc/ru-1-laser-fr/models/sp30k/lstm_nl4.m',
              0.6832500100135803),
             ('data/mldoc/zh-1-laser-de/models/sp30k/lstm_nl4.m',
              0.7232499718666077),
             ('data/mldoc/zh-1-laser-en/models/sp30k/lstm_nl4.m',
              0.8044999837875366),
             ('data/mldoc/zh-1-laser-fr/models/sp30k/lstm_nl4.m',
              0.8464999794960022)])
```


### Laser 10k
### Building dataset 10k
```
 for SRC_LANG in en de fr; do                                                                                                                                         ✘ 130
    for LANG in en de es fr it ru zh; do
        echo $LANG from $SRC_LANG
        python ../../source/classify.py embed10000/mldoc.${SRC_LANG}-${SRC_LANG}.h5 ~/workspace/ulmfit-multilingual/data/mldoc/${LANG}-10 --suffix=10 | grep Test:
    done
done
zsh: command not found: ✘
en from en
 | Test: 92.70% | classes: 24.65 25.52 26.20 23.62
de from en
 | Test: 87.43% | classes: 20.05 25.55 28.88 25.52
es from en
 | Test: 77.38% | classes: 24.70 15.40 22.27 37.62
fr from en
 | Test: 78.70% | classes: 17.65 29.75 34.60 18.00
it from en
 | Test: 72.53% | classes: 20.75 24.60 28.52 26.12
ru from en
 | Test: 67.70% | classes: 33.40 17.12 29.35 20.12
zh from en
 | Test: 75.18% | classes: 27.70 39.00 12.07 21.23
zsh: command not found: ✘
en from de
 | Test: 81.60% | classes: 31.62 23.82 24.93 19.62
de from de
 | Test: 95.40% | classes: 24.57 26.00 25.62 23.80
es from de
 | Test: 83.50% | classes: 28.73 22.60 18.65 30.02
fr from de
 | Test: 82.85% | classes: 27.52 29.23 25.23 18.02
it from de
 | Test: 76.60% | classes: 26.52 25.62 21.43 26.43
ru from de
 | Test: 68.80% | classes: 48.48 13.22 19.43 18.88
zh from de
 | Test: 73.12% | classes: 33.12 40.10 10.53 16.25
zsh: command not found: ✘
en from fr
 | Test: 85.08% | classes: 25.43 23.43 25.43 25.73
de from fr
 | Test: 91.65% | classes: 23.15 27.10 25.00 24.75
es from fr
 | Test: 81.05% | classes: 24.18 17.68 18.93 39.23
fr from fr
 | Test: 93.65% | classes: 24.25 25.10 25.80 24.85
it from fr
 | Test: 75.08% | classes: 24.07 26.85 18.25 30.82
ru from fr
 | Test: 70.73% | classes: 43.17 19.15 16.57 21.10
zh from fr
 | Test: 76.33% | classes: 34.23 34.05 10.95 20.77
 ```
  1 10
### Building dataset 1k
```
 for SRC_LANG in en de fr; do                                                                                                                                        
    for LANG in en de es fr it ru zh; do
        echo $LANG from $SRC_LANG
        python ../../source/classify.py embed1000/mldoc.${SRC_LANG}-${SRC_LANG}.h5 ~/workspace/ulmfit-multilingual/data/mldoc/${LANG}-10 --suffix=1 | grep Test:
    done
 done
 for SRC_LANG in en de fr; do
    for LANG in en de es fr it ru zh; do
        echo $LANG from $SRC_LANG
        python ../../source/classify.py embed1000/mldoc.${SRC_LANG}-${SRC_LANG}.h5 ~/workspace/ulmfit-multilingual/data/mldoc/${LANG}-10 --suffix=1 | grep Test:
    done
 done
en from en
 | Test: 91.48% | classes: 23.77 24.90 26.25 25.07
de from en
 | Test: 87.65% | classes: 21.98 24.45 27.65 25.93
es from en
 | Test: 75.48% | classes: 21.60 15.82 22.10 40.48
fr from en
 | Test: 84.00% | classes: 23.18 29.12 27.90 19.80
it from en
 | Test: 71.18% | classes: 23.65 22.88 25.68 27.80
ru from en
 | Test: 66.58% | classes: 29.48 13.78 34.52 22.23
zh from en
 | Test: 76.65% | classes: 30.25 31.30 13.93 24.52
en from de
 | Test: 78.23% | classes: 31.80 17.73 30.15 20.32
de from de
 | Test: 93.50% | classes: 24.45 25.45 26.00 24.10
es from de
 | Test: 81.40% | classes: 24.15 25.77 20.12 29.95
fr from de
 | Test: 81.50% | classes: 25.52 29.45 27.45 17.57
it from de
 | Test: 74.53% | classes: 24.70 27.25 22.43 25.62
ru from de
 | Test: 64.58% | classes: 45.62  9.12 26.73 18.52
zh from de
 | Test: 73.20% | classes: 31.20 43.38  7.60 17.82
en from fr
 | Test: 81.30% | classes: 28.95 18.02 24.98 28.05
de from fr
 | Test: 88.75% | classes: 24.00 23.75 24.85 27.40
es from fr
 | Test: 80.12% | classes: 24.50 14.82 18.40 42.27
fr from fr
 | Test: 90.85% | classes: 24.50 24.75 24.68 26.07
it from fr
 | Test: 72.58% | classes: 25.45 24.10 17.50 32.95
ru from fr
 | Test: 67.35% | classes: 47.15 13.62 16.68 22.55
zh from fr
 | Test: 79.40% | classes: 33.60 31.12  9.07 26.20
```
### No Unfreeze
#### one epoch
```
python -m ulmfit cls --dataset-path data/mldoc/fr-1-laser  --base-lm-path data/mldoc/fr-1/models/sp30k/lstm_nl4.m  --lang=fr --name 'nl4-no_unfreeze' --cuda-id=1 - train 0 --bs 40 --num-cls-epochs=2 --unfreeze=False
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1-laser/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1-laser/models/sp30k/lstm_nl4-no_unfreeze.m
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1-laser/fr.dev.csv
Tokenized data loaded, lm.trn 13500, lm.val 1500
Tokenized data loaded, cls.trn 1000, cls.val 1000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁.', "'", 's', '▁la', '▁le', '▁et', '▁l', '▁à']
Training args:  {'tie_weights': True, 'clip': 0.12, 'bptt': 70, 'pretrained_fnames': [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/models/sp30k/lstm_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/models/sp30k/lstm_nl4.m/../itos')], 'pretrained_model': None, 'drop_mult': 0.3} dps:  [0.25 0.1  0.2  0.02 0.15]
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1-laser/models/sp30k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1-laser/models/sp30k/lstm_nl4-no_unfreeze.m/info.json
Starting classifier training
epoch     train_loss  valid_loss  accuracy
1         0.800256    0.783174    0.701000
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1-laser/models/sp30k/lstm_nl4-no_unfreeze.m
Loss and accuracy using (cls_best): [1.1735736, tensor(0.5077)]
1.173573613166809
0.5077499747276306
```
#### 4 epochs
ulmfit: 63.67%
```
python -m ulmfit cls --dataset-path data/mldoc/fr-1-laser  --base-lm-path data/mldoc/fr-1/models/sp30k/lstm_nl4.m  --lang=fr --name 'nl4-no_unfreeze2' --cuda-id=1 - train 0 --bs 40 --num-cls-epochs=2 --unfreeze=False --num-cls-frozen-epochs=4
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1-laser/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1-laser/models/sp30k/lstm_nl4-no_unfreeze2.m
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1-laser/fr.dev.csv
Tokenized data loaded, lm.trn 13500, lm.val 1500
Tokenized data loaded, cls.trn 1000, cls.val 1000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁.', "'", 's', '▁la', '▁le', '▁et', '▁l', '▁à']
Training args:  {'tie_weights': True, 'clip': 0.12, 'bptt': 70, 'pretrained_fnames': [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/models/sp30k/lstm_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/models/sp30k/lstm_nl4.m/../itos')], 'pretrained_model': None, 'drop_mult': 0.3} dps:  [0.25 0.1  0.2  0.02 0.15]
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1-laser/models/sp30k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1-laser/models/sp30k/lstm_nl4-no_unfreeze2.m/info.json
Starting classifier training
epoch     train_loss  valid_loss  accuracy
1         0.832118    0.750073    0.717000
2         0.729266    0.617375    0.749000
3         0.645946    0.623189    0.751000
4         0.566385    0.608672    0.760000
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1-laser/models/sp30k/lstm_nl4-no_unfreeze2.m
Loss and accuracy using (cls_best): [0.97152597, tensor(0.6367)]
```