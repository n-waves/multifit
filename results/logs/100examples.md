# MLDoc classification using 100 examples

| Language    | 8 epochs (1)       |  4 epochs (1)       |  4 epochs (2)   | 8 epochs (2) | 8 epochs (3)| 
|-------------|--------------------|---------------------|------------------|----------|------------|
| en          |          77.14     |        66.02        |   70.85          |  87.00   |    83.07   |
| de          |          92.37     |        91.79        |   84.60          |  91.27   |    90.90   |
| es          |          89.52     |        87.55        |   80.17          |  89.57   |    89.00   |
| fr          |          81.44     |        74.25        |   79.97          |  88.15   |    85.03   |
| it          |          81.15     |        69.24        |   74.17          |  77.54   |    80.12   |
| ja          |          78.87     |        70.30        |   69.74          |  78.64   |    80.55   |
| ru          |                     |                     |                 |          |    73.55   |
| zh          |          83.57     |        70.47        |   77.39          |  87.17   |    88.02   |


- (1) - a larger dropout value for output_p=0.7 instead of output_p=0.2, and wd=1e-1
- (2) - normal dropout but still wd=1e-1
- (3) - normal dropout but and normal wd=1e-2

## QRNN sp15k - normal dropout, normal wd
Russian
```
python -m ulmfit eval --glob="mldoc/*-1/models/sp15k/qrnn_nl4.m" --name nl4-100-e8-normal-dp-wd  --limit=100 --num-cls-epochs=8 --lr_sched=1cycle --label-smoothing-eps=0.1 --bs=18
ru-1
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp15k/qrnn_nl4-100-e8-normal-dp-wd.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1/ru.dev.csv
Limiting data set to: 100
Data lm, trn: 9195, val: 1021
Running tokenization clslimit100...
Data clslimit100, trn: 100, val: 100
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁.', '▁в', 'а', 'и', 'е', '▁и', 'й', '▁на', 'х']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
Bptt 70
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp15k/qrnn_nl4-100-e8-normal-dp-wd.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.327627    1.381715    0.230000
2         1.285086    1.290231    0.450000
3         1.131082    1.246561    0.370000
4         1.005522    1.124416    0.530000
5         0.942438    1.133643    0.540000
6         0.858336    1.063248    0.620000
7         0.802710    1.037930    0.640000
8         0.760177    1.013346    0.640000
Total time: 00:26
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp15k/qrnn_nl4-100-e8-normal-dp-wd.m
Loss and accuracy using (cls_best): [0.7943169, tensor(0.7355)]
Processing data/mldoc/zh-1/models/sp15k/qrnn_nl4.m

data/mldoc/de-1/models/sp15k/qrnn_nl4-100-e8-normal-dp-wd.m: 0.9089999794960022
data/mldoc/en-1/models/sp15k/qrnn_nl4-100-e8-normal-dp-wd.m: 0.8307499885559082
data/mldoc/es-1/models/sp15k/qrnn_nl4-100-e8-normal-dp-wd.m: 0.8889999985694885
data/mldoc/fr-1/models/sp15k/qrnn_nl4-100-e8-normal-dp-wd.m: 0.8502500057220459
data/mldoc/it-1/models/sp15k/qrnn_nl4-100-e8-normal-dp-wd.m: 0.7987499833106995
data/mldoc/ja-1/models/sp15k/qrnn_nl4-100-e8-normal-dp-wd.m: 0.8015000224113464
data/mldoc/ru-1/models/sp15k/qrnn_nl4-100-e8-normal-dp-wd.m: 0.7354999780654907
data/mldoc/zh-1/models/sp15k/qrnn_nl4-100-e8-normal-dp-wd.m: 0.8774999976158142
```
```bash
python -m ulmfit eval --glob="mldoc/*-1/models/sp15k/qrnn_nl4.m" --name nl4-100-e8-normal-dp-wd  --limit=100 --num-cls-epochs=8 --lr_sched=1cycle --label-smoothing-eps=0.1 --bs=18
Processing data/mldoc/de-1/models/sp15k/qrnn_nl4.m
de-1
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1/models/sp15k/qrnn_nl4-100-e8-normal-dp-wd.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1/de.dev.csv
Limiting data set to: 100
Data lm, trn: 13500, val: 1500
Data clslimit100, trn: 100, val: 100
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
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.299891    1.361820    0.280000
2         1.073693    1.173532    0.520000
3         0.901732    0.802865    0.850000
4         0.814596    0.769840    0.940000
5         0.751278    0.741906    0.920000
6         0.701402    0.704007    0.930000
7         0.660147    0.702021    0.930000
8         0.626870    0.683630    0.920000
Total time: 00:23
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1/models/sp15k/qrnn_nl4-100-e8-normal-dp-wd.m
Loss and accuracy using (cls_best): [0.5047863, tensor(0.9090)]
Processing data/mldoc/en-1/models/sp15k/qrnn_nl4.m
en-1
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/en-1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/en-1/models/sp15k/qrnn_nl4-100-e8-normal-dp-wd.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/en-1/en.dev.csv
Limiting data set to: 100
Data lm, trn: 13500, val: 1500
Data clslimit100, trn: 100, val: 100
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁the', '▁,', 's', '▁.', '▁of', '▁and', '▁in', '▁to', '▁a', 'ed']
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.349818    1.378012    0.340000
2         1.126246    1.169077    0.700000
3         0.975751    1.123801    0.580000
4         0.873535    0.803686    0.880000
5         0.808903    0.850648    0.860000
6         0.751683    0.858902    0.810000
7         0.714687    0.868248    0.760000
8         0.673551    0.842978    0.790000
Total time: 00:21
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/en-1/models/sp15k/qrnn_nl4-100-e8-normal-dp-wd.m
Loss and accuracy using (cls_best): [0.62346387, tensor(0.8307)]
Processing data/mldoc/es-1/models/sp15k/qrnn_nl4.m
es-1
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp15k/qrnn_nl4-100-e8-normal-dp-wd.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/es.dev.csv
Limiting data set to: 100
Data lm, trn: 13013, val: 1445
Data clslimit100, trn: 100, val: 100
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁la', '▁.', 's', '▁en', '▁el', '▁y', '▁a', '▁que']
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.301228    1.357104    0.220000
2         1.085064    1.102389    0.480000
3         0.917675    0.933821    0.700000
4         0.819897    0.867835    0.770000
5         0.754799    0.838266    0.800000
6         0.705542    0.753635    0.860000
7         0.664083    0.705510    0.900000
8         0.631440    0.695309    0.900000
Total time: 00:19
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp15k/qrnn_nl4-100-e8-normal-dp-wd.m
Loss and accuracy using (cls_best): [0.47515148, tensor(0.8900)]
Processing data/mldoc/fr-1/models/sp15k/qrnn_nl4.m
fr-1
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/models/sp15k/qrnn_nl4-100-e8-normal-dp-wd.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/fr.dev.csv
Limiting data set to: 100
Data lm, trn: 13500, val: 1500
Data clslimit100, trn: 100, val: 100
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', 's', '▁.', "'", '▁la', '▁le', '▁et', '▁l', '▁à']
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.342008    1.382562    0.230000
2         1.098667    1.072769    0.720000
3         0.954929    1.140843    0.580000
4         0.848720    0.878910    0.740000
5         0.769108    0.851176    0.770000
6         0.710496    0.773629    0.870000
7         0.666720    0.768422    0.850000
8         0.632068    0.754751    0.870000
Total time: 00:22
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/models/sp15k/qrnn_nl4-100-e8-normal-dp-wd.m
Loss and accuracy using (cls_best): [0.5851091, tensor(0.8503)]
Processing data/mldoc/it-1/models/sp15k/qrnn_nl4.m
it-1
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1/models/sp15k/qrnn_nl4-100-e8-normal-dp-wd.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1/it.dev.csv
Limiting data set to: 100
Data lm, trn: 13500, val: 1500
Data clslimit100, trn: 100, val: 100
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁.', '▁di', '▁e', "▁&'", "'", '▁il', '▁la', '▁in', 'e']
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.422726    1.367270    0.420000
2         1.209058    1.242881    0.410000
3         1.057216    1.163817    0.420000
4         0.933097    0.889416    0.800000
5         0.867725    0.970408    0.720000
6         0.796040    0.905692    0.790000
7         0.748742    0.887287    0.760000
8         0.705950    0.869270    0.790000
Total time: 00:15
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1/models/sp15k/qrnn_nl4-100-e8-normal-dp-wd.m
Loss and accuracy using (cls_best): [0.67111975, tensor(0.8012)]
Processing data/mldoc/ja-1/models/sp15k/qrnn_nl4.m
ja-1
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ja-1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ja-1/models/sp15k/qrnn_nl4-100-e8-normal-dp-wd.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ja-1/ja.dev.csv
Limiting data set to: 100
Data lm, trn: 13500, val: 1500
Data clslimit100, trn: 100, val: 100
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁、', '▁。', '▁の', '▁に', '▁を', '▁年', 'の', '▁は', '▁・', '▁)']
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.255629    1.371983    0.350000
2         1.092927    1.205145    0.630000
3         1.048904    1.064253    0.600000
4         0.943902    0.894827    0.810000
5         0.849500    0.947021    0.710000
6         0.797143    0.908895    0.730000
7         0.746244    0.853216    0.770000
8         0.705181    0.823923    0.800000
Total time: 00:24
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ja-1/models/sp15k/qrnn_nl4-100-e8-normal-dp-wd.m
Loss and accuracy using (cls_best): [0.6462945, tensor(0.8055)]
Processing data/mldoc/zh-1/models/sp15k/qrnn_nl4.m
zh-1
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp15k/qrnn_nl4-100-e8-normal-dp-wd.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/zh.dev.csv
Limiting data set to: 100
Data lm, trn: 13500, val: 1500
Data clslimit100, trn: 100, val: 100
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁的', '▁。', '▁年', '▁、', '▁在', '▁一', '▁是', '▁中', '▁有']
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.262576    1.341136    0.330000
2         1.030734    1.104332    0.800000
3         0.893319    1.021920    0.700000
4         0.791810    1.040577    0.630000
5         0.719387    0.888048    0.760000
6         0.670949    0.809457    0.880000
7         0.634845    0.786899    0.870000
8         0.606350    0.774056    0.860000
Total time: 00:19
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp15k/qrnn_nl4-100-e8-normal-dp-wd.m
Loss and accuracy using (cls_best): [0.59781086, tensor(0.8802)]
OrderedDict([('data/mldoc/de-1/models/sp15k/qrnn_nl4-100-e8-normal-dp-wd.m',
              0.9089999794960022),
             ('data/mldoc/en-1/models/sp15k/qrnn_nl4-100-e8-normal-dp-wd.m',
              0.8307499885559082),
             ('data/mldoc/es-1/models/sp15k/qrnn_nl4-100-e8-normal-dp-wd.m',
              0.8899999856948853),
             ('data/mldoc/fr-1/models/sp15k/qrnn_nl4-100-e8-normal-dp-wd.m',
              0.8502500057220459),
             ('data/mldoc/it-1/models/sp15k/qrnn_nl4-100-e8-normal-dp-wd.m',
              0.8012499809265137),
             ('data/mldoc/ja-1/models/sp15k/qrnn_nl4-100-e8-normal-dp-wd.m',
              0.8054999709129333),
             ('data/mldoc/zh-1/models/sp15k/qrnn_nl4-100-e8-normal-dp-wd.m',
              0.8802499771118164)])
data/mldoc/de-1/models/sp15k/qrnn_nl4-100-e8-normal-dp-wd.m: 0.9089999794960022
data/mldoc/en-1/models/sp15k/qrnn_nl4-100-e8-normal-dp-wd.m: 0.8307499885559082
data/mldoc/es-1/models/sp15k/qrnn_nl4-100-e8-normal-dp-wd.m: 0.8899999856948853
data/mldoc/fr-1/models/sp15k/qrnn_nl4-100-e8-normal-dp-wd.m: 0.8502500057220459
data/mldoc/it-1/models/sp15k/qrnn_nl4-100-e8-normal-dp-wd.m: 0.8012499809265137
data/mldoc/ja-1/models/sp15k/qrnn_nl4-100-e8-normal-dp-wd.m: 0.8054999709129333
data/mldoc/zh-1/models/sp15k/qrnn_nl4-100-e8-normal-dp-wd.m: 0.8802499771118164
```

## QRNN sp15k - normal dropout, wd=1e-1

### 8 epochs
```python -m ulmfit eval --glob="mldoc/*-1/models/sp15k/qrnn_nl4.m" --name nl4-100-e8-normal-dp  --limit=100 --num-cls-epochs=8 --lr_sched=1cycle --label-smoothing-eps=0.1 --bs=18
Processing data/mldoc/de-1/models/sp15k/qrnn_nl4.m
de-1
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1/models/sp15k/qrnn_nl4-100-e8-normal-dp.m
Training
Dropout {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1/de.dev.csv
Limiting data set to: 100
Data lm, trn: 13500, val: 1500
Data clslimit100, trn: 100, val: 100
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
Bptt 70
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1/models/sp15k/qrnn_nl4-100-e8-normal-dp.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.346962    1.354738    0.360000
2         1.083917    1.077360    0.700000
3         0.919886    0.868146    0.820000
4         0.820790    0.769731    0.930000
5         0.743299    0.783199    0.840000
6         0.705464    0.722734    0.910000
7         0.673815    0.693874    0.940000
8         0.640916    0.677240    0.940000
Total time: 00:22
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1/models/sp15k/qrnn_nl4-100-e8-normal-dp.m
Loss and accuracy using (cls_best): [0.49304065, tensor(0.9128)]
Processing data/mldoc/en-1/models/sp15k/qrnn_nl4.m
en-1
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/en-1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/en-1/models/sp15k/qrnn_nl4-100-e8-normal-dp.m
Training
Dropout {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/en-1/en.dev.csv
Limiting data set to: 100
Data lm, trn: 13500, val: 1500
Data clslimit100, trn: 100, val: 100
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁the', '▁,', 's', '▁.', '▁of', '▁and', '▁in', '▁to', '▁a', 'ed']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
Bptt 70
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/en-1/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/en-1/models/sp15k/qrnn_nl4-100-e8-normal-dp.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.351400    1.369022    0.290000
2         1.125228    1.163452    0.670000
3         0.966749    1.370436    0.350000
4         0.876823    0.914753    0.760000
5         0.797715    0.889629    0.760000
6         0.739767    0.833061    0.800000
7         0.699772    0.787709    0.810000
8         0.669680    0.758130    0.820000
Total time: 00:21
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/en-1/models/sp15k/qrnn_nl4-100-e8-normal-dp.m
Loss and accuracy using (cls_best): [0.55098367, tensor(0.8700)]
Processing data/mldoc/es-1/models/sp15k/qrnn_nl4.m
es-1
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp15k/qrnn_nl4-100-e8-normal-dp.m
Training
Dropout {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/es.dev.csv
Limiting data set to: 100
Data lm, trn: 13013, val: 1445
Data clslimit100, trn: 100, val: 100
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁la', '▁.', 's', '▁en', '▁el', '▁y', '▁a', '▁que']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
Bptt 70
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp15k/qrnn_nl4-100-e8-normal-dp.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.324274    1.354593    0.450000
2         1.104991    1.030298    0.810000
3         0.937743    0.761312    0.890000
4         0.846890    0.772061    0.860000
5         0.762197    0.816325    0.850000
6         0.711926    0.747758    0.890000
7         0.669704    0.714664    0.910000
8         0.635587    0.706967    0.910000
Total time: 00:19
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp15k/qrnn_nl4-100-e8-normal-dp.m
Loss and accuracy using (cls_best): [0.50185955, tensor(0.8957)]
Processing data/mldoc/fr-1/models/sp15k/qrnn_nl4.m
fr-1
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/models/sp15k/qrnn_nl4-100-e8-normal-dp.m
Training
Dropout {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/fr.dev.csv
Limiting data set to: 100
Data lm, trn: 13500, val: 1500
Data clslimit100, trn: 100, val: 100
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', 's', '▁.', "'", '▁la', '▁le', '▁et', '▁l', '▁à']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
Bptt 70
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/models/sp15k/qrnn_nl4-100-e8-normal-dp.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.311578    1.391519    0.200000
2         1.070487    1.097862    0.730000
3         0.908473    0.982642    0.700000
4         0.847152    0.954593    0.670000
5         0.773764    0.740223    0.890000
6         0.720973    0.732911    0.860000
7         0.680840    0.725618    0.880000
8         0.646478    0.711748    0.880000
Total time: 00:22
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/models/sp15k/qrnn_nl4-100-e8-normal-dp.m
Loss and accuracy using (cls_best): [0.5118136, tensor(0.8815)]
Processing data/mldoc/it-1/models/sp15k/qrnn_nl4.m
it-1
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1/models/sp15k/qrnn_nl4-100-e8-normal-dp.m
Training
Dropout {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1/it.dev.csv
Limiting data set to: 100
Data lm, trn: 13500, val: 1500
Data clslimit100, trn: 100, val: 100
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁.', '▁di', '▁e', "▁&'", "'", '▁il', '▁la', '▁in', 'e']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
Bptt 70
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1/models/sp15k/qrnn_nl4-100-e8-normal-dp.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.342177    1.374349    0.230000
2         1.180374    1.193104    0.720000
3         1.057902    0.981448    0.750000
4         0.964218    1.267683    0.390000
5         0.865437    0.927968    0.720000
6         0.801164    0.933495    0.740000
7         0.753568    0.910974    0.740000
8         0.710150    0.893166    0.750000
Total time: 00:15
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1/models/sp15k/qrnn_nl4-100-e8-normal-dp.m
Loss and accuracy using (cls_best): [0.6821853, tensor(0.7755)]
Processing data/mldoc/ja-1/models/sp15k/qrnn_nl4.m
ja-1
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ja-1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ja-1/models/sp15k/qrnn_nl4-100-e8-normal-dp.m
Training
Dropout {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ja-1/ja.dev.csv
Limiting data set to: 100
Data lm, trn: 13500, val: 1500
Data clslimit100, trn: 100, val: 100
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁、', '▁。', '▁の', '▁に', '▁を', '▁年', 'の', '▁は', '▁・', '▁)']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
Bptt 70
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ja-1/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ja-1/models/sp15k/qrnn_nl4-100-e8-normal-dp.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.390914    1.373147    0.450000
2         1.179142    1.231387    0.400000
3         0.984903    1.356539    0.410000
4         0.934303    1.067314    0.570000
5         0.862403    0.971283    0.660000
6         0.789983    0.919483    0.700000
7         0.738834    0.877783    0.770000
8         0.698692    0.849805    0.790000
Total time: 00:24
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ja-1/models/sp15k/qrnn_nl4-100-e8-normal-dp.m
Loss and accuracy using (cls_best): [0.7138088, tensor(0.7865)]
Processing data/mldoc/zh-1/models/sp15k/qrnn_nl4.m
zh-1
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp15k/qrnn_nl4-100-e8-normal-dp.m
Training
Dropout {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/zh.dev.csv
Limiting data set to: 100
Data lm, trn: 13500, val: 1500
Data clslimit100, trn: 100, val: 100
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁的', '▁。', '▁年', '▁、', '▁在', '▁一', '▁是', '▁中', '▁有']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
Bptt 70
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp15k/qrnn_nl4-100-e8-normal-dp.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.281217    1.359122    0.370000
2         1.047607    1.106353    0.720000
3         0.879789    0.842300    0.790000
4         0.796210    0.830705    0.810000
5         0.733291    0.782483    0.830000
6         0.681939    0.757719    0.870000
7         0.642002    0.751473    0.880000
8         0.611668    0.735970    0.860000
Total time: 00:20
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp15k/qrnn_nl4-100-e8-normal-dp.m
Loss and accuracy using (cls_best): [0.56022245, tensor(0.8717)]
OrderedDict([('data/mldoc/de-1/models/sp15k/qrnn_nl4-100-e8-normal-dp.m',
              0.9127500057220459),
             ('data/mldoc/en-1/models/sp15k/qrnn_nl4-100-e8-normal-dp.m',
              0.8700000047683716),
             ('data/mldoc/es-1/models/sp15k/qrnn_nl4-100-e8-normal-dp.m',
              0.8957499861717224),
             ('data/mldoc/fr-1/models/sp15k/qrnn_nl4-100-e8-normal-dp.m',
              0.8815000057220459),
             ('data/mldoc/it-1/models/sp15k/qrnn_nl4-100-e8-normal-dp.m',
              0.7754999995231628),
             ('data/mldoc/ja-1/models/sp15k/qrnn_nl4-100-e8-normal-dp.m',
              0.7864999771118164),
             ('data/mldoc/zh-1/models/sp15k/qrnn_nl4-100-e8-normal-dp.m',
              0.871749997138977)])
data/mldoc/de-1/models/sp15k/qrnn_nl4-100-e8-normal-dp.m: 0.9127500057220459
data/mldoc/en-1/models/sp15k/qrnn_nl4-100-e8-normal-dp.m: 0.8700000047683716
data/mldoc/es-1/models/sp15k/qrnn_nl4-100-e8-normal-dp.m: 0.8957499861717224
data/mldoc/fr-1/models/sp15k/qrnn_nl4-100-e8-normal-dp.m: 0.8815000057220459
data/mldoc/it-1/models/sp15k/qrnn_nl4-100-e8-normal-dp.m: 0.7754999995231628
data/mldoc/ja-1/models/sp15k/qrnn_nl4-100-e8-normal-dp.m: 0.7864999771118164
data/mldoc/zh-1/models/sp15k/qrnn_nl4-100-e8-normal-dp.m: 0.871749997138977
```

 
### 4 epochs
```
python -m ulmfit eval --glob="mldoc/*-1/models/sp15k/qrnn_nl4.m" --name nl4-100-e4-normal-dp  --limit=100 --num-cls-epochs=4 --lr_sched=1cycle --label-smoothing-eps=0.1 --bs=18
Processing data/mldoc/de-1/models/sp15k/qrnn_nl4.m
de-1
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1/models/sp15k/qrnn_nl4-100-e4-normal-dp.m
Training
Dropout {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1/de.dev.csv
Limiting data set to: 100
Data lm, trn: 13500, val: 1500
Data clslimit100, trn: 100, val: 100
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
Bptt 70
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1/models/sp15k/qrnn_nl4-100-e4-normal-dp.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.284350    1.332569    0.280000
2         1.016471    0.983958    0.840000
3         0.867660    0.927123    0.910000
4         0.761539    0.911453    0.840000
Total time: 00:11
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1/models/sp15k/qrnn_nl4-100-e4-normal-dp.m
Loss and accuracy using (cls_best): [0.819043, tensor(0.8460)]
Processing data/mldoc/en-1/models/sp15k/qrnn_nl4.m
en-1
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/en-1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/en-1/models/sp15k/qrnn_nl4-100-e4-normal-dp.m
Training
Dropout {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/en-1/en.dev.csv
Limiting data set to: 100
Data lm, trn: 13500, val: 1500
Data clslimit100, trn: 100, val: 100
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁the', '▁,', 's', '▁.', '▁of', '▁and', '▁in', '▁to', '▁a', 'ed']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
Bptt 70
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/en-1/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/en-1/models/sp15k/qrnn_nl4-100-e4-normal-dp.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.290013    1.338139    0.370000
2         1.052497    1.222131    0.380000
3         0.904921    1.098593    0.440000
4         0.809407    0.984065    0.680000
Total time: 00:10
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/en-1/models/sp15k/qrnn_nl4-100-e4-normal-dp.m
Loss and accuracy using (cls_best): [0.89315945, tensor(0.7085)]
Processing data/mldoc/es-1/models/sp15k/qrnn_nl4.m
es-1
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp15k/qrnn_nl4-100-e4-normal-dp.m
Training
Dropout {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/es.dev.csv
Limiting data set to: 100
Data lm, trn: 13013, val: 1445
Data clslimit100, trn: 100, val: 100
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁la', '▁.', 's', '▁en', '▁el', '▁y', '▁a', '▁que']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
Bptt 70
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp15k/qrnn_nl4-100-e4-normal-dp.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.286507    1.319209    0.390000
2         1.063101    1.230659    0.450000
3         0.925757    1.026052    0.600000
4         0.812203    0.907415    0.800000
Total time: 00:09
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp15k/qrnn_nl4-100-e4-normal-dp.m
Loss and accuracy using (cls_best): [0.7879555, tensor(0.8018)]
Processing data/mldoc/fr-1/models/sp15k/qrnn_nl4.m
fr-1
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/models/sp15k/qrnn_nl4-100-e4-normal-dp.m
Training
Dropout {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/fr.dev.csv
Limiting data set to: 100
Data lm, trn: 13500, val: 1500
Data clslimit100, trn: 100, val: 100
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', 's', '▁.', "'", '▁la', '▁le', '▁et', '▁l', '▁à']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
Bptt 70
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/models/sp15k/qrnn_nl4-100-e4-normal-dp.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.267655    1.295306    0.460000
2         1.034598    1.207809    0.360000
3         0.869741    0.965482    0.690000
4         0.772019    0.871552    0.810000
Total time: 00:10
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/models/sp15k/qrnn_nl4-100-e4-normal-dp.m
Loss and accuracy using (cls_best): [0.7594081, tensor(0.7997)]
Processing data/mldoc/it-1/models/sp15k/qrnn_nl4.m
it-1
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1/models/sp15k/qrnn_nl4-100-e4-normal-dp.m
Training
Dropout {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1/it.dev.csv
Limiting data set to: 100
Data lm, trn: 13500, val: 1500
Data clslimit100, trn: 100, val: 100
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁.', '▁di', '▁e', "▁&'", "'", '▁il', '▁la', '▁in', 'e']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
Bptt 70
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1/models/sp15k/qrnn_nl4-100-e4-normal-dp.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.338249    1.345434    0.320000
2         1.179869    1.195013    0.480000
3         1.027482    1.109100    0.560000
4         0.900706    1.021052    0.730000
Total time: 00:07
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1/models/sp15k/qrnn_nl4-100-e4-normal-dp.m
Loss and accuracy using (cls_best): [0.8819375, tensor(0.7418)]
Processing data/mldoc/ja-1/models/sp15k/qrnn_nl4.m
ja-1
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ja-1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ja-1/models/sp15k/qrnn_nl4-100-e4-normal-dp.m
Training
Dropout {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ja-1/ja.dev.csv
Limiting data set to: 100
Data lm, trn: 13500, val: 1500
Data clslimit100, trn: 100, val: 100
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁、', '▁。', '▁の', '▁に', '▁を', '▁年', 'の', '▁は', '▁・', '▁)']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
Bptt 70
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ja-1/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ja-1/models/sp15k/qrnn_nl4-100-e4-normal-dp.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.329939    1.365095    0.200000
2         1.109706    1.175099    0.420000
3         0.978721    1.141211    0.440000
4         0.870699    1.028071    0.700000
Total time: 00:12
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ja-1/models/sp15k/qrnn_nl4-100-e4-normal-dp.m
Loss and accuracy using (cls_best): [0.9101604, tensor(0.6975)]
Processing data/mldoc/zh-1/models/sp15k/qrnn_nl4.m
zh-1
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp15k/qrnn_nl4-100-e4-normal-dp.m
Training
Dropout {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/zh.dev.csv
Limiting data set to: 100
Data lm, trn: 13500, val: 1500
Data clslimit100, trn: 100, val: 100
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁的', '▁。', '▁年', '▁、', '▁在', '▁一', '▁是', '▁中', '▁有']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
Bptt 70
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp15k/qrnn_nl4-100-e4-normal-dp.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.256988    1.290418    0.370000
2         1.000337    0.967188    0.780000
3         0.837002    0.955509    0.760000
4         0.740278    0.937393    0.780000
Total time: 00:10
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp15k/qrnn_nl4-100-e4-normal-dp.m
Loss and accuracy using (cls_best): [0.8290863, tensor(0.7740)]
OrderedDict([('data/mldoc/de-1/models/sp15k/qrnn_nl4-100-e4-normal-dp.m',
              0.8460000157356262),
             ('data/mldoc/en-1/models/sp15k/qrnn_nl4-100-e4-normal-dp.m',
              0.7085000276565552),
             ('data/mldoc/es-1/models/sp15k/qrnn_nl4-100-e4-normal-dp.m',
              0.8017500042915344),
             ('data/mldoc/fr-1/models/sp15k/qrnn_nl4-100-e4-normal-dp.m',
              0.7997499704360962),
             ('data/mldoc/it-1/models/sp15k/qrnn_nl4-100-e4-normal-dp.m',
              0.7417500019073486),
             ('data/mldoc/ja-1/models/sp15k/qrnn_nl4-100-e4-normal-dp.m',
              0.6974999904632568),
             ('data/mldoc/zh-1/models/sp15k/qrnn_nl4-100-e4-normal-dp.m',
              0.7739999890327454)])
data/mldoc/de-1/models/sp15k/qrnn_nl4-100-e4-normal-dp.m: 0.8460000157356262
data/mldoc/en-1/models/sp15k/qrnn_nl4-100-e4-normal-dp.m: 0.7085000276565552
data/mldoc/es-1/models/sp15k/qrnn_nl4-100-e4-normal-dp.m: 0.8017500042915344
data/mldoc/fr-1/models/sp15k/qrnn_nl4-100-e4-normal-dp.m: 0.7997499704360962
data/mldoc/it-1/models/sp15k/qrnn_nl4-100-e4-normal-dp.m: 0.7417500019073486
data/mldoc/ja-1/models/sp15k/qrnn_nl4-100-e4-normal-dp.m: 0.6974999904632568
data/mldoc/zh-1/models/sp15k/qrnn_nl4-100-e4-normal-dp.m: 0.7739999890327454
```


## QRNN sp15k - larger dropout
The following dropouts were used: `output_p=0.7` compared to other experimejnts where `output_p=0.2`

| Model                       | Accuracy 8 epochs  | Accuracy 4 epochs
|-----------------------------|--------------------|---------------------|
| de-1 sp15k/qrnn_nl4-100e8.m |          92.37     |        91.79        |
| en-1 sp15k/qrnn_nl4-100e8.m |          77.14     |        66.02        |
| es-1 sp15k/qrnn_nl4-100e8.m |          89.52     |        87.55        |
| fr-1 sp15k/qrnn_nl4-100e8.m |          81.44     |        74.25        |
| it-1 sp15k/qrnn_nl4-100e8.m |          81.15     |        69.24        |
| ja-1 sp15k/qrnn_nl4-100e8.m |          78.87     |        70.30        |
| zh-1 sp15k/qrnn_nl4-100e8.m |          83.57     |        70.47        | 

### 8 epochs of training 
`self.dps = dict(output_p=0.7, hidden_p=0.1, input_p=0.2, embed_p=0.02, weight_p=0.15)`
````
python -m ulmfit eval --glob="mldoc/*-1/models/sp15k/qrnn_nl4.m" --name nl4-100e8  --limit=100 --num-cls-epochs=8 --lr_sched=1cycle --label-smoothing-eps=0.1 --bs=18
Processing data/mldoc/de-1/models/sp15k/qrnn_nl4.m
de-1
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1/models/sp15k/qrnn_nl4-100e8.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1/de.dev.csv
Limiting data set to: 100
Data lm, trn: 13500, val: 1500
Data clslimit100, trn: 100, val: 100
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
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.354484    1.379077    0.280000
2         1.152421    1.152570    0.620000
3         0.980928    1.018830    0.660000
4         0.871912    0.872564    0.820000
5         0.803019    0.789034    0.900000
6         0.748509    0.711389    0.940000
7         0.701289    0.690140    0.950000
8         0.664756    0.669448    0.950000
Total time: 00:23
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1/models/sp15k/qrnn_nl4-100e8.m
Loss and accuracy using (cls_best): [0.48774713, tensor(0.9237)]
Processing data/mldoc/en-1/models/sp15k/qrnn_nl4.m
en-1
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/en-1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/en-1/models/sp15k/qrnn_nl4-100e8.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/en-1/en.dev.csv
Limiting data set to: 100
Data lm, trn: 13500, val: 1500
Data clslimit100, trn: 100, val: 100
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁the', '▁,', 's', '▁.', '▁of', '▁and', '▁in', '▁to', '▁a', 'ed']
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.366923    1.380464    0.310000
2         1.177802    1.236425    0.500000
3         0.999587    0.912387    0.740000
4         0.898221    0.990741    0.660000
5         0.818071    1.066497    0.580000
6         0.757039    0.921970    0.680000
7         0.712545    0.956183    0.660000
8         0.681197    0.911881    0.710000
Total time: 00:22
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/en-1/models/sp15k/qrnn_nl4-100e8.m
Loss and accuracy using (cls_best): [0.66596776, tensor(0.7715)]
Processing data/mldoc/es-1/models/sp15k/qrnn_nl4.m
es-1
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp15k/qrnn_nl4-100e8.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/es.dev.csv
Limiting data set to: 100
Data lm, trn: 13013, val: 1445
Data clslimit100, trn: 100, val: 100
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁la', '▁.', 's', '▁en', '▁el', '▁y', '▁a', '▁que']
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.253322    1.366251    0.220000
2         1.051253    1.117081    0.650000
3         0.905148    0.928994    0.770000
4         0.825990    0.855558    0.810000
5         0.754039    0.828022    0.810000
6         0.704161    0.752015    0.880000
7         0.661291    0.739049    0.880000
8         0.631924    0.731397    0.880000
Total time: 00:19
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp15k/qrnn_nl4-100e8.m
Loss and accuracy using (cls_best): [0.5359205, tensor(0.8953)]
Processing data/mldoc/fr-1/models/sp15k/qrnn_nl4.m
fr-1
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/models/sp15k/qrnn_nl4-100e8.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/fr.dev.csv
Limiting data set to: 100
Data lm, trn: 13500, val: 1500
Data clslimit100, trn: 100, val: 100
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', 's', '▁.', "'", '▁la', '▁le', '▁et', '▁l', '▁à']
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.315270    1.368941    0.220000
2         1.117436    1.137058    0.630000
3         0.959049    0.955980    0.750000
4         0.856808    0.749577    0.900000
5         0.800600    0.859013    0.740000
6         0.739985    0.873451    0.720000
7         0.696956    0.820335    0.780000
8         0.660949    0.784974    0.820000
Total time: 00:22
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/models/sp15k/qrnn_nl4-100e8.m
Loss and accuracy using (cls_best): [0.6222675, tensor(0.8145)]
Processing data/mldoc/it-1/models/sp15k/qrnn_nl4.m
it-1
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1/models/sp15k/qrnn_nl4-100e8.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1/it.dev.csv
Limiting data set to: 100
Data lm, trn: 13500, val: 1500
Data clslimit100, trn: 100, val: 100
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁.', '▁di', '▁e', "▁&'", "'", '▁il', '▁la', '▁in', 'e']
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.406870    1.370643    0.250000
2         1.248155    1.226949    0.620000
3         1.098265    1.045475    0.650000
4         0.989543    0.999063    0.660000
5         0.904690    0.908897    0.750000
6         0.838460    0.906191    0.770000
7         0.790075    0.867051    0.810000
8         0.748298    0.844697    0.820000
Total time: 00:16
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1/models/sp15k/qrnn_nl4-100e8.m
Loss and accuracy using (cls_best): [0.6499791, tensor(0.8115)]
Processing data/mldoc/ja-1/models/sp15k/qrnn_nl4.m
ja-1
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ja-1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ja-1/models/sp15k/qrnn_nl4-100e8.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ja-1/ja.dev.csv
Limiting data set to: 100
Data lm, trn: 13500, val: 1500
Data clslimit100, trn: 100, val: 100
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁、', '▁。', '▁の', '▁に', '▁を', '▁年', 'の', '▁は', '▁・', '▁)']
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.284972    1.374997    0.390000
2         1.112592    1.235691    0.490000
3         0.990320    1.235275    0.470000
4         0.884597    0.985890    0.680000
5         0.814330    1.017439    0.660000
6         0.756775    0.912423    0.780000
7         0.757921    0.860843    0.840000
8         0.731727    0.835498    0.810000
Total time: 00:24
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ja-1/models/sp15k/qrnn_nl4-100e8.m
Loss and accuracy using (cls_best): [0.68836695, tensor(0.7887)]
Processing data/mldoc/zh-1/models/sp15k/qrnn_nl4.m
zh-1
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp15k/qrnn_nl4-100e8.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/zh.dev.csv
Limiting data set to: 100
Data lm, trn: 13500, val: 1500
Data clslimit100, trn: 100, val: 100
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁的', '▁。', '▁年', '▁、', '▁在', '▁一', '▁是', '▁中', '▁有']
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.282265    1.340392    0.670000
2         1.029523    1.092677    0.600000
3         0.890638    1.027698    0.650000
4         0.799896    0.919632    0.730000
5         0.735252    0.840827    0.800000
6         0.684259    0.791598    0.860000
7         0.645315    0.794246    0.840000
8         0.615542    0.782311    0.840000
Total time: 00:20
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp15k/qrnn_nl4-100e8.m
Loss and accuracy using (cls_best): [0.63012207, tensor(0.8357)]
OrderedDict([('data/mldoc/de-1/models/sp15k/qrnn_nl4-100e8.m',
              0.9237499833106995),
             ('data/mldoc/en-1/models/sp15k/qrnn_nl4-100e8.m',
              0.7714999914169312),
             ('data/mldoc/es-1/models/sp15k/qrnn_nl4-100e8.m',
              0.8952500224113464),
             ('data/mldoc/fr-1/models/sp15k/qrnn_nl4-100e8.m',
              0.8144999742507935),
             ('data/mldoc/it-1/models/sp15k/qrnn_nl4-100e8.m',
              0.8115000128746033),
             ('data/mldoc/ja-1/models/sp15k/qrnn_nl4-100e8.m',
              0.7887499928474426),
             ('data/mldoc/zh-1/models/sp15k/qrnn_nl4-100e8.m',
              0.8357499837875366)])
data/mldoc/de-1/models/sp15k/qrnn_nl4-100e8.m: 0.9237499833106995
data/mldoc/en-1/models/sp15k/qrnn_nl4-100e8.m: 0.7714999914169312
data/mldoc/es-1/models/sp15k/qrnn_nl4-100e8.m: 0.8952500224113464
data/mldoc/fr-1/models/sp15k/qrnn_nl4-100e8.m: 0.8144999742507935
data/mldoc/it-1/models/sp15k/qrnn_nl4-100e8.m: 0.8115000128746033
data/mldoc/ja-1/models/sp15k/qrnn_nl4-100e8.m: 0.7887499928474426
data/mldoc/zh-1/models/sp15k/qrnn_nl4-100e8.m: 0.8357499837875366
```` 

### 4 epochs of training

````
python -m ulmfit eval --glob="mldoc/*-1/models/sp15k/qrnn_nl4.m" --name nl4-100e4  --limit=100 --num-cls-epochs=4 --lr_sched=1cycle --label-smoothing-eps=0.1 --bs=18
Processing data/mldoc/de-1/models/sp15k/qrnn_nl4.m
de-1
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1/models/sp15k/qrnn_nl4-100e4.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1/de.dev.csv
Limiting data set to: 100
Data lm, trn: 13500, val: 1500
Data clslimit100, trn: 100, val: 100
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁.', '▁,', '▁der', '▁die', 'en', '▁und', 's', '▁in', 'er', "▁&'"]
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.7, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
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
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1/models/sp15k/qrnn_nl4-100e4.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.268574    1.327856    0.330000
2         1.006592    1.120209    0.470000
3         0.865163    0.915048    0.860000
4         0.775797    0.878599    0.910000
Total time: 00:11
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1/models/sp15k/qrnn_nl4-100e4.m
Loss and accuracy using (cls_best): [0.7670934, tensor(0.9180)]
Processing data/mldoc/en-1/models/sp15k/qrnn_nl4.m
en-1
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/en-1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/en-1/models/sp15k/qrnn_nl4-100e4.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/en-1/en.dev.csv
Limiting data set to: 100
Data lm, trn: 13500, val: 1500
Data clslimit100, trn: 100, val: 100
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁the', '▁,', 's', '▁.', '▁of', '▁and', '▁in', '▁to', '▁a', 'ed']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.7, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
Bptt 70
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/en-1/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/en-1/models/sp15k/qrnn_nl4-100e4.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.346742    1.344590    0.400000
2         1.097295    1.131111    0.570000
3         0.927231    0.992626    0.650000
4         0.823589    1.027419    0.600000
Total time: 00:10
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/en-1/models/sp15k/qrnn_nl4-100e4.m
Loss and accuracy using (cls_best): [0.8792457, tensor(0.6603)]
Processing data/mldoc/es-1/models/sp15k/qrnn_nl4.m
es-1
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp15k/qrnn_nl4-100e4.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/es.dev.csv
Limiting data set to: 100
Data lm, trn: 13013, val: 1445
Data clslimit100, trn: 100, val: 100
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁la', '▁.', 's', '▁en', '▁el', '▁y', '▁a', '▁que']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.7, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
Bptt 70
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp15k/qrnn_nl4-100e4.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.249590    1.328701    0.370000
2         0.977393    1.031057    0.540000
3         0.858498    0.886364    0.850000
4         0.777858    0.868647    0.890000
Total time: 00:10
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp15k/qrnn_nl4-100e4.m
Loss and accuracy using (cls_best): [0.72082895, tensor(0.8755)]
Processing data/mldoc/fr-1/models/sp15k/qrnn_nl4.m
fr-1
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/models/sp15k/qrnn_nl4-100e4.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/fr.dev.csv
Limiting data set to: 100
Data lm, trn: 13500, val: 1500
Data clslimit100, trn: 100, val: 100
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', 's', '▁.', "'", '▁la', '▁le', '▁et', '▁l', '▁à']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.7, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
Bptt 70
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/models/sp15k/qrnn_nl4-100e4.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.305505    1.362959    0.200000
2         1.073055    1.068938    0.700000
3         0.892900    0.981664    0.620000
4         0.796118    0.950983    0.700000
Total time: 00:11
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/models/sp15k/qrnn_nl4-100e4.m
Loss and accuracy using (cls_best): [0.8226887, tensor(0.7425)]
Processing data/mldoc/it-1/models/sp15k/qrnn_nl4.m
it-1
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1/models/sp15k/qrnn_nl4-100e4.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1/it.dev.csv
Limiting data set to: 100
Data lm, trn: 13500, val: 1500
Data clslimit100, trn: 100, val: 100
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁.', '▁di', '▁e', "▁&'", "'", '▁il', '▁la', '▁in', 'e']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.7, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
Bptt 70
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1/models/sp15k/qrnn_nl4-100e4.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.372190    1.350273    0.500000
2         1.207576    1.173455    0.750000
3         1.040837    1.107950    0.590000
4         0.931903    1.119772    0.540000
Total time: 00:07
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1/models/sp15k/qrnn_nl4-100e4.m
Loss and accuracy using (cls_best): [0.95755553, tensor(0.6925)]
Processing data/mldoc/ja-1/models/sp15k/qrnn_nl4.m
ja-1
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ja-1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ja-1/models/sp15k/qrnn_nl4-100e4.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ja-1/ja.dev.csv
Limiting data set to: 100
Data lm, trn: 13500, val: 1500
Data clslimit100, trn: 100, val: 100
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁、', '▁。', '▁の', '▁に', '▁を', '▁年', 'の', '▁は', '▁・', '▁)']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.7, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
Bptt 70
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ja-1/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ja-1/models/sp15k/qrnn_nl4-100e4.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.300252    1.326448    0.330000
2         1.175279    1.211864    0.500000
3         1.019274    1.129269    0.460000
4         0.895053    1.059557    0.640000
Total time: 00:11
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ja-1/models/sp15k/qrnn_nl4-100e4.m
Loss and accuracy using (cls_best): [0.96151954, tensor(0.7030)]
Processing data/mldoc/zh-1/models/sp15k/qrnn_nl4.m
zh-1
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp15k/qrnn_nl4-100e4.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/zh.dev.csv
Limiting data set to: 100
Data lm, trn: 13500, val: 1500
Data clslimit100, trn: 100, val: 100
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁的', '▁。', '▁年', '▁、', '▁在', '▁一', '▁是', '▁中', '▁有']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.7, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
Bptt 70
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp15k/qrnn_nl4-100e4.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.230801    1.302317    0.660000
2         0.944098    0.992352    0.710000
3         0.810783    1.026886    0.630000
4         0.736866    0.985942    0.680000
Total time: 00:09
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp15k/qrnn_nl4-100e4.m
Loss and accuracy using (cls_best): [0.8833954, tensor(0.7048)]
OrderedDict([('data/mldoc/de-1/models/sp15k/qrnn_nl4-100e4.m',
              0.9179999828338623),
             ('data/mldoc/en-1/models/sp15k/qrnn_nl4-100e4.m',
              0.6602500081062317),
             ('data/mldoc/es-1/models/sp15k/qrnn_nl4-100e4.m',
              0.8755000233650208),
             ('data/mldoc/fr-1/models/sp15k/qrnn_nl4-100e4.m',
              0.7425000071525574),
             ('data/mldoc/it-1/models/sp15k/qrnn_nl4-100e4.m',
              0.6924999952316284),
             ('data/mldoc/ja-1/models/sp15k/qrnn_nl4-100e4.m',
              0.703000009059906),
             ('data/mldoc/zh-1/models/sp15k/qrnn_nl4-100e4.m',
              0.7047500014305115)])
data/mldoc/de-1/models/sp15k/qrnn_nl4-100e4.m: 0.9179999828338623
data/mldoc/en-1/models/sp15k/qrnn_nl4-100e4.m: 0.6602500081062317
data/mldoc/es-1/models/sp15k/qrnn_nl4-100e4.m: 0.8755000233650208
data/mldoc/fr-1/models/sp15k/qrnn_nl4-100e4.m: 0.7425000071525574
data/mldoc/it-1/models/sp15k/qrnn_nl4-100e4.m: 0.6924999952316284
data/mldoc/ja-1/models/sp15k/qrnn_nl4-100e4.m: 0.703000009059906
data/mldoc/zh-1/models/sp15k/qrnn_nl4-100e4.m: 0.7047500014305115
````

