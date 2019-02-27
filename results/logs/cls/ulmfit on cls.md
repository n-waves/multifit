# CLS execution logs

| model name                                 | Accuracy |
|--------------------------------------------|----------|
| data/cls/de-books/models/sp15k/qrnn_nl4.m: |   93.19  |
| data/cls/de-dvd/models/sp15k/qrnn_nl4.m:   |   90.54  |
| data/cls/de-music/models/sp15k/qrnn_nl4.m: |   93.00  | 
| data/cls/en-books/models/sp15k/qrnn_nl4.m: |   90.75  |
| data/cls/en-dvd/models/sp15k/qrnn_nl4.m:   |   89.30  |
| data/cls/en-music/models/sp15k/qrnn_nl4.m: |   89.45  |
| data/cls/fr-books/models/sp15k/qrnn_nl4.m: |   91.25  |
| data/cls/fr-dvd/models/sp15k/qrnn_nl4.m:   |   89.55  |
| data/cls/fr-music/models/sp15k/qrnn_nl4.m: |   93.40  |
| data/cls/ja-books/models/sp15k/qrnn_nl4.m: |   86.29  |
| data/cls/ja-dvd/models/sp15k/qrnn_nl4.m:   |   85.75  |
| data/cls/ja-music/models/sp15k/qrnn_nl4.m: |   86.59  |

### All langs books

```
python -m ulmfit eval --glob="wiki/*-100/models/sp15k/qrnn_nl4.m" --name nl4 --dataset-template='../cls/${lang}-books' --num-lm-epochs=20  --num-cls-epochs=8  --bs=18 --lr_sched=1cycle --label-smoothing-eps=0.1
Processing data/wiki/de-100/models/sp15k/qrnn_nl4.m
../cls/de-books
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/cls/de-books/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/cls/de-books/models/sp15k/qrnn_nl4.m
Training
Validation set not found using 10% of trn
Error You have NaN values in column(s) of your dataframe, please fix it.
Processing data/wiki/en-100/models/sp15k/qrnn_nl4.m
../cls/en-books
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/cls/en-books/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/cls/en-books/models/sp15k/qrnn_nl4.m
Training
Validation set not found using 10% of trn
Running tokenization lm...
Data lm, trn: 48600, val: 5400
Running tokenization cls...
Data cls, trn: 1800, val: 200
Running tokenization tst...
Data tst, trn: 200, val: 2000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁the', '▁,', 's', '▁.', '▁of', '▁and', '▁in', '▁to', '▁a', 'ed']
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
Training lm from:  [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/wiki/en-100/models/sp15k/qrnn_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/wiki/en-100/models/sp15k/qrnn_nl4.m/../itos')]
epoch     train_loss  valid_loss  accuracy
1         5.411075    4.929663    0.299280
Total time: 05:18
epoch     train_loss  valid_loss  accuracy
1         4.911766    4.729969    0.328842
2         4.749516    4.571162    0.352025
3         4.574124    4.432505    0.369642
4         4.496035    4.315759    0.384135
5         4.402706    4.238598    0.393190
6         4.336373    4.176495    0.401667
7         4.305870    4.111050    0.410795
8         4.251368    4.055121    0.419378
9         4.215136    4.005383    0.426659
10        4.242251    3.959000    0.434112
11        4.128317    3.919106    0.440762
12        4.102277    3.881066    0.447852
13        4.109466    3.840265    0.455235
14        4.074686    3.798793    0.463265
15        4.034000    3.765705    0.469861
16        3.984559    3.736501    0.475469
17        3.961936    3.718403    0.479141
18        3.986044    3.708190    0.481330
19        3.934352    3.700921    0.482685
20        3.977421    3.698027    0.483279
Total time: 2:33:09
/home/pczapla/workspace/ulmfit-multilingual/data/cls/en-books/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/cls/en-books/models/sp15k/qrnn_nl4.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.511342    0.485253    0.890000
2         0.476206    0.497634    0.875000
3         0.423625    0.495749    0.880000
4         0.383525    0.472684    0.890000
5         0.371009    0.455094    0.895000
6         0.359045    0.453902    0.905000
7         0.354646    0.453879    0.910000
8         0.348541    0.462702    0.885000
Total time: 02:30
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/cls/en-books/models/sp15k/qrnn_nl4.m
Loss and accuracy using (cls_best): [0.27204505, tensor(0.9060)]
Processing data/wiki/es-100/models/sp15k/qrnn_nl4.m
../cls/es-books
Processing data/wiki/fr-100/models/sp15k/qrnn_nl4.m
../cls/fr-books
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/cls/fr-books/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/cls/fr-books/models/sp15k/qrnn_nl4.m
Training
Validation set not found using 10% of trn
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
Training lm from:  [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/wiki/fr-100/models/sp15k/qrnn_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/wiki/fr-100/models/sp15k/qrnn_nl4.m/../itos')]
epoch     train_loss  valid_loss  accuracy
1         4.844616    4.303658    0.358888
Total time: 02:32
epoch     train_loss  valid_loss  accuracy
1         4.558980    4.236440    0.368490
2         4.336353    4.111036    0.389355
3         4.215691    3.997827    0.406318
4         4.124804    3.892600    0.422174
5         4.052222    3.802997    0.435620
6         3.958529    3.729386    0.446996
7         3.889615    3.646023    0.460472
8         3.860830    3.590203    0.470407
9         3.763688    3.521166    0.482480
10        3.735285    3.465606    0.493439
11        3.699683    3.411256    0.504397
12        3.666552    3.347546    0.517700
13        3.580596    3.308754    0.526673
14        3.551433    3.255782    0.537871
15        3.517213    3.215991    0.546388
16        3.480192    3.188064    0.553690
17        3.416178    3.162537    0.559454
18        3.391725    3.151500    0.561885
19        3.400107    3.144848    0.563330
20        3.400756    3.143354    0.563702
Total time: 1:12:34
/home/pczapla/workspace/ulmfit-multilingual/data/cls/fr-books/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/cls/fr-books/models/sp15k/qrnn_nl4.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.532118    0.548179    0.820000
2         0.495962    0.509888    0.860000
3         0.427645    0.489982    0.900000
4         0.385989    0.513428    0.850000
5         0.372433    0.446763    0.905000
6         0.364960    0.467484    0.885000
7         0.350225    0.455279    0.905000
8         0.347019    0.466804    0.885000
Total time: 01:48
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/cls/fr-books/models/sp15k/qrnn_nl4.m
Loss and accuracy using (cls_best): [0.27588147, tensor(0.9130)]
Processing data/wiki/it-100/models/sp15k/qrnn_nl4.m
../cls/it-books
Processing data/wiki/ja-100/models/sp15k/qrnn_nl4.m
../cls/ja-books
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/cls/ja-books/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/cls/ja-books/models/sp15k/qrnn_nl4.m
Training
Validation set not found using 10% of trn
Error You have NaN values in column(s) of your dataframe, please fix it.
Processing data/wiki/ru-100/models/sp15k/qrnn_nl4.m
../cls/ru-books
Processing data/wiki/zh-100/models/sp15k/qrnn_nl4.m
../cls/zh-books
OrderedDict([('data/cls/en-books/models/sp15k/qrnn_nl4.m', 0.906000018119812),
             ('data/cls/fr-books/models/sp15k/qrnn_nl4.m', 0.9129999876022339)])
data/cls/en-books/models/sp15k/qrnn_nl4.m: 0.906000018119812
```
### All datasets
```bash
python -m ulmfit eval --glob="wiki/*-100/models/sp15k/qrnn_nl4.m" --name nl4 --dataset-template='../cls/${lang}-*' --num-lm-epochs=20  --num-cls-epochs=8  --bs=18 --lr_sched=1cycle --label-smoothing-eps=0.1
Processing data/wiki/de-100/models/sp15k/qrnn_nl4.m
../cls/de-*
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/cls/de-books/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/cls/de-books/models/sp15k/qrnn_nl4.m
Evaluating previously trained model
Validation set not found using 10% of trn
Data lm, trn: 152523, val: 16947
Data cls, trn: 1800, val: 200
Data tst, trn: 200, val: 2000
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
Loss and accuracy using (cls_best): [0.23488419, tensor(0.9320)]
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/cls/de-dvd/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/cls/de-dvd/models/sp15k/qrnn_nl4.m
Training
Validation set not found using 10% of trn
Running tokenization lm...
Data lm, trn: 85965, val: 9551
Running tokenization cls...
Data cls, trn: 1800, val: 200
Running tokenization tst...
Data tst, trn: 200, val: 2000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁.', '▁,', '▁der', '▁die', 'en', '▁und', 's', '▁in', 'er', "▁&'"]
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
Training lm from:  [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/wiki/de-100/models/sp15k/qrnn_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/wiki/de-100/models/sp15k/qrnn_nl4.m/../itos')]
epoch     train_loss  valid_loss  accuracy
1         4.596065    4.161942    0.436606
Total time: 11:33
epoch     train_loss  valid_loss  accuracy
1         4.127364    3.947751    0.466350
2         3.918618    3.776825    0.490808
3         3.800280    3.655171    0.506950
4         3.690864    3.582598    0.516176
5         3.698156    3.535314    0.522662
6         3.610991    3.498398    0.527869
7         3.592985    3.466045    0.532311
8         3.566916    3.444989    0.535283
9         3.563903    3.422111    0.539163
10        3.557671    3.400061    0.542508
11        3.486254    3.377414    0.546246
12        3.489782    3.356982    0.549719
13        3.451218    3.333650    0.553665
14        3.464072    3.312912    0.557414
15        3.431759    3.292514    0.560929
16        3.387984    3.274624    0.564236
17        3.405241    3.262639    0.566367
18        3.374643    3.253385    0.568177
19        3.374621    3.248559    0.569108
20        3.377307    3.247425    0.569270
Total time: 5:31:07
/home/pczapla/workspace/ulmfit-multilingual/data/cls/de-dvd/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/cls/de-dvd/models/sp15k/qrnn_nl4.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.519669    0.485357    0.865000
2         0.479460    0.426823    0.920000
3         0.431123    0.432494    0.885000
4         0.386461    0.431496    0.915000
5         0.365019    0.427543    0.920000
6         0.355551    0.415073    0.930000
7         0.346821    0.412974    0.930000
8         0.347380    0.413793    0.935000
Total time: 02:57
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/cls/de-dvd/models/sp15k/qrnn_nl4.m
Loss and accuracy using (cls_best): [0.2746336, tensor(0.9055)]
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/cls/de-music/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/cls/de-music/models/sp15k/qrnn_nl4.m
Training
Validation set not found using 10% of trn
Running tokenization lm...
Data lm, trn: 57953, val: 6439
Running tokenization cls...
Data cls, trn: 1800, val: 200
Running tokenization tst...
Data tst, trn: 200, val: 2000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁.', '▁,', '▁der', '▁die', 'en', '▁und', 's', '▁in', 'er', "▁&'"]
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
Training lm from:  [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/wiki/de-100/models/sp15k/qrnn_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/wiki/de-100/models/sp15k/qrnn_nl4.m/../itos')]
epoch     train_loss  valid_loss  accuracy
1         4.650980    4.152651    0.433679
Total time: 07:28
epoch     train_loss  valid_loss  accuracy
1         4.209763    3.965040    0.459827
2         3.935442    3.798236    0.482987
3         3.841647    3.667273    0.500565
4         3.699245    3.578814    0.512209
5         3.652425    3.512940    0.521151
6         3.658080    3.467384    0.527574
7         3.583572    3.427991    0.533427
8         3.558736    3.393701    0.538799
9         3.497921    3.360512    0.544095
10        3.513804    3.333250    0.548685
11        3.495725    3.303407    0.553937
12        3.442093    3.278184    0.558270
13        3.431648    3.249587    0.563309
14        3.394415    3.224657    0.567926
15        3.418715    3.201570    0.572469
16        3.334841    3.180893    0.576305
17        3.356635    3.166274    0.579287
18        3.339209    3.156838    0.581144
19        3.307631    3.151842    0.582141
20        3.313863    3.150544    0.582361
Total time: 3:28:27
/home/pczapla/workspace/ulmfit-multilingual/data/cls/de-music/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/cls/de-music/models/sp15k/qrnn_nl4.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.518283    0.456440    0.880000
2         0.474427    0.466113    0.865000
3         0.420782    0.572000    0.850000
4         0.377621    0.467714    0.865000
5         0.361156    0.430405    0.915000
6         0.345733    0.429622    0.890000
7         0.347064    0.419322    0.910000
8         0.342849    0.419672    0.900000
Total time: 02:41
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/cls/de-music/models/sp15k/qrnn_nl4.m
Loss and accuracy using (cls_best): [0.24861218, tensor(0.9300)]
Processing data/wiki/en-100/models/sp15k/qrnn_nl4.m
../cls/en-*
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/cls/en-books/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/cls/en-books/models/sp15k/qrnn_nl4.m
Evaluating previously trained model
Validation set not found using 10% of trn
Data lm, trn: 48600, val: 5400
Data cls, trn: 1800, val: 200
Data tst, trn: 200, val: 2000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁the', '▁,', 's', '▁.', '▁of', '▁and', '▁in', '▁to', '▁a', 'ed']
Loss and accuracy using (cls_best): [0.2731999, tensor(0.9075)]
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/cls/en-dvd/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/cls/en-dvd/models/sp15k/qrnn_nl4.m
Training
Validation set not found using 10% of trn
Running tokenization lm...
Data lm, trn: 30600, val: 3400
Running tokenization cls...
Data cls, trn: 1800, val: 200
Running tokenization tst...
Data tst, trn: 200, val: 2000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁the', '▁,', 's', '▁.', '▁of', '▁and', '▁in', '▁to', '▁a', 'ed']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
Training lm from:  [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/wiki/en-100/models/sp15k/qrnn_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/wiki/en-100/models/sp15k/qrnn_nl4.m/../itos')]
epoch     train_loss  valid_loss  accuracy
1         5.459846    4.880299    0.304455
Total time: 03:40
epoch     train_loss  valid_loss  accuracy
1         4.914763    4.696301    0.329923
2         4.679338    4.542424    0.351552
3         4.545784    4.381050    0.373713
4         4.435306    4.261026    0.388227
5         4.381686    4.146891    0.402335
6         4.339701    4.057630    0.414286
7         4.193101    3.985903    0.425350
8         4.188499    3.904590    0.437700
9         4.141142    3.831017    0.449915
10        4.073668    3.769752    0.460387
11        4.034776    3.704058    0.472169
12        4.017927    3.645482    0.483226
13        3.939799    3.571460    0.498373
14        3.954270    3.544665    0.504316
15        3.891879    3.494558    0.514514
16        3.816200    3.460811    0.522036
17        3.817204    3.436978    0.527317
18        3.837135    3.423382    0.530152
19        3.786701    3.413906    0.532207
20        3.794206    3.412058    0.532526
Total time: 1:43:34
/home/pczapla/workspace/ulmfit-multilingual/data/cls/en-dvd/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/cls/en-dvd/models/sp15k/qrnn_nl4.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.546413    0.469370    0.895000
2         0.487449    0.455633    0.880000
3         0.434380    0.471036    0.890000
4         0.402179    0.457251    0.885000
5         0.375344    0.444693    0.910000
6         0.356960    0.440818    0.900000
7         0.352019    0.427976    0.900000
8         0.347301    0.427571    0.905000
Total time: 02:39
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/cls/en-dvd/models/sp15k/qrnn_nl4.m
Loss and accuracy using (cls_best): [0.28846368, tensor(0.8930)]
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/cls/en-music/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/cls/en-music/models/sp15k/qrnn_nl4.m
Training
Validation set not found using 10% of trn
Running tokenization lm...
Data lm, trn: 26298, val: 2922
Running tokenization cls...
Data cls, trn: 1800, val: 200
Running tokenization tst...
Data tst, trn: 200, val: 2000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁the', '▁,', 's', '▁.', '▁of', '▁and', '▁in', '▁to', '▁a', 'ed']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
Training lm from:  [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/wiki/en-100/models/sp15k/qrnn_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/wiki/en-100/models/sp15k/qrnn_nl4.m/../itos')]
epoch     train_loss  valid_loss  accuracy
1         5.360476    4.782286    0.319278
Total time: 02:41
epoch     train_loss  valid_loss  accuracy
1         4.751396    4.553875    0.351461
2         4.583414    4.389054    0.374384
3         4.383287    4.250410    0.394109
4         4.299458    4.122505    0.409391
5         4.248813    4.014976    0.423715
6         4.149672    3.927886    0.434895
7         4.099454    3.838499    0.448373
8         4.001449    3.762594    0.459935
9         3.931528    3.687276    0.472542
10        3.904028    3.625594    0.483216
11        3.892965    3.560731    0.495610
12        3.851050    3.500596    0.506427
13        3.748619    3.447429    0.517146
14        3.783010    3.407018    0.526054
15        3.737605    3.356801    0.536732
16        3.642893    3.315823    0.544464
17        3.664337    3.291245    0.550013
18        3.636087    3.277205    0.553098
19        3.614743    3.269276    0.554796
20        3.615276    3.267146    0.555189
Total time: 1:16:19
/home/pczapla/workspace/ulmfit-multilingual/data/cls/en-music/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/cls/en-music/models/sp15k/qrnn_nl4.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.558971    0.567553    0.840000
2         0.512537    0.503813    0.855000
3         0.456603    0.525335    0.830000
4         0.404400    0.528679    0.870000
5         0.380277    0.508553    0.850000
6         0.370005    0.473182    0.875000
7         0.353120    0.475434    0.875000
8         0.349033    0.476645    0.875000
Total time: 02:17
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/cls/en-music/models/sp15k/qrnn_nl4.m
Loss and accuracy using (cls_best): [0.30121252, tensor(0.8945)]
Processing data/wiki/es-100/models/sp15k/qrnn_nl4.m
../cls/es-*
Processing data/wiki/fr-100/models/sp15k/qrnn_nl4.m
../cls/fr-*
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/cls/fr-books/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/cls/fr-books/models/sp15k/qrnn_nl4.m
Evaluating previously trained model
Validation set not found using 10% of trn
Data lm, trn: 33183, val: 3687
Data cls, trn: 1800, val: 200
Data tst, trn: 200, val: 2000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', 's', '▁.', "'", '▁la', '▁le', '▁et', '▁l', '▁à']
Loss and accuracy using (cls_best): [0.27542278, tensor(0.9125)]
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/cls/fr-dvd/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/cls/fr-dvd/models/sp15k/qrnn_nl4.m
Training
Validation set not found using 10% of trn
Running tokenization lm...
Data lm, trn: 12023, val: 1335
Running tokenization cls...
Data cls, trn: 1800, val: 200
Running tokenization tst...
Data tst, trn: 200, val: 2000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', 's', '▁.', "'", '▁la', '▁le', '▁et', '▁l', '▁à']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
Training lm from:  [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/wiki/fr-100/models/sp15k/qrnn_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/wiki/fr-100/models/sp15k/qrnn_nl4.m/../itos')]
epoch     train_loss  valid_loss  accuracy
1         5.066386    4.331831    0.359093
Total time: 00:59
epoch     train_loss  valid_loss  accuracy
1         4.724524    4.284094    0.365222
2         4.508605    4.191759    0.380180
3         4.362468    4.066796    0.399366
4         4.244251    3.960160    0.414862
5         4.175961    3.862056    0.428568
6         4.089641    3.769989    0.443156
7         3.996232    3.692802    0.455285
8         3.911177    3.609659    0.469151
9         3.834589    3.517031    0.485899
10        3.753677    3.444421    0.499121
11        3.693849    3.378458    0.512520
12        3.656308    3.304272    0.527849
13        3.619973    3.243898    0.539743
14        3.538301    3.187478    0.551900
15        3.542358    3.146652    0.561410
16        3.433141    3.108872    0.569471
17        3.397583    3.086333    0.574742
18        3.420117    3.073466    0.577351
19        3.364507    3.063923    0.579112
20        3.371653    3.063201    0.579386
Total time: 28:17
/home/pczapla/workspace/ulmfit-multilingual/data/cls/fr-dvd/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/cls/fr-dvd/models/sp15k/qrnn_nl4.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.566780    0.456278    0.895000
2         0.502839    0.438660    0.910000
3         0.451509    0.439451    0.915000
4         0.409567    0.458298    0.905000
5         0.375980    0.451819    0.875000
6         0.365615    0.429869    0.895000
7         0.354849    0.430782    0.900000
8         0.351484    0.429099    0.900000
Total time: 01:58
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/cls/fr-dvd/models/sp15k/qrnn_nl4.m
Loss and accuracy using (cls_best): [0.29015526, tensor(0.8955)]
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/cls/fr-music/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/cls/fr-music/models/sp15k/qrnn_nl4.m
Training
Validation set not found using 10% of trn
Running tokenization lm...
Data lm, trn: 17946, val: 1994
Running tokenization cls...
Data cls, trn: 1800, val: 200
Running tokenization tst...
Data tst, trn: 200, val: 2000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', 's', '▁.', "'", '▁la', '▁le', '▁et', '▁l', '▁à']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
Training lm from:  [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/wiki/fr-100/models/sp15k/qrnn_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/wiki/fr-100/models/sp15k/qrnn_nl4.m/../itos')]
epoch     train_loss  valid_loss  accuracy
1         4.902670    4.297194    0.366058
Total time: 01:43
epoch     train_loss  valid_loss  accuracy
1         4.611322    4.230604    0.375839
2         4.375579    4.099567    0.396295
3         4.221557    3.955517    0.419109
4         4.053433    3.821369    0.439689
5         3.936422    3.676852    0.463543
6         3.822320    3.571660    0.481323
7         3.738963    3.448354    0.505107
8         3.660149    3.338475    0.528736
9         3.517678    3.231433    0.552705
10        3.506994    3.129620    0.575138
11        3.405409    3.064906    0.591049
12        3.381207    2.970587    0.613629
13        3.252796    2.915546    0.627047
14        3.214425    2.860644    0.640598
15        3.154248    2.808179    0.653110
16        3.101139    2.774935    0.661620
17        3.052113    2.748979    0.667787
18        2.992890    2.738183    0.670336
19        3.059078    2.730828    0.671936
20        2.990193    2.729673    0.672222
Total time: 48:59
/home/pczapla/workspace/ulmfit-multilingual/data/cls/fr-music/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/cls/fr-music/models/sp15k/qrnn_nl4.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.506996    0.478671    0.880000
2         0.468533    0.474114    0.900000
3         0.426250    0.440435    0.910000
4         0.382522    0.441184    0.920000
5         0.368826    0.428872    0.925000
6         0.359141    0.421278    0.925000
7         0.345824    0.418836    0.925000
8         0.348028    0.422893    0.920000
Total time: 02:09
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/cls/fr-music/models/sp15k/qrnn_nl4.m
Loss and accuracy using (cls_best): [0.23497759, tensor(0.9340)]
Processing data/wiki/it-100/models/sp15k/qrnn_nl4.m
../cls/it-*
Processing data/wiki/ja-100/models/sp15k/qrnn_nl4.m
../cls/ja-*
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/cls/ja-books/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/cls/ja-books/models/sp15k/qrnn_nl4.m
Evaluating previously trained model
Validation set not found using 10% of trn
Data lm, trn: 156402, val: 17378
Data cls, trn: 1800, val: 200
Data tst, trn: 200, val: 2000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁、', '▁。', '▁の', '▁に', '▁を', '▁年', 'の', '▁は', '▁・', '▁)']
Loss and accuracy using (cls_best): [0.34613457, tensor(0.8630)]
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/cls/ja-dvd/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/cls/ja-dvd/models/sp15k/qrnn_nl4.m
Training
Validation set not found using 10% of trn
Running tokenization lm...
Data lm, trn: 65094, val: 7232
Running tokenization cls...
Data cls, trn: 1800, val: 200
Running tokenization tst...
Data tst, trn: 200, val: 2000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁、', '▁。', '▁の', '▁に', '▁を', '▁年', 'の', '▁は', '▁・', '▁)']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
Training lm from:  [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/wiki/ja-100/models/sp15k/qrnn_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/wiki/ja-100/models/sp15k/qrnn_nl4.m/../itos')]
epoch     train_loss  valid_loss  accuracy
1         5.160996    4.517923    0.367362
Total time: 06:17
epoch     train_loss  valid_loss  accuracy
1         4.543629    4.324594    0.393028
2         4.285625    4.133171    0.418336
3         4.129646    3.980379    0.436407
4         3.999502    3.868094    0.450510
5         3.955512    3.801580    0.458919
6         3.918293    3.744513    0.466177
7         3.851580    3.696822    0.472531
8         3.797014    3.660165    0.478426
9         3.784533    3.629932    0.482919
10        3.750223    3.600744    0.487628
11        3.746155    3.572618    0.492204
12        3.731790    3.549776    0.495799
13        3.721592    3.523684    0.500092
14        3.664623    3.501803    0.504120
15        3.653851    3.482831    0.507431
16        3.655076    3.466049    0.510687
17        3.621635    3.454597    0.512781
18        3.619130    3.447068    0.514104
19        3.598743    3.443191    0.514797
20        3.585334    3.441960    0.515080
Total time: 2:55:48
/home/pczapla/workspace/ulmfit-multilingual/data/cls/ja-dvd/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/cls/ja-dvd/models/sp15k/qrnn_nl4.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.572977    0.500271    0.845000
2         0.530233    0.530061    0.815000
3         0.478677    0.534202    0.845000
4         0.422542    0.536495    0.855000
5         0.382862    0.508948    0.820000
6         0.369442    0.500411    0.860000
7         0.364325    0.500543    0.840000
8         0.353970    0.498926    0.840000
Total time: 02:12
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/cls/ja-dvd/models/sp15k/qrnn_nl4.m
Loss and accuracy using (cls_best): [0.351684, tensor(0.8575)]
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/cls/ja-music/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/cls/ja-music/models/sp15k/qrnn_nl4.m
Training
Validation set not found using 10% of trn
Running tokenization lm...
Data lm, trn: 53903, val: 5989
Running tokenization cls...
Data cls, trn: 1800, val: 200
Running tokenization tst...
Data tst, trn: 200, val: 2000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁、', '▁。', '▁の', '▁に', '▁を', '▁年', 'の', '▁は', '▁・', '▁)']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
Training lm from:  [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/wiki/ja-100/models/sp15k/qrnn_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/wiki/ja-100/models/sp15k/qrnn_nl4.m/../itos')]
epoch     train_loss  valid_loss  accuracy
1         5.209339    4.578931    0.361889
Total time: 04:32
epoch     train_loss  valid_loss  accuracy
1         4.667555    4.387390    0.386283
2         4.378979    4.193202    0.412063
3         4.182499    4.027168    0.434250
4         4.075960    3.904261    0.449052
5         3.978663    3.817029    0.459996
6         3.933847    3.751421    0.468568
7         3.878065    3.694970    0.476144
8         3.796052    3.649740    0.483267
9         3.771566    3.604177    0.490818
10        3.737682    3.573293    0.495396
11        3.690553    3.530821    0.503269
12        3.654486    3.498471    0.508781
13        3.628883    3.467230    0.514346
14        3.606242    3.443541    0.519053
15        3.586881    3.420868    0.523608
16        3.581954    3.402043    0.527174
17        3.492296    3.383712    0.530720
18        3.522726    3.376451    0.532236
19        3.512796    3.371758    0.533116
20        3.461179    3.370992    0.533302
Total time: 2:06:47
/home/pczapla/workspace/ulmfit-multilingual/data/cls/ja-music/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/cls/ja-music/models/sp15k/qrnn_nl4.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.572634    0.558102    0.795000
2         0.538641    0.521115    0.830000
3         0.480845    0.544550    0.830000
4         0.420194    0.540962    0.830000
5         0.394283    0.502248    0.860000
6         0.371833    0.520424    0.840000
7         0.360100    0.511556    0.855000
8         0.353953    0.511548    0.850000
Total time: 01:51
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/cls/ja-music/models/sp15k/qrnn_nl4.m
Loss and accuracy using (cls_best): [0.34172097, tensor(0.8660)]
Processing data/wiki/ru-100/models/sp15k/qrnn_nl4.m
../cls/ru-*
Processing data/wiki/zh-100/models/sp15k/qrnn_nl4.m
../cls/zh-*
OrderedDict([('data/cls/de-books/models/sp15k/qrnn_nl4.m', 0.9319999814033508),
             ('data/cls/de-dvd/models/sp15k/qrnn_nl4.m', 0.9054999947547913),
             ('data/cls/de-music/models/sp15k/qrnn_nl4.m', 0.9300000071525574),
             ('data/cls/en-books/models/sp15k/qrnn_nl4.m', 0.9075000286102295),
             ('data/cls/en-dvd/models/sp15k/qrnn_nl4.m', 0.8930000066757202),
             ('data/cls/en-music/models/sp15k/qrnn_nl4.m', 0.8945000171661377),
             ('data/cls/fr-books/models/sp15k/qrnn_nl4.m', 0.9125000238418579),
             ('data/cls/fr-dvd/models/sp15k/qrnn_nl4.m', 0.8955000042915344),
             ('data/cls/fr-music/models/sp15k/qrnn_nl4.m', 0.9340000152587891),
             ('data/cls/ja-books/models/sp15k/qrnn_nl4.m', 0.8629999756813049),
             ('data/cls/ja-dvd/models/sp15k/qrnn_nl4.m', 0.8575000166893005),
             ('data/cls/ja-music/models/sp15k/qrnn_nl4.m', 0.8659999966621399)])
data/cls/de-books/models/sp15k/qrnn_nl4.m: 0.9319999814033508
data/cls/de-dvd/models/sp15k/qrnn_nl4.m:   0.9054999947547913
data/cls/de-music/models/sp15k/qrnn_nl4.m: 0.9300000071525574
data/cls/en-books/models/sp15k/qrnn_nl4.m: 0.9075000286102295
data/cls/en-dvd/models/sp15k/qrnn_nl4.m:   0.8930000066757202
data/cls/en-music/models/sp15k/qrnn_nl4.m: 0.8945000171661377
data/cls/fr-books/models/sp15k/qrnn_nl4.m: 0.9125000238418579
data/cls/fr-dvd/models/sp15k/qrnn_nl4.m:   0.8955000042915344
data/cls/fr-music/models/sp15k/qrnn_nl4.m: 0.9340000152587891
data/cls/ja-books/models/sp15k/qrnn_nl4.m: 0.8629999756813049
data/cls/ja-dvd/models/sp15k/qrnn_nl4.m:   0.8575000166893005
data/cls/ja-music/models/sp15k/qrnn_nl4.m: 0.8659999966621399
```
# Debugging issues 
## Without col merge
````
python -m ulmfit cls --dataset-path data/cls/${LANG}-books  --base-lm-path data/wiki-m/${LANG}-100/models/sp30k/lstm_nl4.m  --lang=${LANG} --name 'nl4' - train 20 --bs 20 --num-cls-epochs=8 --lr-sched=single
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/cls/fr-books/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/cls/fr-books/models/sp30k/lstm_nl4.m
Validation set not found using 10% of trn
Data lm, trn: 33183, val: 3687
Data cls, trn: 1800, val: 200
Running tokenization tst...
Data tst, trn: 200, val: 2000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁.', "'", 's', '▁la', '▁le', '▁et', '▁l', '▁à']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
Training lm from:  [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/wiki-m/fr-100/models/sp30k/lstm_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/wiki-m/fr-100/models/sp30k/lstm_nl4.m/../itos')]
epoch     train_loss  valid_loss  accuracy
1         3.356221    2.821012    0.518492
Total time: 00:21
epoch     train_loss  valid_loss  accuracy
1         3.041214    2.734799    0.524577
2         2.919576    2.648412    0.535661
3         2.822292    2.542236    0.549206
4         2.721790    2.414110    0.561852
5         2.596515    2.276732    0.579841
6         2.453715    2.140479    0.600370
7         2.333764    2.000186    0.621349
8         2.231092    1.873927    0.644259
9         2.101130    1.765473    0.660529
10        2.006949    1.666797    0.682196
11        1.905025    1.584023    0.696058
12        1.820798    1.513958    0.709841
13        1.751217    1.456632    0.720846
14        1.689076    1.410359    0.729947
15        1.646113    1.371438    0.739868
16        1.594153    1.346142    0.744577
17        1.564375    1.332298    0.746693
18        1.536557    1.322925    0.748995
19        1.532926    1.319159    0.749444
20        1.525449    1.318028    0.749815
Total time: 08:51
/home/pczapla/workspace/ulmfit-multilingual/data/cls/fr-books/models/sp30k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/cls/fr-books/models/sp30k/lstm_nl4.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.588118    0.581087    0.700000
2         0.504373    0.583527    0.720000
3         0.412651    0.538866    0.750000
4         0.295401    0.658459    0.750000
5         0.212442    1.054068    0.720000
6         0.126090    1.302099    0.745000
7         0.078312    1.307932    0.760000
8         0.050346    1.339740    0.745000
Total time: 00:35
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/cls/fr-books/models/sp30k/lstm_nl4.m
Loss and accuracy using (cls_best): [1.3513571, tensor(0.7700)]
1.351357102394104
0.7699999809265137
````

### FR books 
#### CLS second run 91.00%
````bash
python -m ulmfit cls --dataset-path data/cls/${LANG}-books  --base-lm-path data/wiki-m/${LANG}-100/models/sp30k/lstm_nl4.m  --lang=${LANG} --name 'nl4' - train 20 --bs 20 --num-cls-epochs=8 --lr-sched=single
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/cls/fr-books/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/cls/fr-books/models/sp30k/lstm_nl4.m
Validation set not found using 10% of trn
Running tokenization lm...
Data lm, trn: 33183, val: 3687
Running tokenization cls...
Data cls, trn: 1800, val: 200
Running tokenization tst...
Data tst, trn: 200, val: 2000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁.', "'", 's', '▁la', '▁le', '▁et', '▁l', '▁à']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
Training lm from:  [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/wiki-m/fr-100/models/sp30k/lstm_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/wiki-m/fr-100/models/sp30k/lstm_nl4.m/../itos')]
epoch     train_loss  valid_loss  accuracy
1         3.790325    3.409294    0.367234
Total time: 06:02
epoch     train_loss  valid_loss  accuracy
1         3.526303    3.326439    0.378936
2         3.466923    3.226977    0.392378
3         3.342312    3.111874    0.406997
4         3.244619    2.992510    0.422330
5         3.156150    2.877498    0.437467
6         3.070326    2.762509    0.453874
7         2.956969    2.651613    0.471552
8         2.878008    2.535935    0.491058
9         2.790110    2.438724    0.508560
10        2.684145    2.323467    0.528415
11        2.633781    2.231418    0.547093
12        2.535126    2.143523    0.564889
13        2.464436    2.055402    0.582077
14        2.330094    1.989257    0.596582
15        2.372371    1.924338    0.610048
16        2.190224    1.866912    0.621738
17        2.176868    1.834098    0.629221
18        2.168293    1.809196    0.633879
19        2.151132    1.797144    0.636382
20        2.130476    1.793351    0.637044
Total time: 2:30:05
/home/pczapla/workspace/ulmfit-multilingual/data/cls/fr-books/models/sp30k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/cls/fr-books/models/sp30k/lstm_nl4.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.314315    0.530879    0.865000
2         0.336746    0.468635    0.865000
3         0.255810    0.324242    0.870000
4         0.149121    0.480570    0.885000
5         0.093909    0.613743    0.890000
6         0.091678    0.660452    0.885000
7         0.049993    0.649642    0.910000
8         0.034218    0.640008    0.910000
Total time: 04:19
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/cls/fr-books/models/sp30k/lstm_nl4.m
Loss and accuracy using (cls_best): [0.5418505, tensor(0.9100)]
0.5418505072593689
0.9100000262260437
````
#### CLS second run 89.70%
```
Loading pretrained model
Unknown tokens 0, first 100: []
Training lm from:  [PosixPath('/home/n-waves/workspace/ulmfit-multilingual/data/wiki/fr-100/models/sp15k/qrnn_nl4.m/lm_best'), PosixPath('/home/n-waves/workspace/ulmfit-multilingual/data/wiki/fr-100/models/sp15k/qrnn_nl4.m/../itos')]
epoch     train_loss  valid_loss  accuracy
1         4.037580    3.312217    0.364410
Total time: 01:44
epoch     train_loss  valid_loss  accuracy
1         3.709982    3.256320    0.371825
2         3.459413    3.150574    0.386972
3         3.296628    3.037327    0.402039
4         3.186458    2.914899    0.418413
5         3.092632    2.817097    0.431216
6         2.966957    2.726081    0.442906
7         2.924824    2.647339    0.453871
8         2.818279    2.561596    0.466795
9         2.773893    2.501994    0.475877
10        2.736084    2.438490    0.485978
11        2.688937    2.370927    0.496899
12        2.615245    2.314875    0.506508
13        2.583292    2.260717    0.515725
14        2.535631    2.220295    0.522666
15        2.466035    2.179093    0.530148
16        2.461427    2.151952    0.535315
17        2.390641    2.131065    0.538749
18        2.376235    2.116927    0.541430
19        2.407630    2.115370    0.542039
20        2.391378    2.112687    0.542522
Total time: 46:33
/home/n-waves/workspace/ulmfit-multilingual/data/cls/fr-books/models/sp15k
Saving info /home/n-waves/workspace/ulmfit-multilingual/data/cls/fr-books/models/sp15k/qrnn_nl4.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.466671    0.534682    0.745000
2         0.358965    0.372612    0.875000
3         0.251557    0.311034    0.900000
4         0.166484    0.585425    0.865000
5         0.101803    0.726341    0.900000
6         0.072025    0.587875    0.885000
7         0.045328    0.760989    0.890000
8         0.027765    0.727203    0.890000
Total time: 01:17
Saving models at /home/n-waves/workspace/ulmfit-multilingual/data/cls/fr-books/models/sp15k/qrnn_nl4.m
Loss and accuracy using (cls_best): [0.55982095, tensor(0.8970)]
0.5598209500312805
0.8970000147819519
```

````bash
 python -m ulmfit eval --glob="wiki/*-100/models/sp15k/qrnn_nl4.m" --name nl4 --dataset-template='../cls/${lang}-books' --num-lm-epochs=20  --num-cls-epochs=8  --bs=18 --lr_sched=1cycle --label-smoothing-eps=0.1
Processing data/wiki/de-100/models/sp15k/qrnn_nl4.m
../cls/de-books
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/cls/de-books/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/cls/de-books/models/sp15k/qrnn_nl4.m
Training
Validation set not found using 10% of trn
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
Training lm from:  [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/wiki/de-100/models/sp15k/qrnn_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/wiki/de-100/models/sp15k/qrnn_nl4.m/../itos')]
epoch     train_loss  valid_loss  accuracy
1         4.659840    4.155068    0.438915
Total time: 19:49
epoch     train_loss  valid_loss  accuracy
1         4.047657    3.906174    0.473263
2         3.838604    3.750033    0.495044
3         3.759745    3.645392    0.508066
4         3.687292    3.590345    0.515256
5         3.627978    3.558625    0.519270
6         3.645294    3.535127    0.522458
7         3.598592    3.514614    0.525545
8         3.624991    3.498831    0.527508
9         3.579310    3.484266    0.529842
10        3.583765    3.466230    0.532676
11        3.582043    3.449772    0.535180
12        3.555684    3.431577    0.537932
13        3.547026    3.412131    0.541149
14        3.494887    3.394598    0.543857
15        3.508622    3.377836    0.546771
16        3.509469    3.362931    0.549247
17        3.491808    3.351186    0.551273
18        3.493448    3.342843    0.552928
19        3.430579    3.338350    0.553565
20        3.464007    3.337320    0.553756
Total time: 9:30:17
/home/pczapla/workspace/ulmfit-multilingual/data/cls/de-books/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/cls/de-books/models/sp15k/qrnn_nl4.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.483316    0.432666    0.930000
2         0.439075    0.419435    0.900000
3         0.413648    0.415576    0.930000
4         0.378170    0.421523    0.925000
5         0.362544    0.410525    0.935000
6         0.348756    0.407976    0.935000
7         0.345272    0.403396    0.935000
8         0.347165    0.408822    0.935000
Total time: 02:45
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/cls/de-books/models/sp15k/qrnn_nl4.m
Loss and accuracy using (cls_best): [0.23490253, tensor(0.9315)]
Processing data/wiki/en-100/models/sp15k/qrnn_nl4.m
../cls/en-books
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/cls/en-books/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/cls/en-books/models/sp15k/qrnn_nl4.m
Evaluating previously trained model
Validation set not found using 10% of trn
Data lm, trn: 48600, val: 5400
Data cls, trn: 1800, val: 200
Data tst, trn: 200, val: 2000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁the', '▁,', 's', '▁.', '▁of', '▁and', '▁in', '▁to', '▁a', 'ed']
Loss and accuracy using (cls_best): [0.2731999, tensor(0.9075)]
Processing data/wiki/es-100/models/sp15k/qrnn_nl4.m
../cls/es-books
Processing data/wiki/fr-100/models/sp15k/qrnn_nl4.m
../cls/fr-books
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/cls/fr-books/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/cls/fr-books/models/sp15k/qrnn_nl4.m
Evaluating previously trained model
Validation set not found using 10% of trn
Data lm, trn: 33183, val: 3687
Data cls, trn: 1800, val: 200
Data tst, trn: 200, val: 2000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', 's', '▁.', "'", '▁la', '▁le', '▁et', '▁l', '▁à']
Loss and accuracy using (cls_best): [0.27542278, tensor(0.9125)]
Processing data/wiki/it-100/models/sp15k/qrnn_nl4.m
../cls/it-books
Processing data/wiki/ja-100/models/sp15k/qrnn_nl4.m
../cls/ja-books
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/cls/ja-books/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/cls/ja-books/models/sp15k/qrnn_nl4.m
Training
Validation set not found using 10% of trn
Running tokenization lm...
Data lm, trn: 156402, val: 17378
Running tokenization cls...
Data cls, trn: 1800, val: 200
Running tokenization tst...
Data tst, trn: 200, val: 2000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁、', '▁。', '▁の', '▁に', '▁を', '▁年', 'の', '▁は', '▁・', '▁)']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
Training lm from:  [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/wiki/ja-100/models/sp15k/qrnn_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/wiki/ja-100/models/sp15k/qrnn_nl4.m/../itos')]
epoch     train_loss  valid_loss  accuracy
1         4.851995    4.372256    0.377945
Total time: 15:17
epoch     train_loss  valid_loss  accuracy
1         4.281115    4.121970    0.414018
2         4.072245    3.940537    0.438534
3         3.903796    3.822031    0.453624
4         3.863336    3.748684    0.462770
5         3.818518    3.706158    0.468450
6         3.755432    3.675938    0.472679
7         3.738127    3.648453    0.476536
8         3.717736    3.625716    0.479841
9         3.720993    3.605317    0.483234
10        3.695235    3.585903    0.486179
11        3.696312    3.567543    0.489165
12        3.688561    3.550402    0.492248
13        3.654837    3.533271    0.495245
14        3.667123    3.515244    0.498298
15        3.610541    3.499110    0.501204
16        3.589734    3.484431    0.503880
17        3.586404    3.474879    0.505702
18        3.590800    3.466892    0.507203
19        3.570113    3.462836    0.508037
20        3.544942    3.461981    0.508177
Total time: 7:11:56
/home/pczapla/workspace/ulmfit-multilingual/data/cls/ja-books/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/cls/ja-books/models/sp15k/qrnn_nl4.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.536773    0.625124    0.775000
2         0.509836    0.827109    0.735000
3         0.455667    0.636377    0.805000
4         0.411549    0.650928    0.725000
5         0.371732    0.594132    0.795000
6         0.363084    0.580896    0.810000
7         0.351636    0.569889    0.790000
8         0.343057    0.575629    0.790000
Total time: 02:15
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/cls/ja-books/models/sp15k/qrnn_nl4.m
Loss and accuracy using (cls_best): [0.3464668, tensor(0.8630)]
Processing data/wiki/ru-100/models/sp15k/qrnn_nl4.m
../cls/ru-books
Processing data/wiki/zh-100/models/sp15k/qrnn_nl4.m
../cls/zh-books
OrderedDict([('data/cls/de-books/models/sp15k/qrnn_nl4.m', 0.9315000176429749),
             ('data/cls/en-books/models/sp15k/qrnn_nl4.m', 0.9075000286102295),
             ('data/cls/fr-books/models/sp15k/qrnn_nl4.m', 0.9125000238418579),
             ('data/cls/ja-books/models/sp15k/qrnn_nl4.m', 0.8629999756813049)])
````
