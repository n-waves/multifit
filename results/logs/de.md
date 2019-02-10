= DE =
== VF60k LSTM nl 3 ==
=== LM ===
```
python -m ulmfit lm --dataset-path data/wiki/de-100 --cuda-id=1 --tokenizer='vf' --nl 3 --name 'nl3' --max-vocab 60000  --lang de --qrnn=False - train 10 --bs=50 --drop_mult=0
Max vocab: 60000
Cache dir: data/wiki/de-100/models/vf60k
Model dir: data/wiki/de-100/models/vf60k/lstm_nl3.m
Running tokenization
Wiki text was split to 175965 articles
Wiki text was split to 110 articles
Size of vocabulary: 60003
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', ',', 'der', '.', 'und', 'die', 'in', "&'", 'von', 'den', '(', 'im', ')']
Training args:  {'tie_weights': True, 'clip': 0.12, 'bptt': 70, 'pretrained_fnames': None, 'pretrained_model': None, 'drop_mult': 0} dps:  [0.25 0.1  0.2  0.02 0.15]
Training lm from random weights
epoch     train_loss  valid_loss  accuracy
1         3.214624    3.573368    0.397312
2         3.194401    3.549021    0.396143
3         3.116934    3.535322    0.398108
4         3.159205    3.498862    0.400490
5         3.104538    3.454015    0.405504
6         2.996653    3.410940    0.409791
7         2.987909    3.359425    0.413711
8         2.941863    3.311215    0.419416
9         2.914403    3.285807    0.423674
10        2.857530    3.278313    0.425131
data/wiki/de-100/models/vf60k
Saving info data/wiki/de-100/models/vf60k/lstm_nl3.m/info.json
```
=== MLDocs ===
...
== SP30k LSTM nl 4 ==
=== LM ===
```
python -m ulmfit lm --dataset-path data/wiki/de-100 --cuda-id=0 --tokenizer='sp' --nl 4 --name 'nl4' --max-vocab 30000  --lang sp --qrnn=False - train 10 --bs=50 --drop_mult=0
1,2.833101,3.174348,0.472863
2,2.788717,3.171983,0.471377
3,2.831292,3.187135,0.471068
4,2.723390,3.133801,0.475572
5,2.681617,3.064743,0.481984
6,2.662792,2.984701,0.489080
7,2.542035,2.892254,0.499275
8,2.422225,2.806846,0.508663
9,2.462655,2.736171,0.517994
10,2.396778,2.714520,0.521145
data/wiki/de-100/models/sp30k/lstm_nl4.m/lm-history.csv
```

=== MLDocs ===
```
python -m ulmfit cls --dataset-path data/mldoc/de-1  --base-lm-path data/wiki/de-100/models/sp30k/lstm_nl4.m  --lang=de --name 'nl4' - train 20 --bs 40     ✘ 1
Max vocab: 30000
Cache dir: data/mldoc/de-1/models/sp30k
Model dir: data/mldoc/de-1/models/sp30k/lstm_nl4.m
Loading validation data/mldoc/de-1/de.dev.csv
Tokenized data loaded, lm.trn 13500, lm.val 1500
Tokenized data loaded, cls.trn 1000, cls.val 1000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁.', '▁,', '▁der', '▁die', '▁und', '▁in', 'en', "▁&'", 's', '-']
Training args:  {'tie_weights': True, 'clip': 0.12, 'bptt': 70, 'pretrained_fnames': [PosixPath('data/wiki/de-100/models/sp30k/lstm_nl4.m/lm_best'), PosixPath('data/wiki/de-100/models/sp30k/lstm_nl4.m/../itos')], 'pretrained_model': None, 'drop_mult': 0.3} dps:  [0.25 0.1  0.2  0.02 0.15]
Unknown tokens 0, first 100: []
Training lm from:  [PosixPath('data/wiki/de-100/models/sp30k/lstm_nl4.m/lm_best'), PosixPath('data/wiki/de-100/models/sp30k/lstm_nl4.m/../itos')]
epoch     train_loss  valid_loss  accuracy
1         3.042075    2.457199    0.547201
epoch     train_loss  valid_loss  accuracy
1         2.581403    2.305440    0.565500
2         2.366814    2.139165    0.589417
3         2.187646    1.986698    0.612081
4         2.054434    1.857322    0.630642
5         1.948663    1.758499    0.644389
6         1.850596    1.673632    0.655852
7         1.813331    1.593225    0.668256
8         1.738136    1.523946    0.678633
9         1.683469    1.463405    0.688561
10        1.609236    1.410462    0.697171
11        1.599416    1.356008    0.706997
12        1.526982    1.308399    0.715433
13        1.487115    1.263120    0.723749
14        1.430917    1.224060    0.731837
15        1.410333    1.191501    0.738267
16        1.385961    1.166404    0.743477
17        1.349813    1.144801    0.747553
18        1.345938    1.132679    0.750188
19        1.311102    1.127321    0.751208
20        1.355743    1.126064    0.751384
Saving info data/mldoc/de-1/models/sp30k/lstm_nl4.m/info.json
Starting classifier training
epoch     train_loss  valid_loss  accuracy
1         0.490199    0.246640    0.940000
epoch     train_loss  valid_loss  accuracy
1         0.302251    0.243051    0.932000
epoch     train_loss  valid_loss  accuracy
1         0.211028    0.249550    0.932000
epoch     train_loss  valid_loss  accuracy
1         0.159555    0.230822    0.947000
2         0.144418    0.226450    0.943000
Saving models at data/mldoc/de-1/models/sp30k/lstm_nl4.m
Loss and accuracy using (cls_last): [0.22645034, tensor(0.9430)]
Loss and accuracy using (cls_best): [0.22645034, tensor(0.9430)]
```
MultiCCA: 93.7% , ulmfit: 94.3%
 