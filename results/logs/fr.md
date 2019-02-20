# FR 

## SP15k QRNN nl 4
```
python -m ulmfit lm --dataset-path data/wiki/${LANG}-100 --tokenizer='sp' --nl 4 --name 'nl4' --max-vocab 15000 --lang ${LANG} --qrnn=True - train 10 --bs=50 --drop_mult=0
Max vocab: 15000
Cache dir: data/wiki/fr-100/models/sp15k
Model dir: data/wiki/fr-100/models/sp15k/qrnn_nl4.m
Wiki text was split to 174227 articles
Wiki text was split to 491 articles
Running tokenization lm...
Data lm, trn: 174227, val: 491
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', 's', '▁.', "'", '▁la', '▁le',
'▁et', '▁l', '▁à']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Training lm from random weights
epoch     train_loss  valid_loss  accuracy
1         2.881558    2.790402    0.465847
2         2.824942    2.732005    0.471660
3         2.758845    2.672040    0.478273
4         2.715069    2.602380    0.489159
5         2.677029    2.553575    0.494752
6         2.602514    2.476142    0.507337
7         2.564386    2.388670    0.518902
8         2.470835    2.304033    0.532000
9         2.366890    2.243269    0.542781
10        2.390439    2.223538    0.546622
Total time: 9:09:26
data/wiki/fr-100/models/sp15k
Saving info data/wiki/fr-100/models/sp15k/qrnn_nl4.m/info.json

```

## SP30k LSTM nl 4 
### LM 
```
python -m ulmfit lm --dataset-path data/wiki/fr-100 --cuda-id=1 --tokenizer='sp' --nl 4 --name 'nl4' --max-vocab 30000 \                                  ✘ 130
--lang fr --qrnn=False - train 10 --bs=50 --drop_mult=0
Max vocab: 30000
Cache dir: data/wiki/fr-100/models/sp30k
Model dir: data/wiki/fr-100/models/sp30k/lstm_nl4.m
Running tokenization
Wiki text was split to 113288 articles
Wiki text was split to 88 articles
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁.', "'", 's', '▁la', '▁le', '▁et', '▁l', '▁à']
Training args:  {'tie_weights': True, 'clip': 0.12, 'bptt': 70, 'pretrained_fnames': None, 'pretrained_model': None, 'drop_mult': 0} dps:  [0.25 0.1  0.2  0.02 0.15]
Training lm from random weights
epoch     train_loss  valid_loss  accuracy
1         3.120035    3.449028    0.383274
2         3.070353    3.431372    0.382520
3         3.092097    3.406521    0.384604
4         3.035016    3.356502    0.391155
5         2.936505    3.297572    0.396365
6         2.926953    3.192980    0.407546
7         2.841542    3.115280    0.417741
8         2.805254    3.008793    0.429512
9         2.681713    2.944207    0.439959
10        2.644920    2.923765    0.442415
data/wiki/fr-100/models/sp30k
Saving info data/wiki/fr-100/models/sp30k/lstm_nl4.m/info.json
```

### MLDocs 
#### First run
MultiCCA 92.05, ulmfit 93.90
```
python -m ulmfit cls --dataset-path data/mldoc/fr-1  --base-lm-path data/wiki/fr-100/models/sp30k/lstm_nl4.m  --lang=fr --name 'nl4' --cuda-id=1 - train 20 --bs 40
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/models/sp30k/lstm_nl4.m
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/fr.dev.csv
Running tokenization...
Saving tokenized: cls.trn 13500, cls.val 1500
Running tokenization...
Saving tokenized: cls.trn 1000, cls.val 1000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁.', "'", 's', '▁la', '▁le', '▁et', '▁l', '▁à']
Training args:  {'tie_weights': True, 'clip': 0.12, 'bptt': 70, 'pretrained_fnames': [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/wiki/fr-100/models/sp30k/lstm_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/wiki/fr-100/models/sp30k/lstm_nl4.m/../itos')], 'pretrained_model': None, 'drop_mult': 0.3} dps:  [0.25 0.1  0.2  0.02 0.15]
Unknown tokens 0, first 100: []
Training lm from:  [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/wiki/fr-100/models/sp30k/lstm_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/wiki/fr-100/models/sp30k/lstm_nl4.m/../itos')]
epoch     train_loss  valid_loss  accuracy
1         3.072937    2.621444    0.468314
epoch     train_loss  valid_loss  accuracy
1         2.737065    2.485359    0.486546
2         2.625827    2.353188    0.507669
3         2.408049    2.224600    0.527609
4         2.332804    2.113603    0.544255
5         2.242967    2.016229    0.560002
6         2.162170    1.925214    0.574050
7         2.094163    1.843778    0.587944
8         2.011285    1.773228    0.599802
9         1.931492    1.708201    0.611245
10        1.883735    1.643842    0.623145
11        1.793858    1.583394    0.635366
12        1.759305    1.526640    0.646132
13        1.741412    1.474198    0.657485
14        1.675670    1.430597    0.666407
15        1.624235    1.390453    0.674829
16        1.588415    1.359892    0.681364
17        1.594124    1.336594    0.686985
18        1.567758    1.322139    0.689745
19        1.536472    1.315883    0.690974
20        1.530144    1.314872    0.691084
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/models/sp30k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/models/sp30k/lstm_nl4.m/info.json
Starting classifier training
epoch     train_loss  valid_loss  accuracy
1         0.528480    0.428756    0.853000
epoch     train_loss  valid_loss  accuracy
1         0.365323    0.224117    0.928000
epoch     train_loss  valid_loss  accuracy
1         0.300881    0.199623    0.936000
epoch     train_loss  valid_loss  accuracy
1         0.217855    0.198016    0.937000
2         0.206357    0.212208    0.938000
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/models/sp30k/lstm_nl4.m
Loss and accuracy using (cls_best): [0.18914989, tensor(0.9390)]
```
 
#### Second run
MultiCCA 92.05, ulmfit 93.67
```
python -m ulmfit cls --dataset-path data/mldoc/fr-1  --base-lm-path data/wiki/fr-100/models/sp30k/lstm_nl4.m  --lang=fr --name 'nl4-2nd' --cuda-id=1 - train 20 --bs 40 --num-cls-epochs=8
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/models/sp30k/lstm_nl4-2nd.m
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/fr.dev.csv
Tokenized data loaded, lm.trn 13500, lm.val 1500
Tokenized data loaded, cls.trn 1000, cls.val 1000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁.', "'", 's', '▁la', '▁le', '▁et', '▁l', '▁à']
Training args:  {'tie_weights': True, 'clip': 0.12, 'bptt': 70, 'pretrained_fnames': [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/wiki/fr-100/models/sp30k/lstm_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/wiki/fr-100/models/sp30k/lstm_nl4.m/../itos')], 'pretrained_model': None, 'drop_mult': 0.3} dps:  [0.25 0.1  0.2  0.02 0.15]
Unknown tokens 0, first 100: []
Training lm from:  [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/wiki/fr-100/models/sp30k/lstm_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/wiki/fr-100/models/sp30k/lstm_nl4.m/../itos')]
epoch     train_loss  valid_loss  accuracy
1         3.080331    2.625329    0.467306
epoch     train_loss  valid_loss  accuracy
1         2.750554    2.485271    0.486776
2         2.578659    2.353940    0.507637
3         2.422981    2.224983    0.527749
4         2.342781    2.113364    0.545006
5         2.254575    2.007775    0.560709
6         2.124016    1.920536    0.575680
7         2.068470    1.847463    0.586699
8         2.013289    1.775580    0.599840
9         1.929649    1.705369    0.612201
10        1.916013    1.646228    0.623175
11        1.825515    1.586714    0.634298
12        1.795780    1.529771    0.645840
13        1.725532    1.476651    0.656197
14        1.673942    1.429790    0.666030
15        1.639384    1.392116    0.674128
16        1.605681    1.359316    0.681356
17        1.560283    1.337794    0.686116
18        1.543926    1.323153    0.689276
19        1.531950    1.318164    0.690415
20        1.494068    1.316459    0.690586
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/models/sp30k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/models/sp30k/lstm_nl4-2nd.m/info.json
Starting classifier training
epoch     train_loss  valid_loss  accuracy
1         0.537720    0.395818    0.886000
epoch     train_loss  valid_loss  accuracy
1         0.332603    0.232112    0.930000
epoch     train_loss  valid_loss  accuracy
1         0.267323    0.230307    0.927000
epoch     train_loss  valid_loss  accuracy
1         0.216402    0.226042    0.930000
2         0.231040    0.232696    0.936000
3         0.182048    0.217882    0.934000
4         0.170389    0.212531    0.937000
5         0.148332    0.214293    0.937000
6         0.124968    0.210322    0.936000
7         0.117591    0.234207    0.936000
8         0.109146    0.218597    0.938000
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/models/sp30k/lstm_nl4-2nd.m
Loss and accuracy using (cls_best): [0.21502711, tensor(0.9367)]
```