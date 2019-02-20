# ZH

## SP15k QRNN

## SP30k LSTM nl 4
### LM
```
 python -m ulmfit lm --dataset-path data/wiki/zh-100 --cuda-id=0 --tokenizer='sp' --nl 4 --name 'nl4' --max-vocab 30000 --lang zh --qrnn=False - train 10 --bs=50 --drop_mult=0
Max vocab: 30000
Cache dir: data/wiki/zh-100/models/sp30k
Model dir: data/wiki/zh-100/models/sp30k/lstm_nl4.m
Tokenized data loaded
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁的', '▁。', '▁年', '▁、', '▁在', '▁一', '▁中', '▁人', '▁是']
Training args:  {'tie_weights': True, 'clip': 0.12, 'bptt': 70, 'pretrained_fnames': None, 'pretrained_model': None, 'drop_mult': 0} dps:  [0.25 0.1  0.2  0.02 0.15]
Training lm from random weights
epoch     train_loss  valid_loss  accuracy
1         2.736679    3.050473    0.428462
2         2.664505    3.011505    0.432414
3         2.607435    2.942389    0.439985
4         2.561503    2.851523    0.451965
5         2.499060    2.798222    0.459438
6         2.387191    2.720054    0.471021
7         2.356725    2.648299    0.479029
8         2.301895    2.553860    0.493597
9         2.275601    2.481724    0.505979
10        2.187606    2.465159    0.509590
```
### MLDoc
```
python -m ulmfit cls --dataset-path data/mldoc/zh-1  --base-lm-path data/wiki/zh-100/models/sp30k/lstm_nl4.m  --lang=zh --name 'nl4' --cuda-id=0 - train 20 --bs 40 --num-cls-epochs=2
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp30k/lstm_nl4.m
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/zh.dev.csv
Tokenized data loaded, lm.trn 13500, lm.val 1500
Tokenized data loaded, cls.trn 1000, cls.val 1000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁的', '▁。', '▁年', '▁、', '▁在', '▁一', '▁中', '▁人', '▁是']
Training args:  {'tie_weights': True, 'clip': 0.12, 'bptt': 70, 'pretrained_fnames': [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/wiki/zh-100/models/sp30k/lstm_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/wiki/zh-100/models/sp30k/lstm_nl4.m/../itos')], 'pretrained_model': None, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  [0.25 0.1  0.2  0.02 0.15]
Unknown tokens 0, first 100: []
Training lm from:  [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/wiki/zh-100/models/sp30k/lstm_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/wiki/zh-100/models/sp30k/lstm_nl4.m/../itos')]
epoch     train_loss  valid_loss  accuracy
1         2.604460    2.225315    0.546099
epoch     train_loss  valid_loss  accuracy
1         2.240892    2.020697    0.578796
2         2.025043    1.816424    0.613192
3         1.832658    1.646025    0.640532
4         1.746628    1.530125    0.659058
5         1.621672    1.425179    0.675305
6         1.544814    1.345650    0.689195
7         1.464704    1.271710    0.702200
8         1.412583    1.204830    0.714764
9         1.332440    1.147108    0.725389
10        1.327941    1.092910    0.736447
11        1.227284    1.039441    0.747662
12        1.200814    0.991910    0.758105
13        1.161579    0.947898    0.768121
14        1.100010    0.908599    0.776732
15        1.059006    0.872309    0.785161
16        1.045412    0.844972    0.791998
17        1.026688    0.824872    0.796891
18        1.013831    0.812786    0.799699
19        0.978586    0.807678    0.800954
20        0.982473    0.805671    0.801201
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp30k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp30k/lstm_nl4.m/info.json
Starting classifier training
epoch     train_loss  valid_loss  accuracy
1         0.637427    0.505143    0.836000
epoch     train_loss  valid_loss  accuracy
1         0.471189    0.317678    0.887000
epoch     train_loss  valid_loss  accuracy
1         0.384985    0.288901    0.904000
epoch     train_loss  valid_loss  accuracy
1         0.316358    0.275456    0.906000
2         0.295534    0.278589    0.907000
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp30k/lstm_nl4.m
Loss and accuracy using (cls_best): [0.28411642, tensor(0.9020)]
0.2841164171695709
0.9020000100135803
```



## SP60k LSTM nl 4
### LM
```
Wiki text was split to 153503 articles
Wiki text was split to 145 articles
Size of vocabulary: 60000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁的', '▁。', '▁年', '▁、', '▁在', '▁一', '▁中', '▁是', '▁人']
Training args:  {'tie_weights': True, 'clip': 0.12, 'bptt': 70, 'pretrained_fnames': None, 'pretrained_model': None, 'drop_mult': 0} dps:  [0.25 0.1  0.2  0.02 0.15]
Training lm from random weights
epoch     train_loss  valid_loss  accuracy
1         3.312704    3.701317    0.334740
2         3.212988    3.648671    0.336709
3         3.060103    3.584413    0.344427
4         3.108131    3.477978    0.356738
5         2.952951    3.410785    0.365901
6         2.919397    3.325265    0.376316
7         2.839392    3.224750    0.391707
8         2.750095    3.132644    0.404416
9         2.805704    3.066595    0.415245
10        2.653435    3.055314    0.417736
data/wiki/zh-100/models/sp60k
Saving info data/wiki/zh-100/models/sp60k/lstm_nl4.m/info.json
```
### MLDoc
```
python -m ulmfit cls --dataset-path data/mldoc/zh-1  --base-lm-path data/wiki/zh-100/models/sp60k/lstm_nl4.m  --lang=zh --name 'nl4' --cu
da-id=0 - train 20 --bs 40 --num-cls-epochs=2
Max vocab: 60000
Cache dir: /home/n-waves/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp60k
Model dir: /home/n-waves/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp60k/lstm_nl4.m
Loading validation /home/n-waves/workspace/ulmfit-multilingual/data/mldoc/zh-1/zh.dev.csv
Tokenized data loaded, lm.trn 13500, lm.val 1500
Tokenized data loaded, cls.trn 1000, cls.val 1000
Size of vocabulary: 60000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁的', '▁。', '▁年', '▁、', '▁在', '▁一', '▁中', '▁是', '▁人']
Training args:  {'tie_weights': True, 'clip': 0.12, 'bptt': 70, 'pretrained_fnames': [PosixPath('/home/n-waves/workspace/ulmfit-multilingual/data/wiki/zh-100/models/sp60k/lstm_nl4.m/lm_best'), Po
sixPath('/home/n-waves/workspace/ulmfit-multilingual/data/wiki/zh-100/models/sp60k/lstm_nl4.m/../itos')], 'pretrained_model': None, 'drop_mult': 0.3} dps:  [0.25 0.1  0.2  0.02 0.15]
Unknown tokens 0, first 100: []
Training lm from:  [PosixPath('/home/n-waves/workspace/ulmfit-multilingual/data/wiki/zh-100/models/sp60k/lstm_nl4.m/lm_best'), PosixPath('/home/n-waves/workspace/ulmfit-multilingual/data/wiki/zh-
100/models/sp60k/lstm_nl4.m/../itos')]
epoch     train_loss  valid_loss  accuracy
1         3.055914    2.690310    0.467917
epoch     train_loss  valid_loss  accuracy
1         2.713421    2.464873    0.503386
2         2.429520    2.215309    0.543961
3         2.247576    2.010849    0.578106
4         2.083628    1.853473    0.602419
5         1.969939    1.734762    0.621440
6         1.904438    1.624005    0.640240
7         1.783416    1.526202    0.656981
8         1.719215    1.445780    0.671753
9         1.621891    1.366912    0.687187
10        1.589463    1.295759    0.701207
11        1.510032    1.223578    0.716387
12        1.404720    1.160607    0.729603
13        1.414636    1.107378    0.741273
14        1.364716    1.056422    0.753112
15        1.327804    1.011525    0.763934
16        1.255990    0.976447    0.771864
17        1.181438    0.951213    0.778309
18        1.192709    0.936060    0.781858
19        1.190164    0.928613    0.783513
20        1.172130    0.927612    0.783722
/home/n-waves/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp60k
Saving info /home/n-waves/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp60k/lstm_nl4.m/info.json
Starting classifier training
epoch     train_loss  valid_loss  accuracy
1         0.646537    0.516221    0.836000
epoch     train_loss  valid_loss  accuracy
1         0.441884    0.361802    0.873000
epoch     train_loss  valid_loss  accuracy
1         0.376583    0.318426    0.893000
epoch     train_loss  valid_loss  accuracy
1         0.280910    0.314279    0.889000
2         0.308887    0.309718    0.903000
Saving models at /home/n-waves/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp60k/lstm_nl4.m
Loss and accuracy using (cls_last): [0.30276635, tensor(0.8978)]
```