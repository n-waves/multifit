# MLDoc

## QRNN 15k

Exec 1
```
python -m ulmfit eval --glob="mldoc/zh-1/models/sp15k/qrnn_nl4.m" --name nl4-1cyc-sl  --num-cls-epochs=8 --bs=18 --lr_sched=1cycle --label-smoothing-eps=0.1
Processing data/mldoc/zh-1/models/sp15k/qrnn_nl4.m
zh-1
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp15k/qrnn_nl4-1cyc-sl.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/zh.dev.csv
Data lm, trn: 13500, val: 1500
Data cls, trn: 1000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁的', '▁。', '▁年', '▁、', '▁在', '▁一', '▁是', '▁中', '▁有']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
/home/pczapla/anaconda3/envs/fastaiv1/lib/python3.7/site-packages/torch/utils/cpp_extension.py:152: UserWarning:
Loading pretrained model
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp15k/qrnn_nl4-1cyc-sl.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.859153    0.767961    0.864000
2         0.768888    0.775161    0.904000
3         0.658956    0.685653    0.902000
4         0.589073    0.618438    0.923000
5         0.540008    0.622157    0.915000
6         0.508080    0.606979    0.914000
7         0.487228    0.599491    0.918000
8         0.477516    0.602196    0.923000
Total time: 02:18
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp15k/qrnn_nl4-1cyc-sl.m
Loss and accuracy using (cls_best): [0.2829206, tensor(0.9205)]
OrderedDict([('data/mldoc/zh-1/models/sp15k/qrnn_nl4-1cyc-sl.m',
              0.9204999804496765)])
data/mldoc/zh-1/models/sp15k/qrnn_nl4-1cyc-sl.m: 0.9204999804496765
```
Exec 2
````python -m ulmfit eval --glob="mldoc/zh-1/models/sp15k/qrnn_nl4.m" --name nl4-1cyc-sl1  --num-cls-epochs=8 --bs=18 --lr_sched=1cycle --label-smoothing-eps=0.1
Processing data/mldoc/zh-1/models/sp15k/qrnn_nl4.m
zh-1
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp15k/qrnn_nl4-1cyc-sl1.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/zh.dev.csv
Data lm, trn: 13500, val: 1500
Data cls, trn: 1000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁的', '▁。', '▁年', '▁、', '▁在', '▁一', '▁是', '▁中', '▁有']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp15k/qrnn_nl4-1cyc-sl1.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.881513    0.712176    0.865000
2         0.743687    0.665091    0.906000
3         0.677436    0.687689    0.873000
4         0.595139    0.626483    0.920000
5         0.542732    0.600652    0.914000
6         0.512080    0.597546    0.916000
7         0.487021    0.597065    0.912000
8         0.476598    0.596792    0.914000
Total time: 02:20
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp15k/qrnn_nl4-1cyc-sl1.m
Loss and accuracy using (cls_best): [0.29172945, tensor(0.9178)]
OrderedDict([('data/mldoc/zh-1/models/sp15k/qrnn_nl4-1cyc-sl1.m',
              0.9177500009536743)])
data/mldoc/zh-1/models/sp15k/qrnn_nl4-1cyc-sl1.m: 0.9177500009536743
````
Exec 4
```bash
python -m ulmfit eval --glob="mldoc/zh-1/models/sp15k/qrnn_nl4.m" --name nl4-1cyc-sl-e4  --num-cls-epochs=4 --bs=18 --lr_sched=1cycle --label-smoothing-eps=0.1               ✘ 130
Processing data/mldoc/zh-1/models/sp15k/qrnn_nl4.m
zh-1
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp15k/qrnn_nl4-1cyc-sl-e4.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/zh.dev.csv
Data lm, trn: 13500, val: 1500
Data cls, trn: 1000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁的', '▁。', '▁年', '▁、', '▁在', '▁一', '▁是', '▁中', '▁有']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp15k/qrnn_nl4-1cyc-sl-e4.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.880415    0.677291    0.901000
2         0.729670    0.659975    0.911000
3         0.624817    0.603056    0.921000
4         0.542027    0.601961    0.921000
Total time: 01:08
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp15k/qrnn_nl4-1cyc-sl-e4.m
Loss and accuracy using (cls_best): [0.28558904, tensor(0.9222)]
OrderedDict([('data/mldoc/zh-1/models/sp15k/qrnn_nl4-1cyc-sl-e4.m',
              0.922249972820282)])
data/mldoc/zh-1/models/sp15k/qrnn_nl4-1cyc-sl-e4.m: 0.922249972820282
```
## LSTM sp30k
### 0.1
```bash
 python -m ulmfit eval --glob="mldoc/zh-1/models/sp30k/lstm_nl4.m" --name nl4-1cyc-sl  --num-cls-epochs=8 --bs=18 --lr_sched=1cycle --label-smoothing-eps=0.1
Processing data/mldoc/zh-1/models/sp30k/lstm_nl4.m
zh-1
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp30k/lstm_nl4-1cyc-sl.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/zh.dev.csv
Data lm, trn: 13500, val: 1500
Data cls, trn: 1000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁的', '▁。', '▁年', '▁、', '▁在', '▁一', '▁中', '▁人', '▁是']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp30k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp30k/lstm_nl4-1cyc-sl.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.870432    0.670671    0.882000
2         0.754248    0.824157    0.895000
3         0.654601    0.727428    0.885000
4         0.602772    0.668668    0.901000
5         0.542110    0.625137    0.903000
6         0.506150    0.617842    0.913000
7         0.480944    0.616885    0.912000
8         0.472876    0.614381    0.911000
Total time: 06:38
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp30k/lstm_nl4-1cyc-sl.m
Loss and accuracy using (cls_best): [0.2977172, tensor(0.9233)]
OrderedDict([('data/mldoc/zh-1/models/sp30k/lstm_nl4-1cyc-sl.m',
              0.9232500195503235)])
data/mldoc/zh-1/models/sp30k/lstm_nl4-1cyc-sl.m: 0.9232500195503235
```
### 0.2
```bash
python -m ulmfit eval --glob="mldoc/zh-1/models/sp30k/lstm_nl4.m" --name nl4-1cyc-sl2  --num-cls-epochs=8 --bs=18 --lr_sched=1cycle --label-smoothing-eps=0.2
Processing data/mldoc/zh-1/models/sp30k/lstm_nl4.m
zh-1
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp30k/lstm_nl4-1cyc-sl2.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/zh.dev.csv
Data lm, trn: 13500, val: 1500
Data cls, trn: 1000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁的', '▁。', '▁年', '▁、', '▁在', '▁一', '▁中', '▁人', '▁是']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp30k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp30k/lstm_nl4-1cyc-sl2.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.045619    0.908213    0.874000
2         0.957379    0.857977    0.921000
3         0.891791    0.852157    0.905000
4         0.845289    0.849923    0.914000
5         0.818228    0.848613    0.921000
6         0.787021    0.840483    0.920000
7         0.776123    0.844006    0.919000
8         0.762384    0.857240    0.916000
Total time: 06:33
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp30k/lstm_nl4-1cyc-sl2.m
Loss and accuracy using (cls_best): [0.40299156, tensor(0.9170)]
OrderedDict([('data/mldoc/zh-1/models/sp30k/lstm_nl4-1cyc-sl2.m',
              0.9169999957084656)])
data/mldoc/zh-1/models/sp30k/lstm_nl4-1cyc-sl2.m: 0.9169999957084656
```
### 0.4
```bash
 python -m ulmfit eval --glob="mldoc/zh-1/models/sp30k/lstm_nl4.m" --name nl4-1cyc-sl4  --num-cls-epochs=8 --bs=18 --lr_sched=1cycle --label-smoothing-eps=0.4
Processing data/mldoc/zh-1/models/sp30k/lstm_nl4.m
zh-1
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp30k/lstm_nl4-1cyc-sl4.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/zh.dev.csv
Data lm, trn: 13500, val: 1500
Data cls, trn: 1000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁的', '▁。', '▁年', '▁、', '▁在', '▁一', '▁中', '▁人', '▁是']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp30k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp30k/lstm_nl4-1cyc-sl4.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.251581    1.183341    0.898000
2         1.214358    1.201266    0.834000
3         1.190343    1.165525    0.919000
4         1.168018    1.172510    0.903000
5         1.149965    1.161660    0.914000
6         1.140140    1.161689    0.915000
7         1.135877    1.159853    0.912000
8         1.134425    1.160039    0.911000
Total time: 06:34
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp30k/lstm_nl4-1cyc-sl4.m
Loss and accuracy using (cls_best): [0.64041936, tensor(0.9195)]
OrderedDict([('data/mldoc/zh-1/models/sp30k/lstm_nl4-1cyc-sl4.m',
              0.9194999933242798)])
data/mldoc/zh-1/models/sp30k/lstm_nl4-1cyc-sl4.m: 0.9194999933242798
```