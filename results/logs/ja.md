##
```
python -m ulmfit lm --dataset-path data/wiki/ja-100 --cuda-id=0 --tokenizer='sp' --nl 4 --name 'nl4' --max-vocab 30000 \
--lang ja --qrnn=False - train 10 --bs=50 --drop_mult=0
Max vocab: 30000
Cache dir: data/wiki/ja-100/models/sp30k
Model dir: data/wiki/ja-100/models/sp30k/lstm_nl4.m
Running tokenization
Wiki text was split to 98375 articles
Wiki text was split to 138 articles
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁、', '▁の', '▁。', '▁に', '▁を', '▁は', '▁年', '▁が', '▁)', '▁(']
Training args:  {'tie_weights': True, 'clip': 0.12, 'bptt': 70, 'pretrained_fnames': None, 'pretrained_model': None, 'drop_mult': 0} dps:  [0.25 0.1  0.2  0.02 0.15]
Training lm from random weights
epoch     train_loss  valid_loss  accuracy
1         3.211025    3.328014    0.396197
2         3.119410    3.286294    0.395946
3         3.042064    3.247161    0.403915
4         3.023840    3.161323    0.413816
5         2.944752    3.102044    0.423163
6         2.907167    3.015610    0.434095
7         2.796073    2.927566    0.447088
8         2.715568    2.828766    0.461556
9         2.717255    2.747889    0.473289
10        2.619846    2.731164    0.477403
data/wiki/ja-100/models/sp30k
Saving info data/wiki/ja-100/models/sp30k/lstm_nl4.m/info.json
```

## MLDoc
MultiCCA 85.35%, ULMFiT 89.20%
```
python -m ulmfit cls --dataset-path data/mldoc/ja-1  --base-lm-path data/wiki/ja-100/models/sp30k/lstm_nl4.m  --lang=ja --name 'nl4' --cuda-id=1 - train 20 --bs 40 --num-cls-epochs=8
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ja-1/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ja-1/models/sp30k/lstm_nl4.m
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ja-1/ja.dev.csv
Running tokenization...
Saving tokenized: cls.trn 13500, cls.val 1500
Running tokenization...
Saving tokenized: cls.trn 1000, cls.val 1000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁、', '▁の', '▁。', '▁に', '▁を', '▁は', '▁年', '▁が', '▁)', '▁(']
Training args:  {'tie_weights': True, 'clip': 0.12, 'bptt': 70, 'pretrained_fnames': [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/wiki/ja-100/models/sp30k/lstm_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/wiki/ja-100/models/sp30k/lstm_nl4.m/../itos')], 'pretrained_model': None, 'drop_mult': 0.3} dps:  [0.25 0.1  0.2  0.02 0.15]
Unknown tokens 0, first 100: []
Training lm from:  [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/wiki/ja-100/models/sp30k/lstm_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/wiki/ja-100/models/sp30k/lstm_nl4.m/../itos')]
epoch     train_loss  valid_loss  accuracy
1         2.828645    2.386208    0.518716
epoch     train_loss  valid_loss  accuracy
1         2.462274    2.191761    0.549783
2         2.229925    1.982564    0.586238
3         2.043816    1.805435    0.616238
4         1.885779    1.674736    0.637964
5         1.773445    1.575366    0.653925
6         1.713029    1.490263    0.667570
7         1.660558    1.419641    0.680072
8         1.579792    1.357093    0.690826
9         1.459628    1.298609    0.701452
10        1.433604    1.251296    0.710232
11        1.439143    1.202794    0.719104
12        1.399083    1.158469    0.728430
13        1.310390    1.120877    0.736382
14        1.322389    1.085479    0.744013
15        1.272924    1.056051    0.750401
16        1.235312    1.034233    0.755225
17        1.227864    1.016682    0.759288
18        1.209589    1.007038    0.761234
19        1.173158    1.001694    0.762281
20        1.189994    1.000854    0.762526
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ja-1/models/sp30k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ja-1/models/sp30k/lstm_nl4.m/info.json
Starting classifier training
epoch     train_loss  valid_loss  accuracy
1         0.745803    0.554439    0.819000
epoch     train_loss  valid_loss  accuracy
1         0.620647    0.392026    0.856000
epoch     train_loss  valid_loss  accuracy
1         0.489173    0.369560    0.869000
epoch     train_loss  valid_loss  accuracy
1         0.406491    0.365988    0.872000
2         0.392645    0.351823    0.876000
3         0.386403    0.331737    0.880000
4         0.361338    0.333245    0.882000
5         0.319456    0.347253    0.879000
6         0.295419    0.350348    0.885000
7         0.286144    0.348592    0.879000
8         0.278896    0.358145    0.877000
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ja-1/models/sp30k/lstm_nl4.m
Loss and accuracy using (cls_best): [0.29789856, tensor(0.8920)]
```

### JA on 100 elements
```
python -m ulmfit cls --dataset-path data/mldoc/ja-1  --base-lm-path data/wiki/ja-100/models/sp30k/lstm_nl4.m  --lang=ja --name 'nl4-100' --cuda-id=1 - train 20 --bs 40 --num-cls-epochs=8 --limit=100
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ja-1/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ja-1/models/sp30k/lstm_nl4-100.m
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ja-1/ja.dev.csv
Tokenized data loaded, lm.trn 13500, lm.val 1500
Limiting data set to: 100
Running tokenization...
Saving tokenized: cls.trn 100, cls.val 100
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁、', '▁の', '▁。', '▁に', '▁を', '▁は', '▁年', '▁が', '▁)', '▁(']
Training args:  {'tie_weights': True, 'clip': 0.12, 'bptt': 70, 'pretrained_fnames': [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/wiki/ja-100/models/sp30k/lstm_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/wiki/ja-100/models/sp30k/lstm_nl4.m/../itos')], 'pretrained_model': None, 'drop_mult': 0.3} dps:  [0.25 0.1  0.2  0.02 0.15]
Unknown tokens 0, first 100: []
Training lm from:  [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/wiki/ja-100/models/sp30k/lstm_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/wiki/ja-100/models/sp30k/lstm_nl4.m/../itos')]
epoch     train_loss  valid_loss  accuracy
1         2.837937    2.387255    0.518590
epoch     train_loss  valid_loss  accuracy
1         2.466900    2.193583    0.549492
2         2.232762    1.983981    0.586658
3         2.026505    1.810167    0.615649
4         1.918111    1.679784    0.636613
5         1.748909    1.577095    0.653108
6         1.708709    1.491436    0.667657
7         1.640415    1.420449    0.679619
8         1.577434    1.359511    0.690194
9         1.551961    1.302819    0.700306
10        1.475623    1.252393    0.710039
11        1.435565    1.208159    0.718740
12        1.354910    1.161781    0.727927
13        1.351157    1.123244    0.736009
14        1.299070    1.086383    0.743896
15        1.258739    1.055745    0.750383
16        1.210775    1.035209    0.754965
17        1.228421    1.018373    0.758963
18        1.179444    1.007714    0.761158
19        1.197443    1.003041    0.762068
20        1.163223    1.001939    0.762211
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ja-1/models/sp30k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ja-1/models/sp30k/lstm_nl4-100.m/info.json
Starting classifier training
epoch     train_loss  valid_loss  accuracy
1         1.269222    1.360420    0.340000
epoch     train_loss  valid_loss  accuracy
1         0.969350    1.314497    0.400000
epoch     train_loss  valid_loss  accuracy
1         0.832396    1.263416    0.550000
epoch     train_loss  valid_loss  accuracy
1         0.780991    1.225439    0.600000
2         0.765755    1.183010    0.600000
3         0.749420    1.139053    0.600000
4         0.731800    1.093319    0.610000
5         0.711152    1.054695    0.610000
6         0.694611    1.029465    0.580000
7         0.680276    1.004366    0.580000
8         0.668421    0.984848    0.590000
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ja-1/models/sp30k/lstm_nl4-100.m
Loss and accuracy using (cls_best): [0.81621724, tensor(0.7437)]
```