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