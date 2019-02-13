# ES

## SP30k LSTM nl 4
### LM
````
python -m ulmfit lm --dataset-path data/wiki/es-100 --cuda-id=0 --tokenizer='sp' --nl 4 --name 'nl4' --max-vocab 30000 --lang es --qrnn=False - train 10 --bs=50 --drop_mult=0
Running tokenization
Wiki text was split to 96224 articles
Wiki text was split to 105 articles
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁la', '▁.', '▁en', '▁el', '▁y', 's', '▁a', '▁que']
Training args:  {'tie_weights': True, 'clip': 0.12, 'bptt': 70, 'pretrained_fnames': None, 'pretrained_model': None, 'drop_mult': 0} dps:  [0.25 0.1  0.2  0.02 0.15]
Training lm from random weights
epoch     train_loss  valid_loss  accuracy
1         3.269541    3.451855    0.387471
2         3.161740    3.423016    0.386158
3         3.187431    3.419638    0.388626
4         3.115763    3.357066    0.393877
5         2.996527    3.291787    0.402488
6         3.021759    3.202183    0.410873
7         2.998267    3.104373    0.422624
8         2.827225    3.006537    0.436010
9         2.784576    2.937735    0.446654
10        2.789913    2.918509    0.450055
data/wiki/es-100/models/sp30k
Saving info data/wiki/es-100/models/sp30k/lstm_nl4.m/info.json
````

### MLDoc

```
python -m ulmfit cls --dataset-path data/mldoc/es-1  --base-lm-path data/wiki/es-100/models/sp30k/lstm_nl4.m  --lang=es --name 'nl4' --cuda-id=1 - train 20 --bs 40 --num-cls-epochs=2
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp30k/lstm_nl4.m
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/es.dev.csv
Running tokenization...
Saving tokenized: cls.trn 13013, cls.val 1445
Running tokenization...
Saving tokenized: cls.trn 1000, cls.val 1000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁la', '▁.', '▁en', '▁el', '▁y', 's', '▁a', '▁que']
Training args:  {'tie_weights': True, 'clip': 0.12, 'bptt': 70, 'pretrained_fnames': [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/wiki/es-100/models/sp30k/lstm_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/wiki/es-100/models/sp30k/lstm_nl4.m/../itos')], 'pretrained_model': None, 'drop_mult': 0.3} dps:  [0.25 0.1  0.2  0.02 0.15]
Unknown tokens 0, first 100: []
Training lm from:  [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/wiki/es-100/models/sp30k/lstm_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/wiki/es-100/models/sp30k/lstm_nl4.m/../itos')]
epoch     train_loss  valid_loss  accuracy
1         2.805415    2.188974    0.537779
epoch     train_loss  valid_loss  accuracy
1         2.429727    1.989691    0.569048
2         2.218828    1.794969    0.603721
3         2.015097    1.644815    0.629609
4         1.877210    1.537773    0.646898
5         1.775648    1.450283    0.660861
6         1.749334    1.377085    0.672146
7         1.601073    1.311101    0.684400
8         1.564420    1.251074    0.694900
9         1.532728    1.197607    0.704779
10        1.391921    1.145408    0.716044
11        1.379958    1.093550    0.726937
12        1.324111    1.048308    0.735890
13        1.344113    1.007926    0.745691
14        1.243085    0.969521    0.754591
15        1.230809    0.937330    0.762675
16        1.162501    0.913408    0.768044
17        1.170092    0.894892    0.773239
18        1.110860    0.884449    0.775603
19        1.115907    0.880448    0.776671
20        1.083033    0.878421    0.776931
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp30k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp30k/lstm_nl4.m/info.json
Starting classifier training
epoch     train_loss  valid_loss  accuracy
1         0.621574    0.391042    0.856000
epoch     train_loss  valid_loss  accuracy
1         0.411668    0.215625    0.935000
epoch     train_loss  valid_loss  accuracy
1         0.340519    0.222422    0.935000
epoch     train_loss  valid_loss  accuracy
1         0.281729    0.192193    0.949000
2         0.262074    0.202975    0.945000
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp30k/lstm_nl4.m
Loss and accuracy using (cls_best): [0.1749019, tensor(0.9515)]
```