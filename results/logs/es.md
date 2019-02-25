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



## ES optimization



### Smaler vocab 15k
#### LM
```
python -m ulmfit lm --dataset-path data/wiki-m/es-100 --cuda-id=0 --tokenizer='sp' --nl 4 --name 'nl4' --max-vocab 15000 \                                 ✘ 1
--lang es --qrnn=False - train 10 --bs=50 --drop_mult=0
Max vocab: 15000
Cache dir: data/wiki-m/es-100/models/sp15k
Model dir: data/wiki-m/es-100/models/sp15k/lstm_nl4.m
Tokenized data loaded
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁la', '▁.', 's', '▁en', '▁el', '▁y', '▁a', '▁que']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Training lm from random weights
epoch     train_loss  valid_loss  accuracy
1         2.961702    3.187043    0.403149
Better model found at epoch 1 with val_loss value: 3.1870434284210205.
2         2.928991    3.170802    0.402026
Better model found at epoch 2 with val_loss value: 3.170802354812622.
3         2.931906    3.128328    0.407816
Better model found at epoch 3 with val_loss value: 3.128328323364258.
4         2.869332    3.072160    0.414345
Better model found at epoch 4 with val_loss value: 3.072160243988037.
5         2.803377    2.997071    0.424847
Better model found at epoch 5 with val_loss value: 2.997070550918579.
6         2.758087    2.927369    0.432256
Better model found at epoch 6 with val_loss value: 2.927368640899658.
7         2.657733    2.825029    0.446440
Better model found at epoch 7 with val_loss value: 2.8250293731689453.
8         2.563273    2.728652    0.459271
Better model found at epoch 8 with val_loss value: 2.7286524772644043.
9         2.475741    2.654844    0.470864
Better model found at epoch 9 with val_loss value: 2.654844045639038.
10        2.428898    2.634355    0.474821
Better model found at epoch 10 with val_loss value: 2.634355306625366.
Total time: 17:53:59
data/wiki-m/es-100/models/sp15k
Saving info data/wiki-m/es-100/models/sp15k/lstm_nl4.m/info.json
```

#### MLDoc
```
 python -m ulmfit cls --dataset-path data/mldoc/es-1  --base-lm-path data/wiki-m/es-100/models/sp15k/lstm_nl4.m  --lang=es --name 'nl4' --cuda-id=0 - train 20 --bs 20 --num-cls-epochs=8
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp15k/lstm_nl4.m
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/es.dev.csv
Running tokenization...
Saving tokenized: cls.trn 13013, cls.val 1445
Running tokenization...
Saving tokenized: cls.trn 1000, cls.val 1000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁la', '▁.', 's', '▁en', '▁el', '▁y', '▁a', '▁que']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
Training lm from:  [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/wiki-m/es-100/models/sp15k/lstm_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/wiki-m/es-100/models/sp15k/lstm_nl4.m/../itos')]
epoch     train_loss  valid_loss  accuracy
1         2.355042    1.850417    0.589128
Better model found at epoch 1 with val_loss value: 1.850416898727417.
Total time: 03:25
epoch     train_loss  valid_loss  accuracy
1         2.087364    1.677666    0.619909
Better model found at epoch 1 with val_loss value: 1.6776657104492188.
2         1.880730    1.522996    0.648134
Better model found at epoch 2 with val_loss value: 1.5229955911636353.
3         1.767530    1.403644    0.668117
Better model found at epoch 3 with val_loss value: 1.4036436080932617.
4         1.659950    1.309353    0.684900
Better model found at epoch 4 with val_loss value: 1.3093526363372803.
5         1.546585    1.232220    0.699358
Better model found at epoch 5 with val_loss value: 1.2322196960449219.
6         1.592862    1.161846    0.713034
Better model found at epoch 6 with val_loss value: 1.1618456840515137.
7         1.444965    1.098108    0.726811
Better model found at epoch 7 with val_loss value: 1.0981075763702393.
8         1.340874    1.029193    0.741337
Better model found at epoch 8 with val_loss value: 1.0291931629180908.
9         1.351407    0.974317    0.753408
Better model found at epoch 9 with val_loss value: 0.9743167757987976.
10        1.231713    0.915328    0.767088
Better model found at epoch 10 with val_loss value: 0.9153280854225159.
11        1.151926    0.852391    0.782414
Better model found at epoch 11 with val_loss value: 0.852391242980957.
12        1.163565    0.794699    0.797228
Better model found at epoch 12 with val_loss value: 0.7946987152099609.
13        1.054929    0.743652    0.810518
Better model found at epoch 13 with val_loss value: 0.74365234375.
14        0.974651    0.695024    0.823344
Better model found at epoch 14 with val_loss value: 0.6950243711471558.
15        0.869718    0.651691    0.834510
Better model found at epoch 15 with val_loss value: 0.6516908407211304.
16        0.889763    0.615112    0.844947
Better model found at epoch 16 with val_loss value: 0.6151121258735657.
17        0.843503    0.590130    0.851694
Better model found at epoch 17 with val_loss value: 0.5901297926902771.
18        0.752870    0.575217    0.855496
Better model found at epoch 18 with val_loss value: 0.5752172470092773.
19        0.807087    0.567187    0.857605
Better model found at epoch 19 with val_loss value: 0.5671872496604919.
20        0.784531    0.566082    0.857827
Better model found at epoch 20 with val_loss value: 0.5660821199417114.
Total time: 1:26:48
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp15k/lstm_nl4.m/info.json
Starting classifier training
epoch     train_loss  valid_loss  accuracy
1         0.616963    0.275447    0.923000
Better model found at epoch 1 with val_loss value: 0.27544698119163513.
Total time: 00:24
epoch     train_loss  valid_loss  accuracy
1         0.425890    0.201535    0.932000
Better model found at epoch 1 with val_loss value: 0.2015346735715866.
Total time: 00:27
epoch     train_loss  valid_loss  accuracy
1         0.265563    0.186434    0.951000
Better model found at epoch 1 with val_loss value: 0.18643426895141602.
Total time: 00:32
epoch     train_loss  valid_loss  accuracy
1         0.162097    0.180026    0.955000
Better model found at epoch 1 with val_loss value: 0.1800260841846466.
2         0.161748    0.187014    0.957000
3         0.142789    0.166486    0.961000
Better model found at epoch 3 with val_loss value: 0.1664857715368271.
4         0.105920    0.173207    0.963000
5         0.078164    0.184849    0.962000
6         0.070540    0.189451    0.962000
7         0.051490    0.208019    0.959000
8         0.047614    0.193699    0.962000
Total time: 05:20
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp15k/lstm_nl4.m
Loss and accuracy using (cls_best): [0.1623594, tensor(0.9538)]
0.16235940158367157
0.9537500143051147
```

### Larger dropout - no luck
```
python -m ulmfit cls --dataset-path data/mldoc/es-1  --base-lm-path data/wiki-m/es-100/models/sp30k/lstm_nl4.m  --lang=es --name 'nl4-drop' --cuda-id=0 - train 0 --bs 20 --num-cls-epochs=8 --drop-mul-lm=0.5 --drop-mul-cls=0.8
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp30k/lstm_nl4-drop.m
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/es.dev.csv
Running tokenization...
Saving tokenized: cls.trn 13013, cls.val 1445
Running tokenization...
Saving tokenized: cls.trn 1000, cls.val 1000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁la', '▁.', '▁en', '▁el', '▁y', 's', '▁a', '▁que']
Starting classifier training
epoch     train_loss  valid_loss  accuracy
1         1.310594    0.903076    0.724000
Better model found at epoch 1 with val_loss value: 0.9030755758285522.
Total time: 00:22
epoch     train_loss  valid_loss  accuracy
1         1.209348    0.714140    0.756000
Better model found at epoch 1 with val_loss value: 0.714139997959137.
Total time: 00:23
epoch     train_loss  valid_loss  accuracy
1         1.122636    0.627526    0.797000
Better model found at epoch 1 with val_loss value: 0.6275263428688049.
Total time: 00:29
epoch     train_loss  valid_loss  accuracy
1         1.102433    0.593338    0.801000
Better model found at epoch 1 with val_loss value: 0.5933384895324707.
2         1.096480    0.543266    0.818000
Better model found at epoch 2 with val_loss value: 0.5432664155960083.
3         1.082919    0.501089    0.837000
Better model found at epoch 3 with val_loss value: 0.5010889172554016.
4         1.069694    0.518807    0.812000
5         1.040208    0.508399    0.825000
6         1.032841    0.512187    0.838000
7         1.031225    0.504557    0.825000
8         1.016486    0.502335    0.837000
Total time: 04:56
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp30k/lstm_nl4-drop.m
Loss and accuracy using (cls_best): [0.52473265, tensor(0.8160)]
0.5247326493263245
0.8159999847412109
```
