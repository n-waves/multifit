# RU
## SP15k nl4
```
Training lm from random weights
epoch     train_loss  valid_loss  accuracy
1         3.053061    3.070487    0.450466
2         2.874137    2.999093    0.455027
3         2.864496    2.969308    0.458116
4         2.890568    2.903564    0.466970
5         2.746530    2.839789    0.474205
6         2.683900    2.750476    0.486806
7         2.674458    2.658535    0.499701
8         2.595780    2.573735    0.512515
9         2.530827    2.512999    0.522372
10        2.505664    2.491850    0.526431
Total time: 10:43:03
data/wiki/ru-100/models/sp15k
Saving info data/wiki/ru-100/models/sp15k/qrnn_nl4.m/info.json
```


## SP30k nl4 
### LM
```
python -m ulmfit lm --dataset-path data/wiki/ru-100 --cuda-id=0 --tokenizer='sp' --nl 4 --name 'nl4' --max-vocab 30000 --lang ru --qrnn=False - train 10 --bs=50 --drop_mult=0
Size of vocabulary: 30000                                                                                                                                                                  [39/805]
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁.', '▁в', 'а', '▁и', 'е', 'и', 'й', '▁на', 'х']
Training args:  {'tie_weights': True, 'clip': 0.12, 'bptt': 70, 'pretrained_fnames': None, 'pretrained_model': None, 'drop_mult': 0} dps:  [0.25 0.1  0.2  0.02 0.15]
Training lm from random weights
epoch     train_loss  valid_loss  accuracy
1         3.200520    3.295865    0.436852
2         3.027569    3.168700    0.445551
3         3.007320    3.132495    0.450450
4         2.940000    3.041745    0.459344
5         2.876227    2.952338    0.469182
6         2.742553    2.860888    0.480943
7         2.684717    2.769994    0.492934
8         2.569419    2.669971    0.507300
9         2.525698    2.604086    0.516840
10        2.495174    2.591011    0.519415
data/wiki/ru-100/models/sp30k
Saving info data/wiki/ru-100/models/sp30k/lstm_nl4.m/info.json
```
### MLDoc - bsp
MultiCCA: 85.65% ulmfit: 87.27%
```
python -m ulmfit cls --dataset-path data/mldoc/ru-1  --base-lm-path data/wiki/ru-100/models/sp30k/lstm_nl4.m  --lang=ru --name 'nl4-100' --cuda-id=1 - train 20 --bs 40 --num-cls-epochs=2
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp30k/lstm_nl4.m
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1/ru.dev.csv
Running tokenization...
Saving tokenized: cls.trn 9195, cls.val 1021
Running tokenization...
Saving tokenized: cls.trn 1000, cls.val 1000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁.', '▁в', 'а', '▁и', 'е', 'и', 'й', '▁на', 'х']
Training args:  {'tie_weights': True, 'clip': 0.12, 'bptt': 70, 'pretrained_fnames': [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/wiki/ru-100/models/sp30k/lstm_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/wiki/ru-100/models/sp30k/lstm_nl4.m/../itos')], 'pretrained_model': None, 'drop_mult': 0.3} dps:  [0.25 0.1  0.2  0.02 0.15]
Unknown tokens 0, first 100: []
Training lm from:  [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/wiki/ru-100/models/sp30k/lstm_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/wiki/ru-100/models/sp30k/lstm_nl4.m/../itos')]
epoch     train_loss  valid_loss  accuracy
1         2.764138    2.289755    0.552181
epoch     train_loss  valid_loss  accuracy
1         2.414295    2.161708    0.572407
2         2.310551    2.013092    0.596075
3         2.124479    1.864450    0.620103
4         1.970015    1.723395    0.642392
5         1.883664    1.623308    0.658949
6         1.793856    1.513542    0.677954
7         1.625767    1.424582    0.693092
8         1.677054    1.335406    0.709802
9         1.578936    1.264322    0.723626
10        1.523383    1.194463    0.737942
11        1.436643    1.129712    0.750586
12        1.351507    1.072792    0.762524
13        1.357552    1.020739    0.773266
14        1.310516    0.975852    0.783653
15        1.216484    0.940323    0.791262
16        1.187942    0.909915    0.797675
17        1.141316    0.885367    0.803305
18        1.114629    0.871992    0.805929
19        1.075366    0.867010    0.807009
20        1.166387    0.865594    0.807241
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp30k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp30k/lstm_nl4.m/info.json
Starting classifier training
epoch     train_loss  valid_loss  accuracy
1         0.831180    0.610087    0.787000
epoch     train_loss  valid_loss  accuracy
1         0.678307    0.435860    0.856000
epoch     train_loss  valid_loss  accuracy
1         0.547668    0.399889    0.870000
epoch     train_loss  valid_loss  accuracy
1         0.445839    0.396535    0.869000
2         0.417901    0.369961    0.882000
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp30k/lstm_nl4.m
Loss and accuracy using (cls_best): [0.38499942, tensor(0.8727)]
```

### MLDoc run 2x sp
```
python -m ulmfit cls --dataset-path data/mldoc/ru-1  --base-lm-path data/wiki/ru-100/models/sp30k/lstm_nl4.m  --lang=ru --name 'nl4' --cuda-id=1 - train 20 --bs 40 --num-cls-epochs=2
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp30k/lstm_nl4.m
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1/ru.dev.csv
Running tokenization...
Saving tokenized: cls.trn 9195, cls.val 1021
Running tokenization...
Saving tokenized: cls.trn 1000, cls.val 1000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁.', '▁в', 'а', '▁и', 'е', 'и', 'й', '▁на', 'х']
Training args:  {'tie_weights': True, 'clip': 0.12, 'bptt': 70, 'pretrained_fnames': [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/wiki/ru-100/models/sp30k/lstm_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/wiki/ru-100/models/sp30k/lstm_nl4.m/../itos')], 'pretrained_model': None, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  [0.25 0.1  0.2  0.02 0.15]
Unknown tokens 0, first 100: []
Training lm from:  [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/wiki/ru-100/models/sp30k/lstm_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/wiki/ru-100/models/sp30k/lstm_nl4.m/../itos')]
epoch     train_loss  valid_loss  accuracy
1         2.662225    2.284158    0.552927
epoch     train_loss  valid_loss  accuracy
1         2.436114    2.151187    0.574219
2         2.260576    2.012279    0.595820
3         2.067110    1.862512    0.620246
4         2.000703    1.729883    0.641713
5         1.860899    1.609955    0.661346
6         1.751010    1.522195    0.676297
7         1.705993    1.420628    0.694044
8         1.592143    1.338552    0.708978
9         1.524927    1.270614    0.722596
10        1.475408    1.198585    0.736638
11        1.438226    1.134858    0.749314
12        1.408821    1.076875    0.761448
13        1.345137    1.020660    0.773432
14        1.321399    0.978076    0.783070
15        1.235357    0.936674    0.791642
16        1.204204    0.906822    0.798548
17        1.198709    0.884949    0.803528
18        1.176732    0.874523    0.805585
19        1.111195    0.871806    0.806239
20        1.031497    0.869280    0.806826
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp30k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp30k/lstm_nl4.m/info.json
Starting classifier training
epoch     train_loss  valid_loss  accuracy
1         0.834704    0.615589    0.786000
epoch     train_loss  valid_loss  accuracy
1         0.679823    0.418461    0.851000
epoch     train_loss  valid_loss  accuracy
1         0.555612    0.426877    0.861000
epoch     train_loss  valid_loss  accuracy
1         0.468084    0.391777    0.873000
2         0.434714    0.388670    0.882000
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp30k/lstm_nl4.m
Loss and accuracy using (cls_best): [0.3987146, tensor(0.8680)]
0.3987146019935608
0.8679999709129333
```

```
Second execution
epoch     train_loss  valid_loss  accuracy
1         2.749340    2.284773    0.552775
epoch     train_loss  valid_loss  accuracy
1         2.418463    2.157943    0.572302
```