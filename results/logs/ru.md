# RU
## SP15k nl4 QRNN
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
```bash
python -m ulmfit cls --dataset-path data/mldoc/ru-1  --base-lm-path data/wiki/ru-100/models/sp30k/lstm_nl4.m  --lang=ru --name 'nl4-100' --cuda-id=1 - train 20 --bs 40 --num-cls-epochs=2

```

## SP25k qrnn
### LM
```bash
 python -m ulmfit lm --dataset-path data/wiki/${LANG}-100 --tokenizer='sp' --nl 4 --name
'nl4' --max-vocab 25000 --lang ${LANG} --qrnn=True - train 10 --bs=50 --drop_mult=0  --label-smoothing-eps=0.10 --tokenizer='sp
Max vocab: 25000
Cache dir: data/wiki/ru-100/models/sp25k
Model dir: data/wiki/ru-100/models/sp25k/qrnn_nl4.m
Wiki text was split to 193047 articles
Wiki text was split to 460 articles
Data lm, trn: 193047, val: 460
Size of vocabulary: 25000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁.', '▁в', 'а', 'и', 'е', '▁и', 'й', 'х', '▁на']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Training lm from random weights
epoch     train_loss  valid_loss  accuracy
1         4.154972    4.198218    0.447508
2         4.030367    4.159642    0.449420
3         4.138530    4.146010    0.451526
4         3.997120    4.097048    0.457177
5         3.999151    4.036350    0.465117
6         3.935380    3.955517    0.476446
7         3.912357    3.875987    0.487591
8         3.785693    3.789099    0.501560
9         3.743162    3.725730    0.512294
10        3.690226    3.706929    0.516769
Total time: 12:10:03
data/wiki/ru-100/models/sp25k
```

```bash
python -m ulmfit cls --dataset-path data/mldoc/${LANG}-1 --base-lm-path data/wiki/${LANG
}-100/models/sp25k/qrnn_${NAME}.m  --lang=${LANG} --name ${NAME} - train 20 --bs 18 --num-cls-epochs=4 --lr_sched=1cycle --label-smoothing-eps=0.1

Max vocab: 25000
Cache dir: /home/n-waves/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp25k
Model dir: /home/n-waves/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp25k/qrnn_nl4.m
Loading validation /home/n-waves/workspace/ulmfit-multilingual/data/mldoc/ru-1/ru.dev.csv
Running tokenization lm...
Data lm, trn: 9195, val: 1021
Running tokenization cls...
Data cls, trn: 1000, val: 1000
Running tokenization tst...
Data tst, trn: 1000, val: 4000
Size of vocabulary: 25000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁.', '▁в', 'а', 'и', 'е', '▁и', 'й', 'х', '▁на']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
Training lm from:  [PosixPath('/home/n-waves/workspace/ulmfit-multilingual/data/wiki/ru-100/models/sp25k/qrnn_nl4.m/lm_best'), PosixPath('/home/n-waves/workspace/ulmfit-multilingual/data/wiki/ru-100/models/sp25k/qrnn_nl4.m/../itos')]
epoch     train_loss  valid_loss  accuracy
1         4.626971    3.868075    0.474742
Total time: 01:58
epoch     train_loss  valid_loss  accuracy
1         3.821786    3.625366    0.519506
2         3.570115    3.379288    0.566803
3         3.517294    3.179166    0.599955
4         3.160131    3.028985    0.626484
5         3.135806    2.923198    0.644557
6         3.055160    2.840300    0.659376
7         3.005086    2.770163    0.672080
8         2.811366    2.708846    0.684065
9         2.818394    2.658951    0.694358
10        2.881018    2.605373    0.705269
11        2.793422    2.560091    0.715893
12        2.708385    2.516373    0.725908
13        2.690258    2.471159    0.735673
14        2.748342    2.436113    0.744533
15        2.601220    2.394404    0.754131
16        2.616882    2.372301    0.760451
17        2.602902    2.349164    0.766014
18        2.560349    2.336217    0.769222
19        2.549936    2.332076    0.770150
20        2.546798    2.331103    0.770472
Total time: 53:22
/home/n-waves/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp25k
Saving info /home/n-waves/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp25k/qrnn_nl4.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.043533    0.961182    0.731000
2         0.859086    0.837210    0.824000
3         0.735276    0.724173    0.871000
4         0.612012    0.711034    0.857000
Total time: 01:15
Saving models at /home/n-waves/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp25k/qrnn_nl4.m
Loss and accuracy using (cls_best): [0.3957597, tensor(0.8720)]
0.3957597017288208
0.871999979019165
```

## VF60k QRNN
### LM

### MLDoc
```bash
Max vocab: 60000
Cache dir: /home/n-waves/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/vf60k
Model dir: /home/n-waves/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/vf60k/qrnn_nl4.m
Loading validation /home/n-waves/workspace/ulmfit-multilingual/data/mldoc/ru-1/ru.dev.csv
Running tokenization lm...
Data lm, trn: 9195, val: 1021
Running tokenization cls...
Data cls, trn: 1000, val: 1000
Running tokenization tst...
Data tst, trn: 1000, val: 4000
Size of vocabulary: 55567
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', ',', '.', '-', 'в', ')', '(', 'на', "&'", 'и', 'по', 'с', 'the']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 34364, first 100: ['рейтер', '941', '8520', '095', 'said', '\x7f', 'доллару', 'янв', 'погашение', '3272', 'reuter', 'xd0', 'фев', '509410', 'уставный', 'which', 'percent', 'объективность', 'торгах', 'купона', 'million', 'its', 'июл', '044', 'алма-атинское', 'валютной', 'триллиона', 'межбанковской', 'would', 'авг', 'government', 'котировки', 'балансовая', 'ртс', 'выплата', 'прц', '8832', 'yeltsin', '983', 'средневзвешенная', '961', 'president', 'дек', 'minister', '2264', 'нацбанка', 'цбр', 'июн', 'newsroom', 'ммвб', 'гособлигаций', 'стр.1', 'also', 'foreign', 'офз', 'заявленный', 'шестимесячных', 'дисконтных', '-сказал', 'предыдущему', 'тбилисское', 'размещенный', 'told', 'riga', 'лари', 'стр.2', 'kroons', 'окт', 'сиданко', '--московское', 'adr', 'мосэнерго', 'shares', 'пресс-релизе', 'дилеры', 'триллионов', 'акциям', 'billion', 'демченко', 'тнк', 'litas', 'lats', 'дилеров', '--алма-атинское', 'щелкните', 'tuesday', 'зинец', 'friday', 'умвб', 'thursday', 'онэксим', 'трейдеры', 'nato', 'feb', 'дивиденды', 'former', 'could', 'нацбанк', 'стр.6', 'economic']
Bptt 70
Training lm from:  [PosixPath('/home/n-waves/workspace/ulmfit-multilingual/data/wiki/ru-100/models/vf60k/qrnn_nl4.m/lm_best'), PosixPath('/home/n-waves/workspace/ulmfit-multilingual/data/wiki/ru-100/models/vf60k/qrnn_nl4.m/../itos')]
epoch     train_loss  valid_loss  accuracy
1         5.637876    4.853484    0.379844
Total time: 01:28
epoch     train_loss  valid_loss  accuracy
1         4.906714    4.683807    0.405109
2         4.850066    4.490903    0.434562
3         4.591409    4.284740    0.464436
4         4.379681    4.103634    0.490118
5         4.079576    3.954377    0.511206
6         4.199800    3.811692    0.531036
7         4.004812    3.694871    0.548372
8         3.995378    3.584868    0.567285
9         3.884090    3.499729    0.583162
10        3.897333    3.416602    0.598120
11        3.726276    3.338907    0.613920
12        3.690300    3.263694    0.629643
13        3.614015    3.192474    0.646335
14        3.530548    3.136064    0.659729
15        3.451486    3.100320    0.668686
16        3.444497    3.058001    0.678824
17        3.407755    3.024943    0.686764
18        3.383617    3.008939    0.690451
19        3.342304    2.999911    0.692378
20        3.339514    2.998623    0.692671
Total time: 36:01
/home/n-waves/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/vf60k
Saving info /home/n-waves/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/vf60k/qrnn_nl4.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.946690    0.855268    0.805000
2         0.808650    0.750561    0.866000
3         0.701750    0.712251    0.884000
4         0.596392    0.687266    0.884000
Total time: 00:44
Saving models at /home/n-waves/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/vf60k/qrnn_nl4.m
Loss and accuracy using (cls_best): [0.37472174, tensor(0.8802)]
0.3747217357158661
0.8802499771118164
```


## SP30k LSTM nl4 
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