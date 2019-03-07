# DE
## SP15k LSTM nl4
```
$ python -m  ulmfit lm --dataset-path data/wiki/de-100 --cuda-id=0 --tokenizer='sp' --nl 4 --name 'nl4' --max-vocab 15000 --lang de --qrnn=False - train 10 --bs=100 --drop_mult=0

First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁.', '▁,', '▁der',
'▁die', 'en', '▁und', 's', '▁in', 'er', "▁&'"]
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Training lm from random weights
epoch     train_loss  valid_loss  accuracy
1         2.519809    2.600072    0.529963
2         2.436580    2.538897    0.534651
3         2.402220    2.510569    0.537314
4         2.305741    2.439347    0.546574
5         2.265683    2.376482    0.553794
6         2.210663    2.305362    0.562672
7         2.134196    2.230041    0.572958
8         2.085375    2.150917    0.584621
9         2.037781    2.097170    0.593747
10        1.986773    2.081469    0.595799
Total time: 19:18:33
data/wiki/de-100/models/sp15k
Saving info data/wiki/de-100/models/sp15k/lstm_nl4.m/info.jso
```
### MLDoc
```bash
LANG=de
python -m ulmfit cls --dataset-path data/mldoc-m/${LANG}-1  --base-lm-path data/wiki-m/${LANG}-100/models/sp15k/lstm_nl4.m  --lang=${LANG} --name 'nl4' - train 20 --bs 20 --num-cls-
epochs=8 --lr_sched=1cycle
Max vocab: 15000
Cache dir: /home/test/workspace/ulmfit-multilingual/data/mldoc-m/de-1/models/sp15k
Model dir: /home/test/workspace/ulmfit-multilingual/data/mldoc-m/de-1/models/sp15k/lstm_nl4.m
Loading validation /home/test/workspace/ulmfit-multilingual/data/mldoc-m/de-1/de.dev.csv
Running tokenization lm...
Data lm, trn: 13500, val: 1500
Running tokenization cls...
Data cls, trn: 1000, val: 1000
Running tokenization tst...
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁.', '▁,', '▁der', '▁die', 'en', '▁und', 's', '▁in', 'er', "▁&'"]
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
Training lm from:  [PosixPath('/home/test/workspace/ulmfit-multilingual/data/wiki-m/de-100/models/sp15k/lstm_nl4.m/lm_best'), PosixPath('/home/test/workspace/ulmfit-multilingual/data/wiki-m/de-100/models/sp15k/lstm_nl4.m/../itos')]
epoch     train_loss  valid_loss  accuracy
1         2.333600    2.005051    0.596875
Total time: 07:40
epoch     train_loss  valid_loss  accuracy
1         2.120653    1.886799    0.615784
2         1.980713    1.763139    0.636041
3         1.805195    1.655620    0.654068
4         1.729641    1.564017    0.668772
5         1.681813    1.491185    0.680613
6         1.682965    1.422562    0.692458
7         1.580731    1.357177    0.703143
8         1.506753    1.297219    0.714487
9         1.515824    1.235473    0.725413
10        1.427750    1.178680    0.737216
11        1.371839    1.118909    0.749590
12        1.342978    1.068754    0.760473
13        1.286842    1.011940    0.772384
14        1.254822    0.960727    0.784244
15        1.195136    0.919377    0.793910
16        1.118260    0.881799    0.802814
17        1.071546    0.855769    0.809040
18        1.079081    0.839280    0.812895
19        1.052724    0.831323    0.814723
20        1.024207    0.829737    0.815070
Total time: 3:08:58
/home/test/workspace/ulmfit-multilingual/data/mldoc-m/de-1/models/sp15k
Saving info /home/test/workspace/ulmfit-multilingual/data/mldoc-m/de-1/models/sp15k/lstm_nl4.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.539181    0.239851    0.938000
2         0.326801    0.374512    0.917000
3         0.225103    0.330872    0.945000
4         0.121660    0.444890    0.938000
5         0.078411    0.422513    0.948000
6         0.061354    0.509489    0.949000
7         0.029890    0.438118    0.949000
8         0.014213    0.441808    0.949000
Total time: 09:00
Saving models at /home/test/workspace/ulmfit-multilingual/data/mldoc-m/de-1/models/sp15k/lstm_nl4.m
Loss and accuracy using (cls_best): [0.3710725, tensor(0.9553)]

0.3710725009441376
0.9552500247955322
```


## VF60k LSTM nl 3
### LM 
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
### MLDocs
```
python -m ulmfit cls --dataset-path data/mldoc/de-1  --base-lm-path data/wiki/de-100/models/vf60k/lstm_nl3.m  --lang=de --name 'nl3' - train 20 --bs 40
Max vocab: 60000
Cache dir: data/mldoc/de-1/models/vf60k
Model dir: data/mldoc/de-1/models/vf60k/lstm_nl3.m
Loading validation data/mldoc/de-1/de.dev.csv
Running tokenization...
Saving tokenized: cls.trn 13500, cls.val 1500
Running tokenization...
Saving tokenized: cls.trn 1000, cls.val 1000
Size of vocabulary: 39171
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '.', 'der', ',', 'die', ')', '(', 'in', 'und', 'auf', 'von', 'den', 'im']
Training args:  {'tie_weights': True, 'clip': 0.12, 'bptt': 70, 'pretrained_fnames': [PosixPath('data/wiki/de-100/models/vf60k/lstm_nl3.m/lm_best'), PosixPath('data/wiki/de-100/models/vf60k/lstm_nl3.m/../itos')], 'pretrained_model': None, 'drop_mult': 0.3} dps:  [0.25 0.1  0.2  0.02 0.15]
Unknown tokens 20582, first 100: ['"', 'vh', 'ös', 'brs', '&', 'geg', 'lpo', 'vormonat', 'fgc', 'waigel', 'tcs', 'mic', 'bund-future', 'ajs', 'brn', 'dih', 'analysten', 'mrd', 'rpk', 'notierten', 'dividende', 'feb', 'aktienmarkt', 'rev', 'rentenmarkt', 'basispunkte', 'müßten', 'gewinnmitnahmen', 'aktienbörse', 'rußland', 'volkswirte', 'fls', 'steuerreform', 'kontrakte', 'kps', 'mge', 'zählern', 'vortagesschluß', 'umsätzen', 'prozent.', 'snb', 'dow-jones-index', 'reingewinn', 'notierungen', "\\'", 'gesamtmarkt', 'industrieproduktion', 'akr', 'kjf', '49-69-7565', 'abl', 'hoh', 'finanzdienst', 'atx', 'feinunze', 'zinserhöhung', 'zugelegt', 'netanjahu', 'verbraucherpreise', 'pence', 'ticks', 'arafat', 'kursgewinne', 'ker', 'aktienindex', 'rlb', 'smi', 'vorbörslich', 'dst', 'mkl', 'kontrakten', 'calls', 'veraenderung', 'gwa', 'gesamtjahr', 'auftragseingang', 'überschuß', 'erwarte', 'verlautete', 'eju', 'tms', 'jahresvergleich', 'vorjahreszeitraum', 'werden.', 'betriebsergebnis', 'rin', 'bobl-future', 'puts', 'fri', '4.50', 'schluß', 'ewu', 'standardwerte', 'jahresüberschuß', 'rechne', '49-69-756525', '16.00', 'peh', 'hmh', 'dtb']
Training lm from:  [PosixPath('data/wiki/de-100/models/vf60k/lstm_nl3.m/lm_best'), PosixPath('data/wiki/de-100/models/vf60k/lstm_nl3.m/../itos')]
epoch     train_loss  valid_loss  accuracy
1         3.532079    3.059487    0.465283
epoch     train_loss  valid_loss  accuracy
1         3.219148    2.945357    0.475736
2         3.014822    2.804256    0.494567
3         2.896143    2.652700    0.513166
4         2.756027    2.516747    0.528836
5         2.629735    2.383480    0.543956
6         2.515785    2.281831    0.556083
7         2.422463    2.178855    0.567950
8         2.351060    2.091266    0.579531
9         2.297676    2.017783    0.590206
10        2.205688    1.937085    0.601936
11        2.155664    1.871271    0.612579
12        2.065812    1.806647    0.623888
13        2.038635    1.748420    0.634389
14        1.957434    1.696571    0.643807
15        1.895242    1.653865    0.651743
16        1.910458    1.618776    0.658140
17        1.843909    1.598143    0.662129
18        1.837299    1.583182    0.664999
19        1.788718    1.573136    0.666785
20        1.780236    1.574308    0.666625
data/mldoc/de-1/models/vf60k
Saving info data/mldoc/de-1/models/vf60k/lstm_nl3.m/info.json
Starting classifier training
epoch     train_loss  valid_loss  accuracy
1         0.526303    0.328480    0.892000
epoch     train_loss  valid_loss  accuracy
1         0.346665    0.238605    0.920000
epoch     train_loss  valid_loss  accuracy
1         0.266841    0.285444    0.921000
epoch     train_loss  valid_loss  accuracy
1         0.175013    0.280545    0.921000
2         0.178333    0.286059    0.923000
Saving models at data/mldoc/de-1/models/vf60k/lstm_nl3.m
Loss and accuracy using (cls_best): [0.16954255, tensor(0.9475)]
OrderedDict([('data/mldoc/de-1/models/vf60k/lstm_nl3.m', 0.9474999904632568)])
```
MultiCCA: 93.7% , ulmfit: 94.74%
## SP30k LSTM nl 4
### LM
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

### MLDocs
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
Loss and accuracy using (cls_last): [0.16306259, tensor(0.9540)]
```
MultiCCA: 93.7% , ulmfit: 95.4%
 ```
 Max vocab: 30000
Cache dir: data/mldoc/de-1/models/sp30k
Model dir: data/mldoc/de-1/models/sp30k/lstm_nl4-2nd.m
Loading validation data/mldoc/de-1/de.dev.csv
Tokenized data loaded, lm.trn 13500, lm.val 1500
Tokenized data loaded, cls.trn 1000, cls.val 1000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁.', '▁,', '▁der', '▁die', '▁und', '▁in', 'en', "▁&'", 's', '-']
Training args:  {'tie_weights': True, 'clip': 0.12, 'bptt': 70, 'pretrained_fnames': [PosixPath('data/mldoc/de-1/models/sp30k/lstm_nl4.m/lm_best'), PosixPath('data/mldoc/de-1/models/sp30k/lstm_nl4.m/../itos')], 'pretrained_model': None, 'drop_mult': 0.3} dps:  [0.25 0.1  0.2  0.02 0.15]
Unknown tokens 0, first 100: []
data/mldoc/de-1/models/sp30k
Saving info data/mldoc/de-1/models/sp30k/lstm_nl4-2nd.m/info.json
Starting classifier training
epoch     train_loss  valid_loss  accuracy
1         0.464957    0.258905    0.928000
epoch     train_loss  valid_loss  accuracy
1         0.284900    0.243053    0.937000
epoch     train_loss  valid_loss  accuracy
1         0.298546    0.204188    0.948000
epoch     train_loss  valid_loss  accuracy
1         0.159097    0.199651    0.952000
2         0.112476    0.203827    0.953000
Saving models at data/mldoc/de-1/models/sp30k/lstm_nl4-2nd.m
Loss and accuracy using (cls_last): [0.1689675, tensor(0.9550)]
 ```
### examples limited to 100
 
#### 2x run
first run
```
python -m ulmfit cls --dataset-path data/mldoc/de-1  --base-lm-path data/mldoc/de-1/models/sp30k/lstm_nl4.m  --lang=de --name 'nl4-100-2x' --cuda-id=1 - train 0 --bs 40 --limit=100 --drop-mult-cls=0.3
Max vocab: 30000
Cache dir: data/mldoc/de-1/models/sp30k
Model dir: data/mldoc/de-1/models/sp30k/lstm_nl4-100-2x.m
Loading validation data/mldoc/de-1/de.dev.csv
Tokenized data loaded, lm.trn 13500, lm.val 1500
Limiting data set to: 100
Running tokenization...
Saving tokenized: cls.trn 100, cls.val 100
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁.', '▁,', '▁der', '▁die', '▁und', '▁in', 'en', "▁&'", 's', '-']
Training args:  {'tie_weights': True, 'clip': 0.12, 'bptt': 70, 'pretrained_fnames': [PosixPath('data/mldoc/de-1/models/sp30k/lstm_nl4.m/lm_best'), PosixPath('data/mldoc/de-1/models/sp30k/lstm_nl4.m/../itos')], 'pretrained_model': None, 'drop_mult': 0.3} dps:  [0.25 0.1  0.2  0.02 0.15]
Unknown tokens 0, first 100: []
data/mldoc/de-1/models/sp30k
Saving info data/mldoc/de-1/models/sp30k/lstm_nl4-100-2x.m/info.json
Starting classifier training
epoch     train_loss  valid_loss  accuracy
1         1.181207    1.315258    0.300000
epoch     train_loss  valid_loss  accuracy
1         0.749909    1.204297    0.660000
epoch     train_loss  valid_loss  accuracy
1         0.558658    1.083666    0.830000
epoch     train_loss  valid_loss  accuracy
1         0.486175    1.020435    0.850000
2         0.485117    0.958238    0.880000
Saving models at data/mldoc/de-1/models/sp30k/lstm_nl4-100-2x.m
 ..? ..
```
2nd run
```
python -m ulmfit cls --dataset-path data/mldoc/de-1  --base-lm-path data/mldoc/de-1/models/sp30k/lstm_nl4.m  --lang=de --name 'nl4-100-2x' --cuda-id=1 - train 0 --bs 40 --limit=100 
Max vocab: 30000
Cache dir: data/mldoc/de-1/models/sp30k
Model dir: data/mldoc/de-1/models/sp30k/lstm_nl4-100-2x.m
Loading validation data/mldoc/de-1/de.dev.csv
Tokenized data loaded, lm.trn 13500, lm.val 1500
Limiting data set to: 100
Tokenized data loaded, cls.trn 100, cls.val 100
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁.', '▁,', '▁der', '▁die', '▁und', '▁in', 'en', "▁&'", 's', '-']
Loading last classifier
Starting classifier training
epoch     train_loss  valid_loss  accuracy
1         0.441340    0.716396    0.840000
epoch     train_loss  valid_loss  accuracy
1         0.312035    0.532610    0.910000
epoch     train_loss  valid_loss  accuracy
1         0.267714    0.462694    0.920000
epoch     train_loss  valid_loss  accuracy
1         0.242031    0.430018    0.930000
2         0.231161    0.398335    0.930000
Saving models at data/mldoc/de-1/models/sp30k/lstm_nl4-100-2x.m
Loss and accuracy using (cls_last): [0.33284584, tensor(0.9252)]
```
#### 8 epoches at the end
```
python -m ulmfit cls --dataset-path data/mldoc/de-1  --base-lm-path data/mldoc/de-1/models/sp30k/lstm_nl4.m  --lang=de --name 'nl4-100-e8' --cuda-id=1 - train 0 --bs 40 --limit=100 --num-cls-epochs=8  --drop-mult-cls=0.3
Max vocab: 30000
Cache dir: data/mldoc/de-1/models/sp30k
Model dir: data/mldoc/de-1/models/sp30k/lstm_nl4-100-e8.m
Loading validation data/mldoc/de-1/de.dev.csv
Tokenized data loaded, lm.trn 13500, lm.val 1500
Limiting data set to: 100
Tokenized data loaded, cls.trn 100, cls.val 100
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁.', '▁,', '▁der', '▁die', '▁und', '▁in', 'en', "▁&'", 's', '-']
Training args:  {'tie_weights': True, 'clip': 0.12, 'bptt': 70, 'pretrained_fnames': [PosixPath('data/mldoc/de-1/models/sp30k/lstm_nl4.m/lm_best'), PosixPath('data/mldoc/de-1/models/sp30k/lstm_nl4.m/../itos')], 'pretrained_model': None, 'drop_mult': 0.3} dps:  [0.25 0.1  0.2  0.02 0.15]
Unknown tokens 0, first 100: []
data/mldoc/de-1/models/sp30k
Saving info data/mldoc/de-1/models/sp30k/lstm_nl4-100-e8.m/info.json
Starting classifier training
epoch     train_loss  valid_loss  accuracy
1         1.208816    1.324359    0.280000
epoch     train_loss  valid_loss  accuracy
1         0.716811    1.195012    0.440000
epoch     train_loss  valid_loss  accuracy
1         0.535809    1.075753    0.590000
epoch     train_loss  valid_loss  accuracy
1         0.499198    1.018431    0.760000
2         0.480971    0.948658    0.880000
3         0.468659    0.866477    0.860000
4         0.460322    0.770794    0.880000
5         0.461138    0.704613    0.900000
6         0.442423    0.623944    0.900000
7         0.422423    0.568031    0.920000
8         0.417041    0.527571    0.930000
Saving models at data/mldoc/de-1/models/sp30k/lstm_nl4-100-e8.m
Loss and accuracy using (cls_last): [0.47343642, tensor(0.9070)]
```
Dropout 0.6
```
python -m ulmfit cls --dataset-path data/mldoc/de-1  --base-lm-path data/mldoc/de-1/models/sp30k/lstm_nl4.m  --lang=de --name 'nl4-100-e8dp6' --cuda-id=1 - train 0 --bs 40 --limit=100 --num-cls-epochs=8  --drop-mult-cls=0.6
Max vocab: 30000
Cache dir: data/mldoc/de-1/models/sp30k
Model dir: data/mldoc/de-1/models/sp30k/lstm_nl4-100-e8dp6.m
Loading validation data/mldoc/de-1/de.dev.csv
Tokenized data loaded, lm.trn 13500, lm.val 1500
Limiting data set to: 100
Tokenized data loaded, cls.trn 100, cls.val 100
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁.', '▁,', '▁der', '▁die', '▁und', '▁in', 'en', "▁&'", 's', '-']
Training args:  {'tie_weights': True, 'clip': 0.12, 'bptt': 70, 'pretrained_fnames': [PosixPath('data/mldoc/de-1/models/sp30k/lstm_nl4.m/lm_best'), PosixPath('data/mldoc/de-1/models/sp30k/lstm_nl4.m/../itos')], 'pretrained_model': None, 'drop_mult': 0.3} dps:  [0.25 0.1  0.2  0.02 0.15]
Unknown tokens 0, first 100: []
data/mldoc/de-1/models/sp30k
Saving info data/mldoc/de-1/models/sp30k/lstm_nl4-100-e8dp6.m/info.json
Starting classifier training
epoch     train_loss  valid_loss  accuracy
1         1.151162    1.323036    0.280000
epoch     train_loss  valid_loss  accuracy
1         0.745338    1.160084    0.610000
epoch     train_loss  valid_loss  accuracy
1         0.535118    1.041519    0.770000
epoch     train_loss  valid_loss  accuracy
1         0.459913    0.995187    0.860000
2         0.451289    0.949036    0.830000
3         0.460395    0.885940    0.800000
4         0.454847    0.848194    0.770000
5         0.447404    0.788741    0.810000
6         0.428524    0.748181    0.760000
7         0.419571    0.696069    0.760000
8         0.408938    0.661937    0.770000
Saving models at data/mldoc/de-1/models/sp30k/lstm_nl4-100-e8dp6.m
Loss and accuracy using (cls_last): [0.53202456, tensor(0.8830)]
```
```
python -m ulmfit cls --dataset-path data/mldoc/de-1  --base-lm-path data/mldoc/de-1/models/sp30k/lstm_nl4.m  --lang=de --name 'nl4-100-e8dp6x2' --cuda-id=1 - train 0 --bs 40 --limit=100 --num-cls-epochs=8  --drop-mult-cls=0.6
Max vocab: 30000
Cache dir: data/mldoc/de-1/models/sp30k
Model dir: data/mldoc/de-1/models/sp30k/lstm_nl4-100-e8dp6x2.m
Loading validation data/mldoc/de-1/de.dev.csv
Tokenized data loaded, lm.trn 13500, lm.val 1500
Limiting data set to: 100
Tokenized data loaded, cls.trn 100, cls.val 100
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁.', '▁,', '▁der', '▁die', '▁und', '▁in', 'en', "▁&'", 's', '-']
Training args:  {'tie_weights': True, 'clip': 0.12, 'bptt': 70, 'pretrained_fnames': [PosixPath('data/mldoc/de-1/models/sp30k/lstm_nl4.m/lm_best'), PosixPath('data/mldoc/de-1/models/sp30k/lstm_nl4.m/../itos')], 'pretrained_model': None, 'drop_mult': 0.3} dps:  [0.25 0.1  0.2  0.02 0.15]
Unknown tokens 0, first 100: []
data/mldoc/de-1/models/sp30k
Saving info data/mldoc/de-1/models/sp30k/lstm_nl4-100-e8dp6x2.m/info.json
Starting classifier training
epoch     train_loss  valid_loss  accuracy
1         1.126586    1.338514    0.470000
epoch     train_loss  valid_loss  accuracy
1         0.751379    1.181071    0.550000
epoch     train_loss  valid_loss  accuracy
1         0.559534    1.083532    0.810000
epoch     train_loss  valid_loss  accuracy
1         0.444137    1.034607    0.870000
2         0.438850    0.983929    0.830000
3         0.436560    0.906958    0.840000
4         0.447400    0.847952    0.840000
5         0.431961    0.783818    0.850000
6         0.422364    0.713126    0.850000
7         0.414145    0.662799    0.840000
8         0.407066    0.630168    0.840000
Saving models at data/mldoc/de-1/models/sp30k/lstm_nl4-100-e8dp6x2.m
Loss and accuracy using (cls_last): [0.46259913, tensor(0.9147)]
```

```
python -m ulmfit cls --dataset-path data/mldoc/de-1  --base-lm-path data/mldoc/de-1/models/sp30k/lstm_nl4.m  --lang=de --name 'nl4-100-e8dp2' --cuda-id=1 - train 0 --bs 40 --limit=100 --num-cls-epochs=8  --drop-mult-cls=0.2
Max vocab: 30000
Cache dir: data/mldoc/de-1/models/sp30k
Model dir: data/mldoc/de-1/models/sp30k/lstm_nl4-100-e8dp2.m
Loading validation data/mldoc/de-1/de.dev.csv
Tokenized data loaded, lm.trn 13500, lm.val 1500
Limiting data set to: 100
Tokenized data loaded, cls.trn 100, cls.val 100
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁.', '▁,', '▁der', '▁die', '▁und', '▁in', 'en', "▁&'", 's', '-']
Training args:  {'tie_weights': True, 'clip': 0.12, 'bptt': 70, 'pretrained_fnames': [PosixPath('data/mldoc/de-1/models/sp30k/lstm_nl4.m/lm_best'), PosixPath('data/mldoc/de-1/models/sp30k/lstm_nl4.m/../itos')], 'pretrained_model': None, 'drop_mult': 0.3} dps:  [0.25 0.1  0.2  0.02 0.15]
Unknown tokens 0, first 100: []
data/mldoc/de-1/models/sp30k
Saving info data/mldoc/de-1/models/sp30k/lstm_nl4-100-e8dp2.m/info.json
Starting classifier training
epoch     train_loss  valid_loss  accuracy
1         1.172059    1.311985    0.280000
epoch     train_loss  valid_loss  accuracy
1         0.721259    1.180611    0.720000
epoch     train_loss  valid_loss  accuracy
1         0.495393    1.051538    0.770000
epoch     train_loss  valid_loss  accuracy
1         0.445929    0.984670    0.830000
2         0.430556    0.897431    0.870000
3         0.442683    0.800808    0.900000
4         0.427033    0.711604    0.880000
5         0.411931    0.624835    0.890000
6         0.397705    0.560819    0.900000
7         0.387848    0.506201    0.900000
8         0.380063    0.459507    0.900000
Saving models at data/mldoc/de-1/models/sp30k/lstm_nl4-100-e8dp2.m
Loss and accuracy using (cls_last): [0.41103342, tensor(0.9105)]
```

#### 2x e8
```
python -m ulmfit cls --dataset-path data/mldoc/de-1  --base-lm-path data/mldoc/de-1/models/sp30k/lstm_nl4.m  --lang=de --name 'nl4-100-2nd-2x' --cuda-id=1 - train 0 --bs 40 --limit=100 --num-cls-epochs=8 --drop-mult-cls=0.2
Max vocab: 30000
Cache dir: data/mldoc/de-1/models/sp30k
Model dir: data/mldoc/de-1/models/sp30k/lstm_nl4-100-2nd-2x.m
Loading validation data/mldoc/de-1/de.dev.csv
Tokenized data loaded, lm.trn 13500, lm.val 1500
Limiting data set to: 100
Tokenized data loaded, cls.trn 100, cls.val 100
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁.', '▁,', '▁der', '▁die', '▁und', '▁in', 'en', "▁&'", 's', '-']
Training args:  {'tie_weights': True, 'clip': 0.12, 'bptt': 70, 'pretrained_fnames': [PosixPath('data/mldoc/de-1/models/sp30k/lstm_nl4.m/lm_best'), PosixPath('data/mldoc/de-1/models/sp30k/lstm_nl4.m/../itos')], 'pretrained_model': None, 'drop_mult': 0.3} dps:  [0.25 0.1  0.2  0.02 0.15]
Unknown tokens 0, first 100: []
data/mldoc/de-1/models/sp30k
Saving info data/mldoc/de-1/models/sp30k/lstm_nl4-100-2nd-2x.m/info.json
Starting classifier training
epoch     train_loss  valid_loss  accuracy
1         1.122616    1.290291    0.300000
epoch     train_loss  valid_loss  accuracy
1         0.746805    1.166377    0.730000
epoch     train_loss  valid_loss  accuracy
1         0.535163    1.058924    0.900000
epoch     train_loss  valid_loss  accuracy
1         0.437773    1.022764    0.850000
2         0.449220    0.949963    0.820000
3         0.443732    0.854293    0.860000
4         0.432721    0.746918    0.900000
5         0.423113    0.718745    0.840000
6         0.403492    0.671295    0.820000
7         0.399091    0.539798    0.900000
8         0.394950    0.508265    0.900000
Saving models at data/mldoc/de-1/models/sp30k/lstm_nl4-100-2nd-2x.m
Loading validation data/mldoc/de-1/de.dev.csv
Loss and accuracy using (cls_last): [0.44688165, tensor(0.9062)]
Loss and accuracy using (cls_best): [0.44688165, tensor(0.9062)]
```
```
python -m ulmfit cls --dataset-path data/mldoc/de-1  --base-lm-path data/mldoc/de-1/models/sp30k/lstm_nl4.m  --lang=de --name 'nl4-100-2nd-2x' --cuda-id=1 - train 0 --bs 40 --limit=100 --num-cls-epochs=8 --drop-mult-cls=0.2
Max vocab: 30000
Cache dir: data/mldoc/de-1/models/sp30k
Model dir: data/mldoc/de-1/models/sp30k/lstm_nl4-100-2nd-2x.m
Loading validation data/mldoc/de-1/de.dev.csv
Tokenized data loaded, lm.trn 13500, lm.val 1500
Limiting data set to: 100
Tokenized data loaded, cls.trn 100, cls.val 100
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁.', '▁,', '▁der', '▁die', '▁und', '▁in', 'en', "▁&'", 's', '-']
Loading last classifier
Starting classifier training
epoch     train_loss  valid_loss  accuracy
1         0.422151    0.797943    0.720000
epoch     train_loss  valid_loss  accuracy
1         0.357166    0.736997    0.780000
epoch     train_loss  valid_loss  accuracy
1         0.238496    1.497305    0.660000
epoch     train_loss  valid_loss  accuracy
1         0.312356    1.522862    0.660000
2         0.267801    1.518249    0.660000
3         0.244077    1.110030    0.680000
4         0.264972    0.798898    0.770000
5         0.236816    0.398245    0.860000
6         0.251284    0.415783    0.860000
7         0.244988    0.417737    0.860000
8         0.240362    0.415114    0.860000
Saving models at data/mldoc/de-1/models/sp30k/lstm_nl4-100-2nd-2x.m
Loss and accuracy using (cls_last): [0.27954015, tensor(0.9125)]
Loss and accuracy using (cls_best): [0.27954015, tensor(0.9125)]
```
### Adding noise 
#### 40%
```
python -m ulmfit cls --dataset-path data/mldoc/de-1  --base-lm-path data/mldoc/de-1/models/sp30k/lstm_nl4.m  --lang=de --name 'nl4-noise0.4' --cuda-id=1 - train 0 --bs 40 --noise=0.4 --num-cls-epochs=8 --drop-mult-cls=0.2
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1/models/sp30k/lstm_nl4-noise0.4.m
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1/de.dev.csv
Tokenized data loaded, lm.trn 13500, lm.val 1500
Added noise to 400 examples, only 0.6 have correct labels
Running tokenization...
Saving tokenized: cls.trn 1000, cls.val 1000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁.', '▁,', '▁der', '▁die', '▁und', '▁in', 'en', "▁&'", 's', '-']
Training args:  {'tie_weights': True, 'clip': 0.12, 'bptt': 70, 'pretrained_fnames': [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1/models/sp30k/lstm_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1/models/sp30k/lstm_nl4.m/../itos')], 'pretrained_model': None, 'drop_mult': 0.3} dps:  [0.25 0.1  0.2  0.02 0.15]
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1/models/sp30k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1/models/sp30k/lstm_nl4-noise0.4.m/info.json
Starting classifier training
epoch     train_loss  valid_loss  accuracy
1         1.053928    0.938391    0.535000
epoch     train_loss  valid_loss  accuracy
1         0.941778    0.599400    0.836000
epoch     train_loss  valid_loss  accuracy
1         0.858363    0.675211    0.760000
epoch     train_loss  valid_loss  accuracy
1         0.768678    0.645293    0.788000
2         0.758538    0.636551    0.780000
3         0.753799    0.673323    0.708000
4         0.731245    0.638630    0.736000
5         0.691206    0.659491    0.717000
6         0.691426    0.682510    0.696000
7         0.672320    0.668610    0.702000
8         0.653569    0.669633    0.694000
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1/models/sp30k/lstm_nl4-noise0.4.m
Loss and accuracy using (cls_last): [0.62477165, tensor(0.7717)]
```
#### 15%
```
python -m ulmfit cls --dataset-path data/mldoc/de-1  --base-lm-path data/mldoc/de-1/models/sp30k/lstm_nl4.m  --lang=de --name 'nl4-noise0.15' --cuda-id=1 - train 0 --bs 40 --noise=0.15 --num-cls-epochs=2 --drop-mult-cls=0.2
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1/models/sp30k/lstm_nl4-noise0.15.m
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1/de.dev.csv
Tokenized data loaded, lm.trn 13500, lm.val 1500
Added noise to 150 examples, only 0.85 have correct labels
Running tokenization...
Saving tokenized: cls.trn 1000, cls.val 1000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁.', '▁,', '▁der', '▁die', '▁und', '▁in', 'en', "▁&'", 's', '-']
Training args:  {'tie_weights': True, 'clip': 0.12, 'bptt': 70, 'pretrained_fnames': [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1/models/sp30k/lstm_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1/models/sp30k/lstm_nl4.m/../itos')], 'pretrained_model': None, 'drop_mult': 0.3} dps:  [0.25 0.1  0.2  0.02 0.15]
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1/models/sp30k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1/models/sp30k/lstm_nl4-noise0.15.m/info.json
Starting classifier training
epoch     train_loss  valid_loss  accuracy
1         0.836104    0.584330    0.897000
epoch     train_loss  valid_loss  accuracy
1         0.692108    0.303470    0.930000
epoch     train_loss  valid_loss  accuracy
1         0.653277    0.330520    0.924000
epoch     train_loss  valid_loss  accuracy
1         0.541086    0.331944    0.922000
2         0.523274    0.335986    0.922000
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1/models/sp30k/lstm_nl4-noise0.15.m
Loss and accuracy using (cls_last): [0.28749043, tensor(0.9355)]
```