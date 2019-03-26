# FR
## LM
```
LANG=fr
python -m ulmfit lm --dataset-path data/wiki/${LANG}-100 --tokenizer='vf' --nl 4 --name 'nl4' --max-vocab 60000 --lang ${LANG} --qrnn=True - train 10 --bs=50 --drop_mult=0  --label-smoothing-eps=0.0
Max vocab: 60000
Cache dir: data/wiki/fr-100/models/vf60k
Model dir: data/wiki/fr-100/models/vf60k/qrnn_nl4.m
Wiki text was split to 174227 articles
Wiki text was split to 491 articles
Running tokenization lm...
Data lm, trn: 174227, val: 491
Size of vocabulary: 60003
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', 'de', ',', '\n', '.', 'la', 'le', 'et', 'à', 'en', "l'", "&'", 'les']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Bptt 70
Training lm from random weights
epoch     train_loss  valid_loss  accuracy
1         3.359852    3.022507    0.434433
2         3.253006    2.955765    0.435078
3         3.274156    2.917242    0.442870
4         3.181276    2.850124    0.451273
5         3.169587    2.813115    0.456411
6         3.075235    2.773676    0.462836
7         3.054632    2.723182    0.469485
8         2.964262    2.661821    0.479831
9         3.019209    2.631244    0.487013
10        2.899521    2.618838    0.489004
Total time: 10:48:33
data/wiki/fr-100/models/vf60k
Saving info data/wiki/fr-100/models/vf60k/qrnn_nl4.m/info.json
```
## CLS



# ES
## LM
```
LANG=es
python -m ulmfit lm --dataset-path data/wiki/${LANG}-100 --tokenizer='vf' --nl 4 --name 'nl4' --max-vocab 60000 --lang ${LANG} --qrnn=True - train 10 --bs=50 --drop_mult=0  --label-smoothing-eps=0.0
Max vocab: 60000
Cache dir: data/wiki/es-100/models/vf60k
Model dir: data/wiki/es-100/models/vf60k/qrnn_nl4.m
Wiki text was split to 161509 articles
Wiki text was split to 78 articles
Running tokenization lm...
Data lm, trn: 161509, val: 78
Size of vocabulary: 60003
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', 'de', ',', '\n', '.', 'la', 'el', 'en', 'y', 'a', "&'", 'que', 'los']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Bptt 70
Training lm from random weights
epoch     train_loss  valid_loss  accuracy
1         3.285345    3.884676    0.312458
2         3.157721    3.832607    0.313905
3         3.193605    3.800210    0.316862
4         3.152273    3.747068    0.319891
5         3.028921    3.713120    0.324912
6         3.067516    3.652925    0.330345
7         3.006576    3.571537    0.339488
8         2.922181    3.529282    0.345483
9         2.871947    3.497736    0.352535
10        2.862057    3.491642    0.354063
Total time: 14:46:42
data/wiki/es-100/models/vf60k
Saving info data/wiki/es-100/models/vf60k/qrnn_nl4.m/info.json
```
## MLDoc

```
python -m ulmfit cls --dataset-path data/mldoc/${LANG}-1 --base-lm-path  data/wiki/${LANG}-100/models/vf60k/qrnn_nl4.m  --lang=${LANG} --name nl4 - train 20 --bs 18 --num-cls-epochs=4 --lr_sched=1cycle --label-smoothing-eps=0.1
Max vocab: 60000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/vf60k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/vf60k/qrnn_nl4.m
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/es.dev.csv
Running tokenization lm...
Data lm, trn: 13013, val: 1445
Running tokenization cls...
Data cls, trn: 1000, val: 1000
Running tokenization tst...
Data tst, trn: 1000, val: 4000
Size of vocabulary: 34317
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', 'de', ',', '.', 'el', 'la', 'a', 'en', ')', '(', 'y', 'los', 'que']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 17152, first 100: ['pct', 'reuter', 'corresponsalía', 'mln', 'indice', 'cotizaba', 'mlns', '585-8308', 'francfort', 'oct', 'jul', 'abr', '585-2154', 'ibex-35', 'feb', 'ibex', 'ago', '585-2152', 'bundesbank', 'ftse', '585-2196', 'interanual', '585-2159', 'cac-40', 'cotizaban', 'uem', 'm.m', '10a', 'alcista', 'bbv', 'anoche', 'argentaria', 'pagarés', 'btp', 'transferibles', 'c.l.p', 'bch', '8,80', '585-8315', 'corros', 'retevisión', '7,35', 'spread', 'bln', 'cnmv', 'decenal', 'opv', 'vespertina', 'greenspan', 'alzas', 'nikkei', 'cambista', 'tir', 'preapertura', 'mibtel', 'tabacalera', 'ptas', 'día-día', 'diff', '18-26', '6-12', 'dif.d.ant', 'max.año', 'min.año', 'spi', 'inem', 'indust', 'fecsa', 'securities', 'repos', 'fomc', 'obligs', 'mibor', 'descartaban', 'sepi', 'interbancario', 'tietmeyer', '5,50', 'piqué', '6,75', 'aprobacion', 'moscu', 'brutas', 'deficit', '0830', 'buba', 'g-7', 'waigel', 'stet', 'petróleo-químicas', '.ibex', '5,25', '6,00', '3m', '5,30', 'trimestrales', 'cauto', 'smi', 'ant-', 'facilitadas']
Bptt 70
Training lm from:  [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/wiki/es-100/models/vf60k/qrnn_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/wiki/es-100/models/vf60k/qrnn_nl4.m/../itos')]
epoch     train_loss  valid_loss  accuracy
1         3.714449    2.774868    0.469673
Total time: 01:42
epoch     train_loss  valid_loss  accuracy
1         3.239635    2.591123    0.496131
2         2.935826    2.367645    0.535486
3         2.631979    2.196012    0.564117
4         2.640709    2.058490    0.582902
5         2.434918    1.949251    0.599310
6         2.293211    1.855961    0.613708
7         2.224960    1.773834    0.626423
8         2.188689    1.698404    0.639268
9         2.024225    1.623230    0.653119
10        2.041964    1.555204    0.665692
11        1.925207    1.492332    0.677868
12        1.864637    1.421467    0.693237
13        1.779024    1.361629    0.706401
14        1.817028    1.301509    0.719889
15        1.719223    1.261717    0.730797
16        1.573684    1.221963    0.740282
17        1.583578    1.192796    0.747645
18        1.590957    1.174528    0.751411
19        1.546806    1.167247    0.753300
20        1.514999    1.165146    0.753615
Total time: 37:16
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/vf60k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/vf60k/qrnn_nl4.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.654116    0.368231    0.907000
2         0.447137    0.287264    0.961000
3         0.308758    0.285717    0.958000
4         0.216707    0.275839    0.962000
Total time: 00:42
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/vf60k/qrnn_nl4.m
Loss and accuracy using (cls_best): [0.14618756, tensor(0.9597)] [0.16216491, tensor(0.9620)]
val_loss:     0.16216491
val_accuracy: 0.9620000123977661
tst_loss:     0.14618756
tst_accuracy: 0.9597499966621399
```


# IT
## LM
```
LANG=it
python -m ulmfit lm --dataset-path data/wiki/${LANG}-100 --tokenizer='vf' --nl 4 --name 'nl4' --max-vocab 60000 --lang ${LANG} --qrnn=True - train 10 --bs=50 --drop_mult=0  --label-smoothing-eps=0.0

Max vocab: 60000
Cache dir: data/wiki/it-100/models/vf60k
Model dir: data/wiki/it-100/models/vf60k/qrnn_nl4.m
Wiki text was split to 164583 articles
Wiki text was split to 98 articles
Data lm, trn: 164583, val: 98
Size of vocabulary: 60003
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', ',', '\n', '.', 'di', 'e', "&'", 'il', 'la', 'in', 'a', 'del', 'che']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Bptt 70
Training lm from random weights
epoch     train_loss  valid_loss  accuracy
1         3.629171    4.075579    0.290754
2         3.496484    4.007234    0.291424
3         3.541803    3.973911    0.294861
4         3.431979    3.926369    0.299076
5         3.432869    3.880250    0.303598
6         3.356332    3.823208    0.309304
7         3.256672    3.760301    0.316393
8         3.312303    3.708765    0.323862
9         3.240380    3.670833    0.329326
10        3.240536    3.661237    0.331286
Total time: 15:32:22
data/wiki/it-100/models/vf60k
Saving info data/wiki/it-100/models/vf60k/qrnn_nl4.m/info.json
```


```bash
python -m ulmfit cls --dataset-path data/mldoc/${LANG}-1 --base-lm-path  data/wiki/${LANG}-100/models/vf60k/qrnn_nl4.m  --lang=${LANG} --name nl4 - train 20 --bs 18 --num-cls-epochs=4 --lr_sched=1cycle --label-smoothing-eps=0.1
Max vocab: 60000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1/models/vf60k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1/models/vf60k/qrnn_nl4.m
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1/it.dev.csv
Data lm, trn: 13500, val: 1500
Running tokenization cls...
Data cls, trn: 1000, val: 1000
Running tokenization tst...
Data tst, trn: 1000, val: 4000
Size of vocabulary: 29600
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '.', ',', 'di', 'e', ')', '(', 'il', "'", 'a', 'in', 'la', 'del']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 13370, first 100: ['pct', 'reuter', 'mld', 'mln', 'societa', 'dealer', 'btp', 'ott', 'dlr', 'attivita', 'venerdi', 'nov', 'feb', 'dic', 'stet', 'mibtel', 'bundesbank', 'bankitalia', 'mib30', 'perche', 'ipsoa', 'comit', 'cct', 'nil', 'cedola', 'puo', 'possibilita', 'lunedi', 'tranche', 'stg', 'warrant', 'stamane', 'ctz', 'giovedi', 'citta', 'ord', 'consob', 'uem', 'martedi', 'spread', 'verra', 't-bond', 'mercoledi', 'risp', 'viv', 'ffr', 'avra', 'compart', 'gmn', 'dovra', 'potra', 'fib30', 'contrattazioni', 'gemina', 'frf', 'controvalore', 'overnight', 'cir', 'apr', 'consensus', 'tendenziale', 'nikkei', 'autorita', 'tus', 'pretasse', 'fib', 'rialzi', 'fomc', 'gilt', 'circ', 'destagionalizzati', 'prec', 'liquidita', 'ecu', 'destagionalizzato', 'cariplo', 'stamani', 'obbligazionario', 'bur', 'imi', 'aggiudicazione', 'treu', 'ambroveneto', 'fixing', 'hpi', 'rnc', 'capacita', 'dietimi', 'greenspan', 'tietmeyer', 'waigel', 'nasdaq', 'eltsin', 'redditivita', 'liffe', 'telematico', 'ifil', 'interpellati', '6,25', 'visco']
Bptt 70
Training lm from:  [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/wiki/it-100/models/vf60k/qrnn_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/wiki/it-100/models/vf60k/qrnn_nl4.m/../itos')]
epoch     train_loss  valid_loss  accuracy
1         4.406445    3.675336    0.338066
Total time: 01:10
epoch     train_loss  valid_loss  accuracy
1         3.870676    3.516882    0.355481
2         3.633525    3.322235    0.383076
3         3.454955    3.121748    0.408930
4         3.210115    2.935245    0.433205
5         3.112426    2.775076    0.452784
6         2.991053    2.638768    0.471221
7         2.904022    2.533667    0.485577
8         2.808465    2.426029    0.501932
9         2.713658    2.320023    0.518699
10        2.580141    2.226892    0.533786
11        2.532727    2.133867    0.549680
12        2.449591    2.034733    0.567797
13        2.387805    1.963019    0.583013
14        2.337399    1.880745    0.598986
15        2.217255    1.818780    0.612503
16        2.175724    1.764977    0.623581
17        2.057536    1.726874    0.631422
18        2.093975    1.705599    0.635835
19        2.030292    1.694430    0.637838
20        2.057254    1.691360    0.638669
Total time: 32:28
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1/models/vf60k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1/models/vf60k/qrnn_nl4.m/info.json

***OOTM**
```

```
python -m ulmfit cls --dataset-path data/mldoc/${LANG}-1 --base-lm-path  data/wiki/${LANG}-100/models/vf60k/qrnn_nl4.m  --lang=${LANG} --name nl4 - train 20 --bs 18 --num-cls-epochs=4 --lr_sched=1cycle --label-smoothing-eps=0.1
python -m ulmfit cls --dataset-path data/mldoc/${LANG}-1 --base-lm-path  data/wiki/${LANG}-100/models/vf60k/qrnn_nl4.m  --lang=${LANG} --name nl4 - train 20 --bs 10 --num-cls-epochs=4 --lr_sched=1cycle --label-smoothing-eps=0.1
Max vocab: 60000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1/models/vf60k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1/models/vf60k/qrnn_nl4.m
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1/it.dev.csv
Data lm, trn: 13500, val: 1500
Data cls, trn: 1000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 29600
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '.', ',', 'di', 'e', ')', '(', 'il', "'", 'a', 'in', 'la', 'del']
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.736275    0.717692    0.837000
2         0.593485    0.444027    0.876000
3         0.376322    0.411704    0.907000
4         0.244267    0.370927    0.915000
Total time: 00:33
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1/models/vf60k/qrnn_nl4.m
Loss and accuracy using (cls_best): [0.3200554, tensor(0.8997)] [0.27118126, tensor(0.9150)]
val_loss:     0.27118126
val_accuracy: 0.9150000214576721
tst_loss:     0.3200554
tst_accuracy: 0.8997499942779541
```