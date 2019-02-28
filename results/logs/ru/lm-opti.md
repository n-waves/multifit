CUDA_VISIBLE_DEVICES=0 python -m ulmfit lm --dataset-path data/wiki/${LANG}-100 --tokenizer='sp' --nl 5 --name 'nl5-merity' --max-vocab 15000 --lang ${LANG} --qrnn=True --bptt=140 --nh 2500 - train 14 --bs=50 --drop_mult=0  --label-smoothing-eps=0.1
CUDA_VISIBLE_DEVICES=1 python -m ulmfit lm --dataset-path data/wiki/${LANG}-100 --tokenizer='sp' --nl 4 --name 'nl4-wide2' --max-vocab 15000 --lang ${LANG} --qrnn=True --nh 3100 - train 10 --bs=100 --drop_mult=0  --label-smoothing-eps=0.1
CUDA_VISIBLE_DEVICES=2 python -m ulmfit lm --dataset-path data/wiki/${LANG}-100 --tokenizer='sp' --nl 4 --name 'nl4-merity' --max-vocab 15000 --lang ${LANG} --qrnn=True --bptt=140 --nh 2500 - train 14 --bs=50 --drop_mult=0  --label-smoothing-eps=0.1
CUDA_VISIBLE_DEVICES=3 python -m ulmfit lm --dataset-path data/wiki/ru-100 --bidir=False --qrnn=True --nl 4 --tokenizer=sp --max-vocab 15000 --lang ru --name nl4sl - train 10 --drop-mult=0 --bs=50 --label-smoothing-eps=0.1

## 25vocab
CUDA_VISIBLE_DEVICES=3 python -m ulmfit lm --dataset-path data/wiki/ru-100 --bidir=False --qrnn=True --nl 4 --tokenizer=sp --max-vocab 25000 --lang ru --name nl4 - train 10 --drop-mult=0 --bs=50 --label-smoothing-eps=0.1

LANG=ru
CUDA_VISIBLE_DEVICES=0 python -m ulmfit lm --dataset-path data/wiki/${LANG}-100 --tokenizer='sp' --nl 4 --name 'nl4-merity-wide2' --max-vocab 15000 --lang ${LANG} --qrnn=True --bptt=140 --nh 3100 - train 10 --bs=50 --drop_mult=0  --label-smoothing-eps=0.1

##### CLS
export CUDA_VISIBLE_DEVICES=0
LANG=ru
NAME=nl5-merity
python -m ulmfit cls --dataset-path data/mldoc/${LANG}-1 --base-lm-path data/wiki/${LANG}-100/models/sp15k/qrnn_${NAME}.m  --lang=${LANG} --name ${NAME} - train 20 --bs 18 --num-cls-epochs=4 --lr_sched=1cycle --label-smoothing-eps=0.1

export CUDA_VISIBLE_DEVICES=1
LANG=ru
NAME=nl4-wide2
python -m ulmfit cls --dataset-path data/mldoc/${LANG}-1 --base-lm-path data/wiki/${LANG}-100/models/sp15k/qrnn_${NAME}.m  --lang=${LANG} --name ${NAME} - train 20 --bs 18 --num-cls-epochs=4 --lr_sched=1cycle --label-smoothing-eps=0.1

export CUDA_VISIBLE_DEVICES=2
LANG=ru
NAME=nl4-merity
python -m ulmfit cls --dataset-path data/mldoc/${LANG}-1 --base-lm-path data/wiki/${LANG}-100/models/sp15k/qrnn_${NAME}.m  --lang=${LANG} --name ${NAME} - train 20 --bs 18 --num-cls-epochs=4 --lr_sched=1cycle --label-smoothing-eps=0.1

export CUDA_VISIBLE_DEVICES=3
LANG=ru
NAME=nl4sl
python -m ulmfit cls --dataset-path data/mldoc/${LANG}-1 --base-lm-path data/wiki/${LANG}-100/models/sp15k/qrnn_${NAME}.m  --lang=${LANG} --name ${NAME} - train 20 --bs 18 --num-cls-epochs=4 --lr_sched=1cycle --label-smoothing-eps=0.1


-----------------------CLS1
export CUDA_VISIBLE_DEVICES=0
LANG=ru
NAME=nl4
python -m ulmfit cls --dataset-path data/mldoc/${LANG}-1 --base-lm-path data/wiki/${LANG}-100/models/sp15k/qrnn_${NAME}.m  --lang=${LANG} --name ${NAME} - train 20 --bs 18 --num-cls-epochs=4 --lr_sched=1cycle --label-smoothing-eps=0.1


export CUDA_VISIBLE_DEVICES=0
LANG=ru
NAME=nl8
python -m ulmfit cls --dataset-path data/mldoc/${LANG}-1 --base-lm-path data/wiki/${LANG}-100/models/sp15k/qrnn_${NAME}.m  --lang=${LANG} --name ${NAME} - train 20 --bs 18 --num-cls-epochs=4 --lr_sched=1cycle --label-smoothing-eps=0.1


python -m ulmfit cls --dataset-path data/wiki/${LANG}-100 --tokenizer='sp' --nl 5 --name 'nl5-merity' --max-vocab 15000 --lang ${LANG} --qrnn=True --bptt=140 --nh 2500 - train 14 --bs=50 --drop_mult=0  --label-smoothing-eps=0.1

CUDA_VISIBLE_DEVICES=1 python -m ulmfit lm --dataset-path data/wiki/${LANG}-100 --tokenizer='sp' --nl 4 --name 'nl4-wide2' --max-vocab 15000 --lang ${LANG} --qrnn=True --nh 3100 - train 10 --bs=100 --drop_mult=0  --label-smoothing-eps=0.1
CUDA_VISIBLE_DEVICES=2 python -m ulmfit lm --dataset-path data/wiki/${LANG}-100 --tokenizer='sp' --nl 4 --name 'nl4-merity' --max-vocab 15000 --lang ${LANG} --qrnn=True --bptt=140 --nh 2500 - train 14 --bs=50 --drop_mult=0  --label-smoothing-eps=0.1
CUDA_VISIBLE_DEVICES=3 python -m ulmfit lm --dataset-path data/wiki/ru-100 --bidir=False --qrnn=True --nl 4 --tokenizer=sp --max-vocab 15000 --lang ru --name nl4sl - train 10 --drop-mult=0 --bs=50 --label-smoothing-eps=0.1
##

------------------------

7         3.680504    3.678406    0.498396
8         3.556062    3.596037    0.512345
9         3.553716    3.535783    0.523509
10        3.523366    3.515352    0.527935
Total time: 20:03:59
data/wiki/ru-100/models/sp15k
Saving info data/wiki/ru-100/models/sp15k/qrnn_ nl4sl.m/info.json

### Ru
```
export CUDA_VISIBLE_DEVICES=3
LANG=ru
NAME=nl4sl
python -m ulmfit cls --dataset-path data/mldoc/${LANG}-1 --base-lm-path data/wiki/${LANG}-100/models/sp15k/qrnn_${NAME}.m  --lang=${LANG} --name ${NAME} - train 20 --bs 18 --num-cls-epochs=4 --lr_sched=1cycle --label-smoothing-eps=0.1
Max vocab: 15000
Cache dir: /home/test/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp15k
Model dir: /home/test/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp15k/qrnn_nl4sl.m
Loading validation /home/test/workspace/ulmfit-multilingual/data/mldoc/ru-1/ru.dev.csv
Data lm, trn: 9195, val: 1021
Data cls, trn: 1000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁.', '▁в', 'а', 'и', 'е', '▁и', 'й', '▁на', 'х']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
Training lm from:  [PosixPath('/home/test/workspace/ulmfit-multilingual/data/wiki/ru-100/models/sp15k/qrnn_nl4sl.m/lm_best'), PosixPath('/home/test/workspace/ulmfit-multilingual/data/wiki/ru-100/models/sp15k/qrnn_nl4sl.m/../itos')]
epoch     train_loss  valid_loss  accuracy
1         4.427713    3.693394    0.484268
Total time: 01:32
epoch     train_loss  valid_loss  accuracy
1         3.758918    3.446661    0.529820
2         3.394254    3.199054    0.577411
3         3.235364    3.014517    0.610520
4         3.125459    2.871101    0.637153
5         2.994313    2.773862    0.654470
6         2.915075    2.693080    0.669942
7         2.855732    2.622858    0.683629
8         2.755074    2.572147    0.694145
9         2.697898    2.517524    0.704816
10        2.689881    2.468190    0.715927
11        2.579573    2.432807    0.723324
12        2.659464    2.387878    0.733931
13        2.520637    2.344804    0.744233
14        2.482952    2.315014    0.751855
15        2.564730    2.279045    0.761163
16        2.552707    2.255916    0.766971
17        2.511244    2.240169    0.770991
18        2.461429    2.228213    0.774309
19        2.426440    2.222140    0.775745
20        2.425955    2.221128    0.775836
Total time: 1:14:17
/home/test/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp15k
Saving info /home/test/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp15k/qrnn_nl4sl.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.022382    0.779370    0.822000
2         0.866379    0.792353    0.832000
3         0.715650    0.698579    0.865000
4         0.603621    0.693501    0.884000
Total time: 02:05
Saving models at /home/test/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp15k/qrnn_nl4sl.m
Loss and accuracy using (cls_best): [0.3978519, tensor(0.8723)]
0.3978519141674042
0.8722500205039978
```

----

```bash
$ export CUDA_VISIBLE_DEVICES=0
$ LANG=ru
$ NAME=nl4
$ python -m ulmfit cls --dataset-path data/mldoc/${LANG}-1 --base-lm-path data/wiki/${LANG}-100/models/sp15k/qrnn_${NAME}.m  --lang=${LANG} --name ${NAME} - train 20 --bs 18 --num-cls-epochs=4 --lr_sched=1cycle --label-smoothing-eps=0.1
Max vocab: 15000
Cache dir: /home/n-waves/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp15k
Model dir: /home/n-waves/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp15k/qrnn_nl4.m
Loading validation /home/n-waves/workspace/ulmfit-multilingual/data/mldoc/ru-1/ru.dev.csv
Data lm, trn: 9195, val: 1021
Data cls, trn: 1000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁.', '▁в', 'а', 'и', 'е', '▁и', 'й', '▁на', 'х']
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁.', '▁в', 'а', 'и', 'е', '▁и', 'й', '▁на', 'х']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
Training lm from:  [PosixPath('/home/n-waves/workspace/ulmfit-multilingual/data/wiki/ru-100/models/sp15k/qrnn_nl4.m/lm_best'), PosixPath('/home/n-waves/workspace/ulmfit-multilingual/data/wiki/ru-100/models/sp15k/qrnn_nl4.m/../itos')]
epoch     train_loss  valid_loss  accuracy
1         4.531968    3.764185    0.474252
Total time: 01:44
epoch     train_loss  valid_loss  accuracy
1         3.770046    3.506013    0.522443
2         3.546580    3.251341    0.571620
3         3.320569    3.055680    0.606364
4         3.130226    2.912925    0.631395
5         3.072772    2.809725    0.649728
6         2.765424    2.731825    0.662963
7         2.959237    2.662104    0.676203
8         2.807999    2.600417    0.688423
9         2.771271    2.548279    0.699473
10        2.809488    2.501688    0.709020
11        2.707221    2.454946    0.719196
12        2.597226    2.417315    0.728432
13        2.609972    2.376176    0.737923
14        2.590427    2.341666    0.746216
15        2.572995    2.306599    0.754747
16        2.496636    2.285632    0.760806
17        2.508584    2.266456    0.765147
18        2.441373    2.253839    0.768449
19        2.430915    2.249204    0.769536
20        2.426130    2.247966    0.769886
Total time: 47:33
/home/n-waves/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp15k
Saving info /home/n-waves/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp15k/qrnn_nl4.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.060299    0.890710    0.716000
2         0.884965    0.769866    0.853000
3         0.722994    0.723213    0.875000
4         0.609488    0.730594    0.865000
Total time: 01:14
Saving models at /home/n-waves/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp15k/qrnn_nl4.m
Loss and accuracy using (cls_best): [0.39589784, tensor(0.8692)]
0.39589783549308777
0.8692499995231628
```


## wide 2
```bash
$ CUDA_VISIBLE_DEVICES=1 python -m ulmfit lm --dataset-path data/wiki/${LANG}-100 --tokenizer='sp' --nl 4 --name 'nl4-wide2' --max-vocab 15000 --lang ${LANG} --qrnn=True --nh 3100 - train 10 --bs=100 --drop_mult=0  --label-smoothing-eps=0.1
Max vocab: 15000
Cache dir: data/wiki/ru-100/models/sp15k
Model dir: data/wiki/ru-100/models/sp15k/qrnn_nl4-wide2.m
Wiki text was split to 193047 articles
Wiki text was split to 460 articles
Data lm, trn: 193047, val: 460
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁.', '▁в', 'а', 'и', 'е', '▁и', 'й', '▁на', 'х']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Training lm from random weights
epoch     train_loss  valid_loss  accuracy
1         3.908233    3.950865    0.463469
2         3.738863    3.815026    0.477703
3         3.696502    3.779513    0.483625
4         3.692592    3.720908    0.490143
5         3.600519    3.652444    0.501671
6         3.564568    3.582584    0.511550
7         3.472859    3.493226    0.525943
8         3.390483    3.407970    0.541749
9         3.351620    3.344207    0.552758
10        3.329683    3.330087    0.556380
Total time: 51:05:43
data/wiki/ru-100/models/sp15k
Saving info data/wiki/ru-100/models/sp15k/qrnn_nl4-wide2.m/info.json
```
### MLDoc
export CUDA_VISIBLE_DEVICES=1
LANG=ru
NAME=nl4-wide2
python -m ulmfit cls --dataset-path data/mldoc/${LANG}-1 --base-lm-path data/wiki/${LANG}-100/models/sp15k/qrnn_${NAME}.m  --lang=${LANG} --name ${NAME} - train 20 --bs 18 --num-cls-epochs=4 --lr_sched=1cycle --label-smoothing-eps=0.1



## Merity nl4
```bash
 CUDA_VISIBLE_DEVICES=2 python -m ulmfit lm --dataset-path data/wiki/${LANG}-100 --tokenizer='sp' --nl 4 --name 'nl4-merity' --max-vocab 15000 --lang ${LANG} --qrnn=True --bptt=140 --nh 2500 - train 14 --bs=50 --drop_mult=0  --label-smoothing-eps=0.1
Max vocab: 15000
Cache dir: data/wiki/ru-100/models/sp15k
Model dir: data/wiki/ru-100/models/sp15k/qrnn_nl4-merity.m
Wiki text was split to 193047 articles
Wiki text was split to 460 articles
Data lm, trn: 193047, val: 460
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁.', '▁в', 'а', 'и', 'е', '▁и', 'й', '▁на', 'х']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Training lm from random weights
epoch     train_loss  valid_loss  accuracy
1         3.965899    3.977734    0.460046
2         3.806082    3.858176    0.472396
3         3.839230    3.874757    0.469224
4         3.762105    3.868653    0.469943
5         3.800827    3.833991    0.474116
6         3.755466    3.796329    0.479868
7         3.691958    3.747888    0.487367
8         3.660529    3.702986    0.493545
9         3.593282    3.635035    0.504086
10        3.585948    3.579200    0.513631
11        3.473865    3.512114    0.525391
12        3.451973    3.455807    0.535520
13        3.418731    3.417129    0.542943
14        3.385637    3.407541    0.545545
Total time: 51:32:09
data/wiki/ru-100/models/sp15k
Saving info data/wiki/ru-100/models/sp15k/qrnn_nl4-merity.m/info.json
```

#### MLDoc

export CUDA_VISIBLE_DEVICES=2
LANG=ru
NAME=nl4-merity
python -m ulmfit cls --dataset-path data/mldoc/${LANG}-1 --base-lm-path data/wiki/${LANG}-100/models/sp15k/qrnn_${NAME}.m  --lang=${LANG} --name ${NAME} - train 20 --bs 18 --num-cls-epochs=4 --lr_sched=1cycle --label-smoothing-eps=0.1
