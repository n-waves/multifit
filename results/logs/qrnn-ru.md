# QRNN RU
## SP15k nl4
## LM
export CUDA_VISIBLE_DEVICES=3
LANG=ru
python -m ulmfit lm --dataset-path data/wiki/ru-100 --bidir=False --qrnn=True --nl 4 --tokenizer='sp' --max-vocab 15000 --lang ru --name 'nl4'  -  train 10 --drop-mult=0 --bs=50 --label-smoothing-eps=0.1

## SP15k nl8
### LM
```
5         2.869308    2.905951    0.466976
6         2.768955    2.782804    0.481852
7         2.654484    2.676304    0.495593
8         2.585963    2.591748    0.508447
9         2.512042    2.526819    0.518860
10        2.520543    2.509287    0.521890
Total time: 18:46:01
data/wiki/ru-100/models/sp15k
Saving info data/wiki/ru-100/models/sp15k/qrnn_nl8.m/info.json
```
### MLDoc
```
python -m ulmfit cls --dataset-path data/mldoc/${LANG}-1  --base-lm-path data/wiki/${LANG}-100/models/sp15k/qrnn_nl8.m  --lang=${LANG} --name 'nl8' - train 20 --bs 20 --num-cls-epochs=8 --lr_sched=1cycle  --label-smoothing-eps=0.1
Max vocab: 15000
Cache dir: /home/n-waves/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp15k
Model dir: /home/n-waves/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp15k/qrnn_nl8.m
Loading validation /home/n-waves/workspace/ulmfit-multilingual/data/mldoc/ru-1/ru.dev.csv
Running tokenization lm...
Data lm, trn: 9195, val: 1021

Running tokenization cls...
Data cls, trn: 1000, val: 1000

Running tokenization tst...
Data tst, trn: 1000, val: 4000

Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁.', '▁в', 'а', 'и', 'е', '▁и', 'й', '▁на', 'х']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
Training lm from:  [PosixPath('/home/n-waves/workspace/ulmfit-multilingual/data/wiki/ru-100/models/sp15k/qrnn_nl8.m/lm_best'), PosixPath('/home/n-waves/workspace/ulmfit-multilingual/data/wiki/ru-100/models/sp15k/qrnn_nl8.m/../itos')]
epoch     train_loss  valid_loss  accuracy
1         2.923423    2.334532    0.529978

Total time: 02:46
epoch     train_loss  valid_loss  accuracy
1         2.462077    2.150593    0.563281

2         2.230013    1.972095    0.596198

3         2.118523    1.812012    0.623204

4         1.916368    1.690016    0.644060

5         1.842718    1.585770    0.661704

6         1.748630    1.513972    0.674130

7         1.675032    1.447667    0.686207

8         1.628485    1.393949    0.695972

9         1.564814    1.330838    0.707272

10        1.553933    1.283114    0.715716

11        1.441891    1.234810    0.726201

12        1.496388    1.185676    0.735977

13        1.383019    1.141014    0.745528

14        1.256620    1.094201    0.755120

15        1.306187    1.052457    0.764280

16        1.297933    1.028387    0.769747

17        1.319773    1.004256    0.775285

18        1.178073    0.989788    0.778480

19        1.252248    0.982740    0.780057

20        1.177640    0.981201    0.780267

Total time: 1:24:58
/home/n-waves/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp15k
Saving info /home/n-waves/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp15k/qrnn_nl8.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.882775    0.510930    0.826000

2         0.683476    0.513669    0.847000

3         0.556661    0.590375    0.839000

4         0.454019    0.757216    0.828000

5         0.344460    0.549675    0.870000

6         0.246039    0.630242    0.861000

7         0.173423    0.649066    0.858000

8         0.098640    0.638015    0.867000

Total time: 05:11
Saving models at /home/n-waves/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp15k/qrnn_nl8.m
Loss and accuracy using (cls_best): [0.64393336, tensor(0.8683)]
```

### MLDoc nl8 -2nd
```
 python -m ulmfit cls --dataset-path data/mldoc/${LANG}-1 --base-lm-path data/wiki/${LANG}-100/models/sp15k/qrnn_${NAME}.m  --lang=${LANG} --name ${NAME}-2 - train 20 --bs 18 --num-cls-epochs=4 --lr_sched=1cycle --label-smoothing-eps=0
.1
Max vocab: 15000
Cache dir: /home/n-waves/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp15k
Model dir: /home/n-waves/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp15k/qrnn_nl8-2.m
Loading validation /home/n-waves/workspace/ulmfit-multilingual/data/mldoc/ru-1/ru.dev.csv
Data lm, trn: 9195, val: 1021
Data cls, trn: 1000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁.', '▁в', 'а', 'и', 'е', '▁и', 'й', '▁на', 'х']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
Training lm from:  [PosixPath('/home/n-waves/workspace/ulmfit-multilingual/data/wiki/ru-100/models/sp15k/qrnn_nl8.m/lm_best'), PosixPath('/home/n-waves/workspace/ulmfit-multilingual/data/wiki/ru-100/models/sp15k/qrnn_nl8.m/../itos')]
epoch     train_loss  valid_loss  accuracy
1         3.966292    3.450758    0.527065
Total time: 02:58
epoch     train_loss  valid_loss  accuracy
1         3.495761    3.276329    0.560047
2         3.319947    3.102911    0.593742
3         3.137904    2.955317    0.620171
4         3.040286    2.839161    0.642270
5         2.869962    2.753622    0.658331
6         2.905739    2.680881    0.672860
7         2.836454    2.620925    0.685026
8         2.857271    2.569716    0.695722
9         2.702872    2.520050    0.705589
10        2.701559    2.473591    0.715346
11        2.740815    2.429558    0.725597
12        2.646513    2.389550    0.735010
13        2.587685    2.349614    0.744885
14        2.546527    2.311087    0.754463
15        2.568136    2.278581    0.762980
16        2.492115    2.252367    0.769275
17        2.338561    2.230529    0.775072
18        2.447506    2.218215    0.778437
19        2.364424    2.212115    0.780085
20        2.367132    2.210520    0.780424
Total time: 1:30:47
/home/n-waves/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp15k
Saving info /home/n-waves/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp15k/qrnn_nl8-2.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.969255    0.769736    0.843000
2         0.846340    0.813483    0.839000
3         0.718175    0.705339    0.867000
4         0.609513    0.726442    0.875000
Total time: 02:54
Saving models at /home/n-waves/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp15k/qrnn_nl8-2.m
Loss and accuracy using (cls_best): [0.4056449, tensor(0.8698)]
0.40564489364624023
```






## cls
```
export CUDA_VISIBLE_DEVICES=0
LANG=ru
python -m ulmfit cls --dataset-path data/mldoc/${LANG}-1  --base-lm-path data/wiki/${LANG}-100/models/sp15k/qrnn_nl8.m  --lang=${LANG} --name 'nl8' - train 20 --bs 20 --num-cls-epochs=8 --lr_sched=1cycle --label-smoothing-eps=0.1
```

## SP30k nl4
### LM
```
python -m ulmfit lm --dataset-path data/wiki/ru-100 --bidir=False --qrnn=True --nl 4 --tokenizer='sp' --max-vocab 30000 --lang ru --name 'nl4' --cuda-id=0  -  train 10 --drop-mult=0 --bs=50

Wiki text was split to 193047 articles
Wiki text was split to 460 articles
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁.', '▁в', 'а', '▁и', 'е', 'и', 'й', '▁на', '▁с']
Training args:  {'clip': 0.12, 'drop_mult': 0} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Training lm from random weights
epoch     train_loss  valid_loss  accuracy
1         3.273207    3.350111    0.429702
2         3.169897    3.274238    0.433682
3         3.162197    3.247077    0.435900
4         3.131630    3.168798    0.445252
5         3.042942    3.096774    0.453532
6         2.950550    3.002989    0.465113
7         2.833593    2.902871    0.478954
8         2.829737    2.805592    0.492138
9         2.746991    2.733609    0.503711
10        2.687201    2.708546    0.508050
```
