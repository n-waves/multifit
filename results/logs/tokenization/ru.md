
## SP25k
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
Saving info data/wiki/ru-100/models/sp25k/qrnn_nl4.m/info.json
```

```
python -m ulmfit cls --dataset-path data/mldoc/${LANG}-1 --base-lm-path data/wiki/${LANG}-100/models/sp25k/qrnn_${NAME}.m  --lang=${LANG} --name ${NAME} - train 20 --bs 18 --num-cls-epochs=4 --lr_sched=1cycle --label-smoothing-eps=0.1

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

## V60k
## VF60k

```
LANG=ru
python -m ulmfit lm --dataset-path data/wiki/${LANG}-100 --tokenizer='vf' --nl 4 --name 'nl4' --max-vocab 60000 --lang ${LANG} --qrnn=True - train 10 --bs=50 --drop_mult=0  --label-smoothing-eps=0.1
Max vocab: 60000
Cache dir: data/wiki/ru-100/models/vf60k
Model dir: data/wiki/ru-100/models/vf60k/qrnn_nl4.m
Wiki text was split to 193047 articles
Wiki text was split to 460 articles
Running tokenization lm...
Data lm, trn: 193047, val: 460
Size of vocabulary: 60003
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', ',', '\n', '.', 'в', 'и', ')', '(', 'на', '—', '«', '»', 'с']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Training lm from random weights
epoch     train_loss  valid_loss  accuracy
1         4.586900    4.478803    0.413023
2         4.483496    4.400461    0.418495
3         4.484620    4.390928    0.418422
4         4.373594    4.350045    0.422567
5         4.350337    4.307665    0.427411
6         4.314571    4.249700    0.436324
7         4.232540    4.183857    0.446341
8         4.252573    4.119820    0.455522
9         4.136978    4.088805    0.462345
10        4.116755    4.079840    0.465394
Total time: 11:24:03
data/wiki/ru-100/models/vf60k
```

## SP15k LSTM nl 3
```bash
python -m ulmfit cls --dataset-path data/mldoc/${LANG}-1 --base-lm-path  data/wiki/${LANG}-100/models/sp15k/lstm_nl4.m  --lang=${LANG} --name nl4 - train 20 --bs 18 --num-cls-epochs=4 --lr_sched=1cycle --label-smoothing-eps=0.1
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp15k/lstm_nl4.m
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1/ru.dev.csv
Data lm, trn: 9195, val: 1021
Data cls, trn: 1000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁.', '▁в', 'а', 'и', 'е', '▁и', 'й', '▁на', 'х']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
Bptt 70
Training lm from:  [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/wiki/ru-100/models/sp15k/lstm_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/wiki/ru-100/models/sp15k/lstm_nl4.m/../itos')]
epoch     train_loss  valid_loss  accuracy
1         3.705343    3.261906    0.558008
Total time: 05:27
epoch     train_loss  valid_loss  accuracy
1         3.243956    3.073661    0.594862
2         3.139877    2.917376    0.625388
3         2.941367    2.786331    0.650792
4         2.846027    2.682831    0.671712
5         2.796714    2.600119    0.687167
6         2.841771    2.527643    0.702408
7         2.726931    2.459425    0.717738
8         2.619217    2.402231    0.729743
9         2.626002    2.349137    0.742474
10        2.535362    2.299844    0.753796
11        2.501980    2.257779    0.764137
12        2.427705    2.209901    0.776203
13        2.393852    2.167961    0.787562
14        2.340693    2.129181    0.797972
15        2.307895    2.094267    0.807763
16        2.330075    2.069201    0.814278
17        2.232444    2.049109    0.820321
18        2.306738    2.038069    0.823257
19        2.232783    2.031799    0.825218
20        2.227589    2.030583    0.825465
Total time: 2:13:57
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp15k/lstm_nl4.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.019235    0.894525    0.820000
2         0.885900    0.831892    0.772000
3         0.714437    0.711899    0.865000
4         0.608688    0.706948    0.868000
Total time: 05:07
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp15k/lstm_nl4.m
Loss and accuracy using (cls_best): [0.42021805, tensor(0.8648)]
0.4202180504798889
0.8647500276565552
```
