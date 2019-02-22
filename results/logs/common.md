# MLDoc
### Different training schedules

### 1cycle -lstm
```
(fastaiv1) pczapla@galatea ~/w/ulmfit-multilingual ❯❯❯ python -m ulmfit eval --glob="mldoc/*-1/models/sp30k/lstm_nl4.m" --name nl4-1cyc  --num-cls-epochs=8 --bs=18 --lr_sched=1cycle                                                  ✘ 1
Processing data/mldoc/de-1/models/sp30k/lstm_nl4.m
de-1
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1/models/sp30k/lstm_nl4-1cyc.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1/de.dev.csv
Data lm, trn: 13500, val: 1500
Data cls, trn: 1000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁.', '▁,', '▁der', '▁die', '▁und', '▁in', 'en', "▁&'", 's', '-']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1/models/sp30k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1/models/sp30k/lstm_nl4-1cyc.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.610423    0.287707    0.920000
2         0.390499    0.266688    0.948000
3         0.366716    0.302463    0.933000
4         0.248321    0.305547    0.937000
5         0.166564    0.411075    0.948000
6         0.083940    0.406182    0.950000
7         0.033326    0.388105    0.949000
8         0.014658    0.397507    0.948000
Total time: 06:42
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1/models/sp30k/lstm_nl4-1cyc.m
Loss and accuracy using (cls_best): [0.3040595, tensor(0.9585)]
Processing data/mldoc/es-1/models/sp30k/lstm_nl4.m
es-1
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp30k/lstm_nl4-1cyc.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/es.dev.csv
Data lm, trn: 13013, val: 1445
Data cls, trn: 1000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁la', '▁.', '▁en', '▁el', '▁y', 's', '▁a', '▁que']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp30k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp30k/lstm_nl4-1cyc.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.610724    0.278892    0.925000
2         0.372022    0.348428    0.937000
3         0.310411    0.386958    0.927000
4         0.215536    0.273834    0.958000
5         0.163195    0.319600    0.958000
6         0.085268    0.313287    0.961000
7         0.037369    0.347500    0.961000
8         0.016851    0.338436    0.963000
Total time: 05:36
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp30k/lstm_nl4-1cyc.m
Loss and accuracy using (cls_best): [0.31034237, tensor(0.9632)]
Processing data/mldoc/fr-1/models/sp30k/lstm_nl4.m
fr-1
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/models/sp30k/lstm_nl4-1cyc.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/fr.dev.csv
Data lm, trn: 13500, val: 1500
Data cls, trn: 1000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁.', "'", 's', '▁la', '▁le', '▁et', '▁l', '▁à']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/models/sp30k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/models/sp30k/lstm_nl4-1cyc.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.642809    0.240702    0.928000
2         0.420564    0.658542    0.852000
3         0.443345    0.244053    0.927000
4         0.338779    0.335634    0.914000
5         0.224778    0.263748    0.928000
6         0.116705    0.280655    0.944000
7         0.072063    0.287557    0.945000
8         0.048084    0.289200    0.946000
Total time: 06:33
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/models/sp30k/lstm_nl4-1cyc.m
Loss and accuracy using (cls_best): [0.29398218, tensor(0.9482)]
Processing data/mldoc/it-1/models/sp30k/lstm_nl4.m
it-1
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1/models/sp30k/lstm_nl4-1cyc.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1/it.dev.csv
Data lm, trn: 13500, val: 1500
Data cls, trn: 1000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁.', '▁di', "▁&'", "'", '▁e', '▁il', '▁la', 'e', '▁in']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1/models/sp30k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1/models/sp30k/lstm_nl4-1cyc.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.736070    0.391008    0.859000
2         0.512531    0.614638    0.860000
3         0.343422    0.594530    0.862000
4         0.370786    0.540225    0.884000
5         0.234727    0.591903    0.892000
6         0.141539    0.589971    0.906000
7         0.073315    0.544248    0.906000
8         0.038477    0.580940    0.904000
Total time: 03:48
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1/models/sp30k/lstm_nl4-1cyc.m
Loss and accuracy using (cls_best): [0.6642357, tensor(0.8988)]
Processing data/mldoc/ja-1/models/sp30k/lstm_nl4.m
ja-1
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ja-1/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ja-1/models/sp30k/lstm_nl4-1cyc.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ja-1/ja.dev.csv
Data lm, trn: 13500, val: 1500
Data cls, trn: 1000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁、', '▁の', '▁。', '▁に', '▁を', '▁は', '▁年', '▁が', '▁)', '▁(']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ja-1/models/sp30k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ja-1/models/sp30k/lstm_nl4-1cyc.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.804430    0.435780    0.837000
2         0.576932    0.452903    0.839000
3         0.499791    0.640789    0.806000
4         0.418024    0.610898    0.839000
5         0.259578    0.582953    0.868000
6         0.188161    0.719131    0.888000
7         0.101508    0.766175    0.877000
8         0.072221    0.795415    0.883000
Total time: 08:19
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ja-1/models/sp30k/lstm_nl4-1cyc.m
Loss and accuracy using (cls_best): [0.636261, tensor(0.9045)]
Processing data/mldoc/ru-1/models/sp30k/lstm_nl4.m
ru-1
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp30k/lstm_nl4-1cyc.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1/ru.dev.csv
Data lm, trn: 9195, val: 1021
Data cls, trn: 1000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁.', '▁в', 'а', '▁и', 'е', 'и', 'й', '▁на', 'х']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp30k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp30k/lstm_nl4-1cyc.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.821141    0.532432    0.814000
2         0.611059    0.457023    0.866000
3         0.464408    0.484035    0.870000
4         0.446560    0.477454    0.858000
5         0.299095    0.906959    0.858000
6         0.176480    0.709579    0.875000
7         0.089683    0.781081    0.876000
8         0.047031    0.772935    0.877000
Total time: 08:58
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp30k/lstm_nl4-1cyc.m
Loss and accuracy using (cls_best): [0.8528109, tensor(0.8795)]
Processing data/mldoc/zh-1/models/sp30k/lstm_nl4.m
zh-1
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp30k/lstm_nl4-1cyc.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/zh.dev.csv
Data lm, trn: 13500, val: 1500
Data cls, trn: 1000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁的', '▁。', '▁年', '▁、', '▁在', '▁一', '▁中', '▁人', '▁是']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp30k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp30k/lstm_nl4-1cyc.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.696346    0.335083    0.895000
2         0.505075    0.360600    0.906000
3         0.427076    0.462661    0.883000
4         0.351177    0.489026    0.919000
5         0.244958    0.415151    0.918000
6         0.156452    0.494367    0.926000
7         0.085807    0.471611    0.927000
8         0.046887    0.484232    0.929000
Total time: 06:41
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp30k/lstm_nl4-1cyc.m
Loss and accuracy using (cls_best): [0.50999963, tensor(0.9165)]
OrderedDict([('data/mldoc/de-1/models/sp30k/lstm_nl4-1cyc.m',
              0.9585000276565552),
             ('data/mldoc/es-1/models/sp30k/lstm_nl4-1cyc.m',
              0.9632499814033508),
             ('data/mldoc/fr-1/models/sp30k/lstm_nl4-1cyc.m',
              0.9482499957084656),
             ('data/mldoc/it-1/models/sp30k/lstm_nl4-1cyc.m',
              0.8987500071525574),
             ('data/mldoc/ja-1/models/sp30k/lstm_nl4-1cyc.m',
              0.9045000076293945),
             ('data/mldoc/ru-1/models/sp30k/lstm_nl4-1cyc.m',
              0.8794999718666077),
             ('data/mldoc/zh-1/models/sp30k/lstm_nl4-1cyc.m',
              0.9164999723434448)])
data/mldoc/de-1/models/sp30k/lstm_nl4-1cyc.m: 0.9585000276565552
data/mldoc/es-1/models/sp30k/lstm_nl4-1cyc.m: 0.9632499814033508
data/mldoc/fr-1/models/sp30k/lstm_nl4-1cyc.m: 0.9482499957084656
data/mldoc/it-1/models/sp30k/lstm_nl4-1cyc.m: 0.8987500071525574
data/mldoc/ja-1/models/sp30k/lstm_nl4-1cyc.m: 0.9045000076293945
data/mldoc/ru-1/models/sp30k/lstm_nl4-1cyc.m: 0.8794999718666077
data/mldoc/zh-1/models/sp30k/lstm_nl4-1cyc.m: 0.9164999723434448
```


### 2cycle
```bash
python -m ulmfit eval --glob="mldoc/*-1/models/sp15k/qrnn_nl4.m" --name nl4-8e-2cycle  --num-cls-epochs=8  --bs=18 --lr_sched=2cycle
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp15k/qrnn_nl4-8e-2cycle.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/es.dev.csv
Data lm, trn: 13013, val: 1445
Data cls, trn: 1000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁la', '▁.', 's', '▁en', '▁el', '▁y', '▁a', '▁que']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
/home/pczapla/anaconda3/envs/fastaiv1/lib/python3.7/site-packages/torch/utils/cpp_extension.py:152: UserWarning:

                               !! WARNING !!

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
Your compiler (c++) may be ABI-incompatible with PyTorch!
Please use a compiler that is ABI-compatible with GCC 4.9 and above.
See https://gcc.gnu.org/onlinedocs/libstdc++/manual/abi.html.

See https://gist.github.com/goldsborough/d466f43e8ffc948ff92de7486c5216d6
for instructions on how to install GCC 4.9 or higher.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                              !! WARNING !!

  warnings.warn(ABI_INCOMPATIBILITY_WARNING.format(compiler))
Loading pretrained model
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp15k/qrnn_nl4-8e-2cycle.m/info.json
2cycle training schedule
epoch     train_loss  valid_loss  accuracy
1         0.600965    0.232749    0.937000
Total time: 00:07
epoch     train_loss  valid_loss  accuracy
1         0.304946    0.202946    0.946000
2         0.326092    0.207825    0.954000
3         0.286274    0.290416    0.943000
4         0.230937    0.263474    0.950000
5         0.153293    0.293336    0.962000
6         0.080219    0.328380    0.960000
7         0.065156    0.343692    0.961000
8         0.046342    0.367162    0.962000
9         0.028884    0.396987    0.960000
10        0.034997    0.366203    0.960000
Total time: 02:27
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp15k/qrnn_nl4-8e-2cycle.m
Loss and accuracy using (cls_best): [0.35007542, tensor(0.9528)]
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp15k/qrnn_nl4-8e-2cycle.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/zh.dev.csv
Data lm, trn: 13500, val: 1500
Data cls, trn: 1000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁的', '▁。', '▁年', '▁、', '▁在', '▁一', '▁是', '▁中', '▁有']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp15k/qrnn_nl4-8e-2cycle.m/info.json
2cycle training schedule
epoch     train_loss  valid_loss  accuracy
1         0.675992    0.460828    0.836000
Total time: 00:09
epoch     train_loss  valid_loss  accuracy
1         0.439060    0.314642    0.888000
2         0.382209    0.374305    0.893000
3         0.338183    0.361669    0.911000
4         0.260323    0.431681    0.901000
5         0.146894    0.597865    0.899000
6         0.090651    0.589435    0.910000
7         0.079902    0.624589    0.918000
8         0.043067    0.558498    0.918000
9         0.022371    0.568702    0.921000
10        0.022498    0.576052    0.922000
Total time: 02:53
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp15k/qrnn_nl4-8e-2cycle.m
Loss and accuracy using (cls_best): [0.6146808, tensor(0.9150)]
OrderedDict([('data/mldoc/es-1/models/sp15k/qrnn_nl4-8e-2cycle.m',
              0.952750027179718),
             ('data/mldoc/zh-1/models/sp15k/qrnn_nl4-8e-2cycle.m',
              0.9150000214576721)])
```

### SIUNGLE 2epochs
```
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/zh.dev.csv
Data lm, trn: 13500, val: 1500
Data cls, trn: 1000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁的', '▁。', '▁年', '▁、', '▁在', '▁一', '▁是', '▁中', '▁有']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp15k/qrnn_nl4-2e-single.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.626836    0.318460    0.912000
2         0.386851    0.327937    0.918000
Total time: 00:34
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp15k/qrnn_nl4-2e-single.m
Loss and accuracy using (cls_best): [0.32642558, tensor(0.9135)]
OrderedDict([('data/mldoc/es-1/models/sp15k/qrnn_nl4-2e-single.m',
              0.9539999961853027),
             ('data/mldoc/zh-1/models/sp15k/qrnn_nl4-2e-single.m',
              0.9135000109672546)])
```
### SINGLE 4epochs
```
OrderedDict([('data/mldoc/es-1/models/sp15k/qrnn_nl4-4e-single.m',
              0.9539999961853027),
             ('data/mldoc/zh-1/models/sp15k/qrnn_nl4-4e-single.m',
              0.9210000038146973)])
```


### SINGLE 5 epochs
```
 python -m ulmfit eval --glob="mldoc/*-1/models/sp15k/qrnn_nl4.m" --name nl4-5e-single  --num-cls-epochs=5  --bs=18 --lr_sched=single
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp15k/qrnn_nl4-5e-single.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/es.dev.csv
Data lm, trn: 13013, val: 1445
Data cls, trn: 1000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁la', '▁.', 's', '▁en', '▁el', '▁y', '▁a', '▁que']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
/home/pczapla/anaconda3/envs/fastaiv1/lib/python3.7/site-packages/torch/utils/cpp_extension.py:152: UserWarning:

                               !! WARNING !!

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
Your compiler (c++) may be ABI-incompatible with PyTorch!
Please use a compiler that is ABI-compatible with GCC 4.9 and above.
See https://gcc.gnu.org/onlinedocs/libstdc++/manual/abi.html.

See https://gist.github.com/goldsborough/d466f43e8ffc948ff92de7486c5216d6
for instructions on how to install GCC 4.9 or higher.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                              !! WARNING !!

  warnings.warn(ABI_INCOMPATIBILITY_WARNING.format(compiler))
Loading pretrained model
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp15k/qrnn_nl4-5e-single.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.559638    0.250022    0.931000
2         0.366238    0.348553    0.932000
3         0.246830    0.243392    0.954000
4         0.136335    0.242888    0.960000
5         0.106189    0.254603    0.965000
Total time: 01:14
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp15k/qrnn_nl4-5e-single.m
Loss and accuracy using (cls_best): [0.24189772, tensor(0.9588)]
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp15k/qrnn_nl4-5e-single.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/zh.dev.csv
Data lm, trn: 13500, val: 1500
Data cls, trn: 1000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁的', '▁。', '▁年', '▁、', '▁在', '▁一', '▁是', '▁中', '▁有']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp15k/qrnn_nl4-5e-single.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.641053    0.334561    0.891000
2         0.486526    0.374095    0.894000
3         0.309409    0.359222    0.908000
4         0.179104    0.403468    0.920000
5         0.083530    0.414904    0.919000
Total time: 01:24
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp15k/qrnn_nl4-5e-single.m
Loss and accuracy using (cls_best): [0.44048822, tensor(0.9110)]
OrderedDict([('data/mldoc/es-1/models/sp15k/qrnn_nl4-5e-single.m',
              0.9587500095367432),
             ('data/mldoc/zh-1/models/sp15k/qrnn_nl4-5e-single.m',
              0.9110000133514404)])
```


#### SINGLE 11epochs
```              
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp15k/qrnn_nl4-8e-single.m/info.json
Starting classifier training
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.676591    0.205210    0.947000
2         0.402020    0.461279    0.912000
3         0.287975    0.496294    0.921000
4         0.258515    0.243489    0.954000
5         0.219352    0.274136    0.949000
6         0.149339    0.352294    0.956000
7         0.092821    0.378696    0.962000
8         0.055485    0.367379    0.963000
9         0.042695    0.367151    0.964000
10        0.034858    0.386749    0.961000
11        0.021245    0.392899    0.963000
Total time: 02:38

Loss and accuracy using (cls_last): [0.4098273, tensor(0.9595)]
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp15k/qrnn_nl4-8e-single.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/zh.dev.csv
Data lm, trn: 13500, val: 1500
Data cls, trn: 1000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁的', '▁。', '▁年', '▁、', '▁在', '▁一', '▁是', '▁中', '▁有']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp15k/qrnn_nl4-8e-single.m/info.json
Starting classifier training
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.710727    0.355552    0.895000
2         0.523836    0.328691    0.895000
3         0.408339    0.440722    0.894000
4         0.326642    0.422076    0.909000
5         0.217328    0.539624    0.906000
6         0.156753    0.583433    0.912000
7         0.102770    0.549827    0.921000
8         0.053252    0.533528    0.928000
9         0.032845    0.568053    0.927000
10        0.022289    0.604236    0.926000
11        0.015289    0.587667    0.929000
Total time: 03:12
OrderedDict([('data/mldoc/es-1/models/sp15k/qrnn_nl4-8e-single.m',
              0.9595000147819519),
             ('data/mldoc/zh-1/models/sp15k/qrnn_nl4-8e-single.m',
              0.9202499985694885)])
           
```

## Limiit to 100 examples
```
python -m ulmfit eval --glob="mldoc/*-1/models/sp30k/lstm_nl4.m" --name nl4-100e8 --cuda-id=1 --limit=100 --num-cls-epochs=8
{
    'data/mldoc/it-1/models/sp30k/lstm_nl4-100e8.m': 0.7799999713897705,
    'data/mldoc/de-1/models/sp30k/lstm_nl4-100e8.m': 0.9135000109672546,
    'data/mldoc/ja-1/models/sp30k/lstm_nl4-100e8.m': 0.7112500071525574,
    'data/mldoc/fr-1/models/sp30k/lstm_nl4-100e8.m': 0.8877500295639038,
    'data/mldoc/ru-1/models/sp30k/lstm_nl4-100e8.m': 0.722000002861023,
    'data/mldoc/es-1/models/sp30k/lstm_nl4-100e8.m': 0.8169999718666077
}
```


## Noise  

```
noise=0.13
lang=de
python -m ulmfit eval --glob="mldoc/${lang}-1/models/sp30k/lstm_nl4.m" --name nl4-noise --cuda-id=1 --num-cls-epochs=2 --noise=${noise}
{'data/mldoc/de-1/models/sp30k/lstm_nl4-noise.m': 0.9449999928474426}

noise=0.18
lang=es
python -m ulmfit eval --glob="mldoc/${lang}-1/models/sp30k/lstm_nl4.m" --name nl4-noise --cuda-id=1 --num-cls-epochs=2 --noise=${noise} 
{'data/mldoc/es-1/models/sp30k/lstm_nl4-noise.m': 0.9312499761581421}

noise=0.18
lang=fr
python -m ulmfit eval --glob="mldoc/${lang}-1/models/sp30k/lstm_nl4.m" --name nl4-noise --cuda-id=1 --num-cls-epochs=2 --noise=${noise} 
{'data/mldoc/fr-1/models/sp30k/lstm_nl4-noise.m': 0.9049999713897705}

noise=0.27
lang=it
python -m ulmfit eval --glob="mldoc/${lang}-1/models/sp30k/lstm_nl4.m" --name nl4-noise --cuda-id=1 --num-cls-epochs=2 --noise=${noise} 
{'data/mldoc/it-1/models/sp30k/lstm_nl4-noise.m': 0.8372499942779541}

noise=0.4
lang=ja
python -m ulmfit eval --glob="mldoc/${lang}-1/models/sp30k/lstm_nl4.m" --name nl4-noise --cuda-id=1 --num-cls-epochs=2 --noise=${noise} 
{'data/mldoc/ja-1/models/sp30k/lstm_nl4-noise.m': 0.7472500205039978

noise=0.32
lang=ru
python -m ulmfit eval --glob="mldoc/${lang}-1/models/sp30k/lstm_nl4.m" --name nl4-noise --cuda-id=1 --num-cls-epochs=2 --noise=${noise} 
{'data/mldoc/ru-1/models/sp30k/lstm_nl4-noise.m': 0.7567499876022339}

noise=0.28
lang=zh
python -m ulmfit eval --glob="mldoc/${lang}-1/models/sp30k/lstm_nl4.m" --name nl4-noise --cuda-id=1 --num-cls-epochs=2 --noise=${noise} 
```



## Fix the sentence piece tokenizer
```
python -m ulmfit eval --glob="mldoc/*-1/models/bsp30k/lstm_nl4.m" --name nl4 --cuda-id=0
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1/models/sp30k/lstm_nl4.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1/de.dev.csv
Running tokenization...
Saving tokenized: cls.trn 13500, cls.val 1500
Running tokenization...
Saving tokenized: cls.trn 1000, cls.val 1000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁.', '▁,', '▁der', '▁die', '▁und', '▁in', 'en', "▁&'", 's', '-']
Training args:  {'tie_weights': True, 'clip': 0.12, 'bptt': 70, 'pretrained_fnames': [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1/models/bsp30k/lstm_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1/models/bsp30k/lstm_nl4.m/../itos')], 'pretrained_model': None, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  [0.25 0.1  0.2  0.02 0.15]
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1/models/sp30k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1/models/sp30k/lstm_nl4.m/info.json
Starting classifier training
epoch     train_loss  valid_loss  accuracy
1         0.480367    0.263987    0.930000
epoch     train_loss  valid_loss  accuracy
1         0.340463    0.209449    0.940000
epoch     train_loss  valid_loss  accuracy
1         0.235290    0.214566    0.952000
epoch     train_loss  valid_loss  accuracy
1         0.169588    0.217762    0.952000
2         0.175439    0.215570    0.946000
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1/models/sp30k/lstm_nl4.m
Loss and accuracy using (cls_best): [0.15467079, tensor(0.9563)]
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp30k/lstm_nl4.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/es.dev.csv
Running tokenization...
Saving tokenized: cls.trn 13013, cls.val 1445
Running tokenization...
Saving tokenized: cls.trn 1000, cls.val 1000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁la', '▁.', '▁en', '▁el', '▁y', 's', '▁a', '▁que']
Training args:  {'tie_weights': True, 'clip': 0.12, 'bptt': 70, 'pretrained_fnames': [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/bsp30k/lstm_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/bsp30k/lstm_nl4.m/../itos')], 'pretrained_model': None, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  [0.25 0.1  0.2  0.02 0.15]
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp30k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp30k/lstm_nl4.m/info.json
Starting classifier training
epoch     train_loss  valid_loss  accuracy
1         0.609305    0.343920    0.914000
epoch     train_loss  valid_loss  accuracy
1         0.380833    0.204708    0.947000
epoch     train_loss  valid_loss  accuracy
1         0.311563    0.210382    0.943000
epoch     train_loss  valid_loss  accuracy
1         0.266655    0.192309    0.955000
2         0.235553    0.183249    0.953000
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp30k/lstm_nl4.m
Loss and accuracy using (cls_best): [0.19250762, tensor(0.9427)]
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/models/sp30k/lstm_nl4.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/fr.dev.csv
Running tokenization...
Saving tokenized: cls.trn 13500, cls.val 1500
Running tokenization...
Saving tokenized: cls.trn 1000, cls.val 1000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁.', "'", 's', '▁la', '▁le', '▁et', '▁l', '▁à']
Training args:  {'tie_weights': True, 'clip': 0.12, 'bptt': 70, 'pretrained_fnames': [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/models/bsp30k/lstm_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/models/bsp30k/lstm_nl4.m/../itos')], 'pretrained_model': None, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  [0.25 0.1  0.2  0.02 0.15]
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/models/sp30k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/models/sp30k/lstm_nl4.m/info.json
Starting classifier training
epoch     train_loss  valid_loss  accuracy
1         0.527443    0.342664    0.901000
epoch     train_loss  valid_loss  accuracy
1         0.359752    0.203693    0.936000
epoch     train_loss  valid_loss  accuracy
1         0.272514    0.188933    0.938000
epoch     train_loss  valid_loss  accuracy
1         0.191520    0.183208    0.938000
2         0.201412    0.178796    0.942000
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/models/sp30k/lstm_nl4.m
Loss and accuracy using (cls_best): [0.18170285, tensor(0.9420)]
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1/models/sp30k/lstm_nl4.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1/it.dev.csv
Running tokenization...
Saving tokenized: cls.trn 13500, cls.val 1500
Running tokenization...
Saving tokenized: cls.trn 1000, cls.val 1000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁.', '▁di', "▁&'", "'", '▁e', '▁il', '▁la', 'e', '▁in']
Training args:  {'tie_weights': True, 'clip': 0.12, 'bptt': 70, 'pretrained_fnames': [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1/models/bsp30k/lstm_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1/models/bsp30k/lstm_nl4.m/../itos')], 'pretrained_model': None, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  [0.25 0.1  0.2  0.02 0.15]
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1/models/sp30k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1/models/sp30k/lstm_nl4.m/info.json
Starting classifier training
epoch     train_loss  valid_loss  accuracy
1         0.754286    0.599928    0.783000
epoch     train_loss  valid_loss  accuracy
1         0.501399    0.379078    0.851000
epoch     train_loss  valid_loss  accuracy
1         0.408768    0.345188    0.867000
epoch     train_loss  valid_loss  accuracy
1         0.324877    0.335110    0.872000
2         0.291118    0.336596    0.879000
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1/models/sp30k/lstm_nl4.m
Loss and accuracy using (cls_best): [0.33244577, tensor(0.8852)]
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ja-1/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ja-1/models/sp30k/lstm_nl4.m
Evaluating previously trained model
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ja-1/ja.dev.csv
Tokenized data loaded, lm.trn 13500, lm.val 1500
Tokenized data loaded, cls.trn 1000, cls.val 1000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁、', '▁の', '▁。', '▁に', '▁を', '▁は', '▁年', '▁が', '▁)', '▁(']
Loss and accuracy using (cls_last): [0.330675, tensor(0.8873)]
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp30k/lstm_nl4.m
Evaluating previously trained model
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1/ru.dev.csv
Tokenized data loaded, lm.trn 9195, lm.val 1021
Tokenized data loaded, cls.trn 1000, cls.val 1000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁.', '▁в', 'а', '▁и', 'е', 'и', 'й', '▁на', 'х']
Loss and accuracy using (cls_last): [0.3987146, tensor(0.8680)]
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp30k/lstm_nl4.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/zh.dev.csv
Running tokenization...
Saving tokenized: cls.trn 13500, cls.val 1500
Running tokenization...
Saving tokenized: cls.trn 1000, cls.val 1000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁的', '▁。', '▁年', '▁、', '▁在', '▁一', '▁中', '▁人', '▁是']
Training args:  {'tie_weights': True, 'clip': 0.12, 'bptt': 70, 'pretrained_fnames': [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/bsp30k/lstm_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/bsp30k/lstm_nl4.m/../itos')], 'pretrained_model': None, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  [0.25 0.1  0.2  0.02 0.15]
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp30k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp30k/lstm_nl4.m/info.json
Starting classifier training
epoch     train_loss  valid_loss  accuracy
1         0.612067    0.487164    0.860000
epoch     train_loss  valid_loss  accuracy
1         0.441519    0.343330    0.886000
epoch     train_loss  valid_loss  accuracy
1         0.384575    0.318655    0.897000
epoch     train_loss  valid_loss  accuracy
1         0.298987    0.305157    0.899000
2         0.293190    0.312572    0.901000
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp30k/lstm_nl4.m
Loss and accuracy using (cls_best): [0.28432375, tensor(0.9047)]
OrderedDict([('data/mldoc/de-1/models/sp30k/lstm_nl4.m', 0.956250011920929),
             ('data/mldoc/es-1/models/sp30k/lstm_nl4.m', 0.9427499771118164),
             ('data/mldoc/fr-1/models/sp30k/lstm_nl4.m', 0.9419999718666077),
             ('data/mldoc/it-1/models/sp30k/lstm_nl4.m', 0.8852499723434448),
             ('data/mldoc/ja-1/models/sp30k/lstm_nl4.m', 0.8872500061988831),
             ('data/mldoc/ru-1/models/sp30k/lstm_nl4.m', 0.8679999709129333),
             ('data/mldoc/zh-1/models/sp30k/lstm_nl4.m', 0.9047499895095825)])
             
--- Additonal run on ru
python -m ulmfit eval --glob="mldoc/ru-1/models/bsp30k/lstm_nl4.m" --name nl4 --cuda-id=0                                                                  ✘ 1
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp30k/lstm_nl4.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1/ru.dev.csv
Running tokenization...
Saving tokenized: cls.trn 9195, cls.val 1021
Running tokenization...
Saving tokenized: cls.trn 1000, cls.val 1000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁.', '▁в', 'а', '▁и', 'е', 'и', 'й', '▁на', 'х']
Training args:  {'tie_weights': True, 'clip': 0.12, 'bptt': 70, 'pretrained_fnames': [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/bsp30k/lstm_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/bsp30k/lstm_nl4.m/../itos')], 'pretrained_model': None, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  [0.25 0.1  0.2  0.02 0.15]
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp30k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp30k/lstm_nl4.m/info.json
Starting classifier training
epoch     train_loss  valid_loss  accuracy
1         0.814752    0.555627    0.808000
epoch     train_loss  valid_loss  accuracy
1         0.659532    0.427135    0.855000
epoch     train_loss  valid_loss  accuracy
1         0.508609    0.427321    0.851000
epoch     train_loss  valid_loss  accuracy
1         0.440108    0.396991    0.872000
2         0.440024    0.388976    0.866000
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp30k/lstm_nl4.m
Loss and accuracy using (cls_best): [0.3959306, tensor(0.8685)]
----

----
 python -m ulmfit eval --glob="mldoc/es-1/models/bsp30k/lstm_nl4.m" --name nl4-2nd --cuda-id=0
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp30k/lstm_nl4-2nd.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/es.dev.csv
Tokenized data loaded, lm.trn 13013, lm.val 1445
Tokenized data loaded, cls.trn 1000, cls.val 1000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁la', '▁.', '▁en', '▁el', '▁y', 's', '▁a', '▁que']
Training args:  {'tie_weights': True, 'clip': 0.12, 'bptt': 70, 'pretrained_fnames': [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/bsp30k/lstm_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/bsp30k/lstm_nl4.m/../itos')], 'pretrained_model': None, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  [0.25 0.1  0.2  0.02 0.15]
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp30k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp30k/lstm_nl4-2nd.m/info.json
Starting classifier training
epoch     train_loss  valid_loss  accuracy
1         0.654984    0.453887    0.818000
epoch     train_loss  valid_loss  accuracy
1         0.451552    0.220058    0.934000
epoch     train_loss  valid_loss  accuracy
1         0.323974    0.193342    0.949000
epoch     train_loss  valid_loss  accuracy
1         0.244755    0.201804    0.945000
2         0.237175    0.183736    0.953000
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp30k/lstm_nl4-2nd.m
Loss and accuracy using (cls_best): [0.18296617, tensor(0.9480)]
OrderedDict([('data/mldoc/es-1/models/sp30k/lstm_nl4-2nd.m',
              0.9480000138282776)])
----

```

### LIMIT LOgs
```
python -m ulmfit eval --name nl4-100e8 --cuda-id=1 --limit=100 --num-cls-epochs=8                                                                         ✘ 130
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1/models/sp30k/lstm_nl4-100e8.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1/it.dev.csv
Tokenized data loaded, lm.trn 13500, lm.val 1500
Limiting data set to: 100
Tokenized data loaded, cls.trn 100, cls.val 100
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁.', '▁di', "▁&'", "'", '▁e', '▁il', '▁la', 'e', '▁in']
Training args:  {'tie_weights': True, 'clip': 0.12, 'bptt': 70, 'pretrained_fnames': [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1/models/sp30k/lstm_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1/models/sp30k/lstm_nl4.m/../itos')], 'pretrained_model': None, 'drop_mult': 0.3} dps:  [0.25 0.1  0.2  0.02 0.15]
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1/models/sp30k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1/models/sp30k/lstm_nl4-100e8.m/info.json
Starting classifier training
epoch     train_loss  valid_loss  accuracy
1         1.214805    1.382632    0.280000
epoch     train_loss  valid_loss  accuracy
1         0.977314    1.269534    0.450000
epoch     train_loss  valid_loss  accuracy
1         0.856274    1.223441    0.530000
epoch     train_loss  valid_loss  accuracy
1         0.718223    1.188048    0.620000
2         0.735718    1.130525    0.730000
3         0.730894    1.069027    0.710000
4         0.715334    1.015253    0.710000
5         0.716080    0.965223    0.720000
6         0.695554    0.918456    0.730000
7         0.689949    0.892840    0.730000
8         0.675208    0.876222    0.720000
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1/models/sp30k/lstm_nl4-100e8.m
Loss and accuracy using (cls_best): [0.7090041, tensor(0.7800)]
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1/models/sp30k/lstm_nl4-100e8.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1/de.dev.csv
Tokenized data loaded, lm.trn 13500, lm.val 1500
Limiting data set to: 100
Running tokenization...
Saving tokenized: cls.trn 100, cls.val 100
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁.', '▁,', '▁der', '▁die', '▁und', '▁in', 'en', "▁&'", 's', '-']
Training args:  {'tie_weights': True, 'clip': 0.12, 'bptt': 70, 'pretrained_fnames': [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1/models/sp30k/lstm_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1/models/sp30k/lstm_nl4.m/../itos')], 'pretrained_model': None, 'drop_mult': 0.3} dps:  [0.25 0.1  0.2  0.02 0.15]
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1/models/sp30k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1/models/sp30k/lstm_nl4-100e8.m/info.json
Starting classifier training
epoch     train_loss  valid_loss  accuracy
1         1.141527    1.328262    0.280000
epoch     train_loss  valid_loss  accuracy
1         0.703434    1.170250    0.510000
epoch     train_loss  valid_loss  accuracy
1         0.568693    1.051980    0.780000
epoch     train_loss  valid_loss  accuracy
1         0.455238    0.990438    0.800000
2         0.475659    0.928943    0.850000
3         0.477652    0.848537    0.920000
4         0.455583    0.769415    0.930000
5         0.450824    0.690618    0.930000
6         0.443699    0.633900    0.940000
7         0.430881    0.563667    0.950000
8         0.419999    0.524655    0.950000
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1/models/sp30k/lstm_nl4-100e8.m
Loss and accuracy using (cls_best): [0.45835665, tensor(0.9135)]
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ja-1/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ja-1/models/sp30k/lstm_nl4-100e8.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ja-1/ja.dev.csv
Tokenized data loaded, lm.trn 13500, lm.val 1500
Limiting data set to: 100
Tokenized data loaded, cls.trn 100, cls.val 100
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁、', '▁の', '▁。', '▁に', '▁を', '▁は', '▁年', '▁が', '▁)', '▁(']
Training args:  {'tie_weights': True, 'clip': 0.12, 'bptt': 70, 'pretrained_fnames': [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ja-1/models/sp30k/lstm_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ja-1/models/sp30k/lstm_nl4.m/../itos')], 'pretrained_model': None, 'drop_mult': 0.3} dps:  [0.25 0.1  0.2  0.02 0.15]
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ja-1/models/sp30k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ja-1/models/sp30k/lstm_nl4-100e8.m/info.json
Starting classifier training
epoch     train_loss  valid_loss  accuracy
1         1.342269    1.399389    0.230000
epoch     train_loss  valid_loss  accuracy
1         0.957341    1.344665    0.280000
epoch     train_loss  valid_loss  accuracy
1         0.881869    1.301798    0.450000
epoch     train_loss  valid_loss  accuracy
1         0.887575    1.280226    0.440000
2         0.835731    1.257639    0.450000
3         0.813987    1.219512    0.510000
4         0.792665    1.181309    0.520000
5         0.785690    1.151372    0.510000
6         0.784095    1.152232    0.500000
7         0.768115    1.133895    0.520000
8         0.769684    1.124231    0.530000
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ja-1/models/sp30k/lstm_nl4-100e8.m
Loss and accuracy using (cls_best): [0.8863698, tensor(0.7113)]
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/models/sp30k/lstm_nl4-100e8.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/fr.dev.csv
Tokenized data loaded, lm.trn 13500, lm.val 1500
Limiting data set to: 100
Running tokenization...
Saving tokenized: cls.trn 100, cls.val 100
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁.', "'", 's', '▁la', '▁le', '▁et', '▁l', '▁à']
Training args:  {'tie_weights': True, 'clip': 0.12, 'bptt': 70, 'pretrained_fnames': [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/models/sp30k/lstm_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/models/sp30k/lstm_nl4.m/../itos')], 'pretrained_model': None, 'drop_mult': 0.3} dps:  [0.25 0.1  0.2  0.02 0.15]
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/models/sp30k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/models/sp30k/lstm_nl4-100e8.m/info.json
Starting classifier training
epoch     train_loss  valid_loss  accuracy
1         1.220506    1.413276    0.200000
epoch     train_loss  valid_loss  accuracy
1         0.791777    1.306999    0.290000
epoch     train_loss  valid_loss  accuracy
1         0.572241    1.190053    0.580000
epoch     train_loss  valid_loss  accuracy
1         0.502800    1.130456    0.710000
2         0.515115    1.056434    0.770000
3         0.522720    0.974482    0.780000
4         0.518296    0.881002    0.840000
5         0.496588    0.825646    0.880000
6         0.490416    0.771587    0.860000
7         0.497172    0.722874    0.850000
8         0.491894    0.682278    0.850000
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/models/sp30k/lstm_nl4-100e8.m
Loss and accuracy using (cls_best): [0.5428351, tensor(0.8878)]
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp30k/lstm_nl4-100e8.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1/ru.dev.csv
Tokenized data loaded, lm.trn 9195, lm.val 1021
Limiting data set to: 100
Running tokenization...
Saving tokenized: cls.trn 100, cls.val 100
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁.', '▁в', 'а', '▁и', 'е', 'и', 'й', '▁на', 'х']
Training args:  {'tie_weights': True, 'clip': 0.12, 'bptt': 70, 'pretrained_fnames': [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp30k/lstm_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp30k/lstm_nl4.m/../itos')], 'pretrained_model': None, 'drop_mult': 0.3} dps:  [0.25 0.1  0.2  0.02 0.15]
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp30k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp30k/lstm_nl4-100e8.m/info.json
Starting classifier training
epoch     train_loss  valid_loss  accuracy
1         1.367201    1.409767    0.240000
epoch     train_loss  valid_loss  accuracy
1         1.099071    1.320811    0.330000
epoch     train_loss  valid_loss  accuracy
1         0.875845    1.253172    0.410000
epoch     train_loss  valid_loss  accuracy
1         0.775657    1.215067    0.580000
2         0.774420    1.171324    0.660000
3         0.766028    1.118901    0.680000
4         0.744478    1.074021    0.680000
5         0.738797    1.033736    0.660000
6         0.733380    0.997304    0.660000
7         0.723470    0.977280    0.670000
8         0.710699    0.953586    0.640000
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp30k/lstm_nl4-100e8.m
Loss and accuracy using (cls_best): [0.8535175, tensor(0.7220)]
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp30k/lstm_nl4-100e8.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/es.dev.csv
Tokenized data loaded, lm.trn 13013, lm.val 1445
Limiting data set to: 100
Running tokenization...
Saving tokenized: cls.trn 100, cls.val 100
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁la', '▁.', '▁en', '▁el', '▁y', 's', '▁a', '▁que']
Training args:  {'tie_weights': True, 'clip': 0.12, 'bptt': 70, 'pretrained_fnames': [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp30k/lstm_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp30k/lstm_nl4.m/../itos')], 'pretrained_model': None, 'drop_mult': 0.3} dps:  [0.25 0.1  0.2  0.02 0.15]
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp30k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp30k/lstm_nl4-100e8.m/info.json
Starting classifier training
epoch     train_loss  valid_loss  accuracy
1         1.142170    1.330161    0.300000
epoch     train_loss  valid_loss  accuracy
1         0.767807    1.212253    0.420000
epoch     train_loss  valid_loss  accuracy
1         0.636803    1.099303    0.540000
epoch     train_loss  valid_loss  accuracy
1         0.584241    0.997207    0.610000
2         0.578480    0.907674    0.710000
3         0.548451    0.830268    0.730000
4         0.535560    0.762040    0.750000
5         0.522172    0.746566    0.740000
6         0.506584    0.676038    0.770000
7         0.493665    0.651112    0.770000
8         0.493031    0.621689    0.770000
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp30k/lstm_nl4-100e8.m
Loss and accuracy using (cls_best): [0.54911107, tensor(0.8170)]
{'data/mldoc/it-1/models/sp30k/lstm_nl4-100e8.m': 0.7799999713897705, 'data/mldoc/de-1/models/sp30k/lstm_nl4-100e8.m': 0.9135000109672546, 'data/mldoc/ja-1/models/sp30k/lstm_nl4-100e8.m': 0.7112500071525574, 'data/mldoc/fr-1/models/sp30k/lstm_nl4-100e8.m': 0.8877500295639038, 'data/mldoc/ru-1/models/sp30k/lstm_nl4-100e8.m': 0.722000002861023, 'data/mldoc/es-1/models/sp30k/lstm_nl4-100e8.m': 0.8169999718666077}

python -m ulmfit eval --glob="mldoc/es-1/models/sp30k/lstm_nl4.m" --name nl4-100-2nd --cuda-id=1 --num-cls-epochs=8 --limit=100
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp30k/lstm_nl4-100-2nd.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/es.dev.csv
Tokenized data loaded, lm.trn 13013, lm.val 1445
Limiting data set to: 100
Tokenized data loaded, cls.trn 100, cls.val 100
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁la', '▁.', '▁en', '▁el', '▁y', 's', '▁a', '▁que']
Training args:  {'tie_weights': True, 'clip': 0.12, 'bptt': 70, 'pretrained_fnames': [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp30k/lstm_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp30k/lstm_nl4.m/../itos')], 'pretrained_model': None, 'drop_mult': 0.3} dps:  [0.25 0.1  0.2  0.02 0.15]
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp30k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp30k/lstm_nl4-100-2nd.m/info.json
Starting classifier training
epoch     train_loss  valid_loss  accuracy
1         1.243127    1.354496    0.290000
epoch     train_loss  valid_loss  accuracy
1         0.840900    1.213333    0.460000
epoch     train_loss  valid_loss  accuracy
1         0.656407    1.055138    0.750000
epoch     train_loss  valid_loss  accuracy
1         0.558013    0.983957    0.780000
2         0.554590    0.915244    0.750000
3         0.536740    0.840074    0.770000
4         0.521179    0.759908    0.790000
5         0.515218    0.692961    0.810000
6         0.500587    0.639504    0.810000
7         0.486596    0.593410    0.840000
8         0.472318    0.550126    0.830000
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp30k/lstm_nl4-100-2nd.m
Loss and accuracy using (cls_best): [0.5382241, tensor(0.8332)]
{'data/mldoc/es-1/models/sp30k/lstm_nl4-100-2nd.m': 0.8332499861717224}

```