

### MLDoc laser zero shoot 10k
Loss and accuracy using (cls_best): [0.58419716, tensor(0.8150)] [0.6386394, tensor(0.7850)]
                                                name  tst_accuracy  tst_loss  val_accuracy  val_loss
0  data/mldoc/de-10-laser-en1/models/sp15k/qrnn_r...       0.90550  0.407956         0.917  0.378366
1  data/mldoc/es-10-laser-en1/models/sp15k/qrnn_r...       0.69725  1.371628         0.747  1.040883
2  data/mldoc/fr-10-laser-en1/models/sp15k/qrnn_r...       0.87350  0.533766         0.882  0.488216
3  data/mldoc/it-10-laser-en1/models/sp15k/qrnn_r...       0.72750  1.168527         0.804  1.180750
4  data/mldoc/ja-10-laser-en1/models/sp15k/qrnn_r...       0.67550  1.941633         0.780  0.991396
5  data/mldoc/ru-10-laser-en1/models/sp15k/qrnn_r...       0.63675  2.139875         0.822  0.765918
6  data/mldoc/zh-10-laser-en1/models/sp15k/qrnn_r...       0.81500  0.584197         0.785  0.638639
ds    de-10-laser-  es-10-laser-  fr-10-laser-  it-10-laser-  ja-10-laser-  ru-10-laser-  zh-10-laser-
best         90.55         69.73         87.35         72.75         67.55         63.67          81.5
max          90.55         69.73         87.35         72.75         67.55         63.67          81.5
avg          90.55         69.73         87.35         72.75         67.55         63.67          81.5

```

Starting classifier from random weights
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.087360    1.237505    0.515000
2         0.810132    1.667991    0.545000
3         0.779427    1.003253    0.679000
4         0.662206    2.510274    0.800000
5         0.604669    2.394876    0.718000
6         0.501023    0.866812    0.810000
7         0.398415    0.639844    0.818000
8         0.448071    0.922495    0.822000
Total time: 11:52
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-10-laser-en1/models/sp15k/qrnn_rnd_nl4.m
```

```
python -m ulmfit eval --glob="mldoc/*-1/models/sp15k/qrnn_nl4.m" --dataset_template='${lang}-10-laser-en1' --name rnd_nl4 --num-cls-epochs=8 --label-smoothing-eps=0.1 --lr_sched=1cycle --bs=18 --random-init=True
python -m ulmfit eval --glob="mldoc/*-1/models/sp15k/qrnn_nl4.m" --dataset_template='${lang}-10-laser-en1' --name rnd_nl4 --num-cls-epochs=8 --label-smoothing-eps=0.1 --lr_sched=1cycle --bs=18 --random-init=True
Processing data/mldoc/de-1/models/sp15k/qrnn_nl4.m
de-10-laser-en1
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10-laser-en1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10-laser-en1/models/sp15k/qrnn_rnd_nl4.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10-laser-en1/de.dev.csv
Data lm, trn: 13500, val: 1500
Data cls, trn: 10000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁.', '▁,', '▁der', '▁die', 'en', '▁und', 's', '▁in', 'er', "▁&'"]
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
Starting classifier from random weights
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.864752    0.760891    0.844000
2         0.734504    1.077554    0.676000
3         0.681645    0.703327    0.885000
4         0.670696    0.779010    0.898000
5         0.620256    0.664871    0.910000
6         0.591837    1.077103    0.915000
7         0.550238    0.607863    0.913000
8         0.543874    0.607274    0.918000
Total time: 19:55
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10-laser-en1/models/sp15k/qrnn_rnd_nl4.m
Loss and accuracy using (cls_best): [0.35624045, tensor(0.9053)]
Processing data/mldoc/en-1/models/sp15k/qrnn_nl4.m
en-10-laser-en1
Processing data/mldoc/es-1/models/sp15k/qrnn_nl4.m
es-10-laser-en1
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10-laser-en1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10-laser-en1/models/sp15k/qrnn_rnd_nl4.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10-laser-en1/es.dev.csv
Data lm, trn: 13013, val: 1445
Data cls, trn: 9458, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁la', '▁.', 's', '▁en', '▁el', '▁y', '▁a', '▁que']
Starting classifier from random weights
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.755512    1.360725    0.500000
2         0.760655    1.362604    0.399000
3         0.760876    20.748863   0.607000
4         0.730208    8.120344    0.369000
5         0.707735    1.149775    0.700000
6         0.679102    1.010318    0.746000
7         0.639611    3.087066    0.713000
8         0.608591    1.327793    0.750000
Total time: 11:38
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10-laser-en1/models/sp15k/qrnn_rnd_nl4.m
Loss and accuracy using (cls_best): [1.3680531, tensor(0.6975)]
Processing data/mldoc/fr-1/models/sp15k/qrnn_nl4.m
fr-10-laser-en1
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-10-laser-en1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-10-laser-en1/models/sp15k/qrnn_rnd_nl4.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-10-laser-en1/fr.dev.csv
Data lm, trn: 13500, val: 1500
Data cls, trn: 10000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', 's', '▁.', "'", '▁la', '▁le', '▁et', '▁l', '▁à']
Starting classifier from random weights
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.922996    1.406875    0.519000
2         0.833284    1.172545    0.640000
3         0.749922    0.697733    0.863000
4         0.724680    0.735842    0.837000
5         0.652541    0.679455    0.876000
6         0.641541    0.671731    0.868000
7         0.577571    0.734958    0.868000
8         0.579186    0.703696    0.883000
Total time: 19:15
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-10-laser-en1/models/sp15k/qrnn_rnd_nl4.m
Loss and accuracy using (cls_best): [0.47713563, tensor(0.8740)]
Processing data/mldoc/it-1/models/sp15k/qrnn_nl4.m
it-10-laser-en1
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-10-laser-en1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-10-laser-en1/models/sp15k/qrnn_rnd_nl4.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-10-laser-en1/it.dev.csv
Data lm, trn: 13500, val: 1500
Data cls, trn: 10000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁.', '▁di', '▁e', "▁&'", "'", '▁il', '▁la', '▁in', 'e']
Starting classifier from random weights
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.991065    1.602189    0.381000
2         0.935880    0.888808    0.734000
3         0.860670    0.868564    0.781000
4         0.818734    0.945302    0.791000
5         0.751467    3.113552    0.808000
6         0.687606    0.921033    0.795000
7         0.677044    1.222023    0.807000
8         0.645511    1.418593    0.805000
Total time: 11:44
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-10-laser-en1/models/sp15k/qrnn_rnd_nl4.m
Loss and accuracy using (cls_best): [1.1276722, tensor(0.7272)]
Processing data/mldoc/ja-1/models/sp15k/qrnn_nl4.m
ja-10-laser-en1
Processing data/mldoc/zh-1/models/sp15k/qrnn_nl4.m
zh-10-laser-en1
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-10-laser-en1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-10-laser-en1/models/sp15k/qrnn_rnd_nl4.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-10-laser-en1/zh.dev.csv
Data lm, trn: 13500, val: 1500
Data cls, trn: 10000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁的', '▁。', '▁年', '▁、', '▁在', '▁一', '▁是', '▁中', '▁有']
Starting classifier from random weights
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.920411    1.159853    0.629000
2         0.937020    1.371089    0.527000
3         0.892036    3.091183    0.615000
4         0.839919    0.939323    0.724000
5         0.797184    1.174206    0.735000
6         0.774195    0.914951    0.733000
7         0.744524    0.875888    0.762000
8         0.721782    0.825969    0.788000
Total time: 19:53
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-10-laser-en1/models/sp15k/qrnn_rnd_nl4.m
Loss and accuracy using (cls_best): [0.54084456, tensor(0.8145)]
OrderedDict([('data/mldoc/de-10-laser-en1/models/sp15k/qrnn_rnd_nl4.m',
              0.9052500128746033),
             ('data/mldoc/es-10-laser-en1/models/sp15k/qrnn_rnd_nl4.m',
              0.6974999904632568),
             ('data/mldoc/fr-10-laser-en1/models/sp15k/qrnn_rnd_nl4.m',
              0.8740000128746033),
             ('data/mldoc/it-10-laser-en1/models/sp15k/qrnn_rnd_nl4.m',
              0.7272499799728394),
             ('data/mldoc/zh-10-laser-en1/models/sp15k/qrnn_rnd_nl4.m',
              0.8144999742507935)])
data/mldoc/de-10-laser-en1/models/sp15k/qrnn_rnd_nl4.m: 0.9052500128746033
data/mldoc/es-10-laser-en1/models/sp15k/qrnn_rnd_nl4.m: 0.6974999904632568
data/mldoc/fr-10-laser-en1/models/sp15k/qrnn_rnd_nl4.m: 0.8740000128746033
data/mldoc/it-10-laser-en1/models/sp15k/qrnn_rnd_nl4.m: 0.7272499799728394
data/mldoc/zh-10-laser-en1/models/sp15k/qrnn_rnd_nl4.m: 0.8144999742507935
```


### MLDoc laser zero shoot 1k

data/mldoc/de-1-laser-en1/models/sp15k/qrnn_nl4-rnd.m: 0.777999997138977
data/mldoc/es-1-laser-en1/models/sp15k/qrnn_nl4-rnd.m: 0.7049999833106995
data/mldoc/fr-1-laser-en1/models/sp15k/qrnn_nl4-rnd.m: 0.7565000057220459
data/mldoc/it-1-laser-en1/models/sp15k/qrnn_nl4-rnd.m: 0.6852499842643738
data/mldoc/ru-1-laser-en1/models/sp15k/qrnn_nl4-rnd.m: 0.6137499809265137
data/mldoc/zh-1-laser-en1/models/sp15k/qrnn_nl4-rnd.m: 0.7919999957084656
```
python -m ulmfit eval --glob="mldoc/*-1/models/sp15k/qrnn_nl4.m" --dataset_template='${lang}-1-laser-en1' --name nl4-rnd --num-cls-epochs=8 --label-smoothing-eps=0.1 --lr_sched=1cycle --bs=18 --random-init=True
Processing data/mldoc/de-1/models/sp15k/qrnn_nl4.m
de-1-laser-en1
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1-laser-en1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1-laser-en1/models/sp15k/qrnn_nl4-rnd.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1-laser-en1/de.dev.csv
Data lm, trn: 13500, val: 1500
Data cls, trn: 1000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁.', '▁,', '▁der', '▁die', 'en', '▁und', 's', '▁in', 'er', "▁&'"]
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
Starting classifier from random weights
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.409097    1.357696    0.412000
2         1.270668    1.920674    0.259000
3         1.121176    1.099281    0.539000
4         1.034688    2.448050    0.263000
5         0.893729    1.306312    0.560000
6         0.794611    0.945334    0.742000
7         0.711580    0.997155    0.703000
8         0.668383    0.877867    0.784000
Total time: 02:21
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1-laser-en1/models/sp15k/qrnn_nl4-rnd.m
Loss and accuracy using (cls_best): [0.64768696, tensor(0.7780)]
Processing data/mldoc/en-1/models/sp15k/qrnn_nl4.m
en-1-laser-en1
Processing data/mldoc/es-1/models/sp15k/qrnn_nl4.m
es-1-laser-en1
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1-laser-en1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1-laser-en1/models/sp15k/qrnn_nl4-rnd.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1-laser-en1/es.dev.csv
Data lm, trn: 13013, val: 1445
Data cls, trn: 1000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁la', '▁.', 's', '▁en', '▁el', '▁y', '▁a', '▁que']
Starting classifier from random weights
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.390623    1.330758    0.365000
2         1.226550    1.615878    0.511000
3         1.097214    1.439728    0.532000
4         0.988972    1.093623    0.688000
5         0.872939    1.278118    0.623000
6         0.811549    0.894074    0.779000
7         0.703965    0.821635    0.818000
8         0.630001    0.782405    0.823000
Total time: 02:00
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1-laser-en1/models/sp15k/qrnn_nl4-rnd.m
Loss and accuracy using (cls_best): [0.9085049, tensor(0.7050)]
Processing data/mldoc/fr-1/models/sp15k/qrnn_nl4.m
fr-1-laser-en1
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1-laser-en1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1-laser-en1/models/sp15k/qrnn_nl4-rnd.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1-laser-en1/fr.dev.csv
Data lm, trn: 13500, val: 1500
Data cls, trn: 1000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', 's', '▁.', "'", '▁la', '▁le', '▁et', '▁l', '▁à']
Starting classifier from random weights
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.331783    1.457629    0.270000
2         1.206117    1.568209    0.442000
3         1.085947    1.397149    0.477000
4         0.965919    1.025710    0.668000
5         0.854985    0.915386    0.732000
6         0.784643    0.922064    0.725000
7         0.691292    0.896307    0.761000
8         0.641204    0.867914    0.785000
Total time: 02:22
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1-laser-en1/models/sp15k/qrnn_nl4-rnd.m
Loss and accuracy using (cls_best): [0.66897815, tensor(0.7565)]
Processing data/mldoc/it-1/models/sp15k/qrnn_nl4.m
it-1-laser-en1
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1-laser-en1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1-laser-en1/models/sp15k/qrnn_nl4-rnd.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1-laser-en1/it.dev.csv
Data lm, trn: 13500, val: 1500
Data cls, trn: 1000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁.', '▁di', '▁e', "▁&'", "'", '▁il', '▁la', '▁in', 'e']
Starting classifier from random weights
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.365793    1.365611    0.388000
2         1.262174    2.729846    0.271000
3         1.109978    1.754144    0.415000
4         0.961007    0.922328    0.731000
5         0.822061    0.961720    0.721000
6         0.724208    0.979705    0.707000
7         0.637849    0.910123    0.754000
8         0.588057    0.915819    0.744000
Total time: 01:25
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1-laser-en1/models/sp15k/qrnn_nl4-rnd.m
Loss and accuracy using (cls_best): [0.9051443, tensor(0.6852)]
Processing data/mldoc/ja-1/models/sp15k/qrnn_nl4.m
ja-1-laser-en1
Processing data/mldoc/ru-1/models/sp15k/qrnn_nl4.m
ru-1-laser-en1
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1-laser-en1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1-laser-en1/models/sp15k/qrnn_nl4-rnd.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1-laser-en1/ru.dev.csv
Data lm, trn: 9195, val: 1021
Data cls, trn: 1000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁.', '▁в', 'а', 'и', 'е', '▁и', 'й', '▁на', 'х']
Starting classifier from random weights
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.381033    1.389893    0.231000
2         1.327294    1.474365    0.396000
3         1.190571    2.075783    0.501000
4         1.070966    1.025509    0.611000
5         0.950332    0.956073    0.718000
6         0.812717    1.165698    0.706000
7         0.737636    0.924283    0.778000
8         0.697346    0.947041    0.779000
Total time: 03:16
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1-laser-en1/models/sp15k/qrnn_nl4-rnd.m
Loss and accuracy using (cls_best): [1.2389272, tensor(0.6137)]
Processing data/mldoc/zh-1/models/sp15k/qrnn_nl4.m
zh-1-laser-en1
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1-laser-en1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1-laser-en1/models/sp15k/qrnn_nl4-rnd.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1-laser-en1/zh.dev.csv
Data lm, trn: 13500, val: 1500
Data cls, trn: 1000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁的', '▁。', '▁年', '▁、', '▁在', '▁一', '▁是', '▁中', '▁有']
Starting classifier from random weights
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.382525    1.435203    0.328000
2         1.171075    3.494349    0.342000
3         1.044511    1.895226    0.475000
4         0.954990    1.675222    0.473000
5         0.880456    0.954913    0.747000
6         0.808870    1.084140    0.669000
7         0.762522    0.920228    0.770000
8         0.710014    0.894134    0.769000
Total time: 02:22
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1-laser-en1/models/sp15k/qrnn_nl4-rnd.m
Loss and accuracy using (cls_best): [0.6264392, tensor(0.7920)]
OrderedDict([('data/mldoc/de-1-laser-en1/models/sp15k/qrnn_nl4-rnd.m',
              0.777999997138977),
             ('data/mldoc/es-1-laser-en1/models/sp15k/qrnn_nl4-rnd.m',
              0.7049999833106995),
             ('data/mldoc/fr-1-laser-en1/models/sp15k/qrnn_nl4-rnd.m',
              0.7565000057220459),
             ('data/mldoc/it-1-laser-en1/models/sp15k/qrnn_nl4-rnd.m',
              0.6852499842643738),
             ('data/mldoc/ru-1-laser-en1/models/sp15k/qrnn_nl4-rnd.m',
              0.6137499809265137),
             ('data/mldoc/zh-1-laser-en1/models/sp15k/qrnn_nl4-rnd.m',
              0.7919999957084656)])
data/mldoc/de-1-laser-en1/models/sp15k/qrnn_nl4-rnd.m: 0.777999997138977
data/mldoc/es-1-laser-en1/models/sp15k/qrnn_nl4-rnd.m: 0.7049999833106995
data/mldoc/fr-1-laser-en1/models/sp15k/qrnn_nl4-rnd.m: 0.7565000057220459
data/mldoc/it-1-laser-en1/models/sp15k/qrnn_nl4-rnd.m: 0.6852499842643738
data/mldoc/ru-1-laser-en1/models/sp15k/qrnn_nl4-rnd.m: 0.6137499809265137
data/mldoc/zh-1-laser-en1/models/sp15k/qrnn_nl4-rnd.m: 0.7919999957084656
```

### MLDoc Classification on 1k

data/mldoc/de-1/models/sp15k/qrnn_rnd-nl4.m: 0.9024999737739563
data/mldoc/en-1/models/sp15k/qrnn_rnd-nl4.m: 0.8149999976158142
data/mldoc/es-1/models/sp15k/qrnn_rnd-nl4.m: 0.8964999914169312
data/mldoc/fr-1/models/sp15k/qrnn_rnd-nl4.m: 0.8220000267028809
data/mldoc/it-1/models/sp15k/qrnn_rnd-nl4.m: 0.7889999747276306
data/mldoc/ja-1/models/sp15k/qrnn_rnd-nl4.m: 0.8302500247955322
data/mldoc/ru-1/models/sp15k/qrnn_rnd-nl4.m: 0.7319999933242798
data/mldoc/zh-1/models/sp15k/qrnn_rnd-nl4.m: 0.8452500104904175


```
python -m ulmfit eval --glob="wiki/*-100/models/sp15k/qrnn_rnd-nl4.m" --name rnd-nl4 --dataset-template='../mldoc/${lang}-1' --num-lm-epochs=0  --num-cls-epochs=8  --bs=18 --lr_sched=1cycle --label-smoothing-eps=0.1
Processing data/wiki/de-100/models/sp15k/qrnn_rnd-nl4.m
../mldoc/de-1
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1/models/sp15k/qrnn_rnd-nl4.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1/de.dev.csv
Data lm, trn: 13500, val: 1500
Data cls, trn: 1000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁.', '▁,', '▁der', '▁die', 'en', '▁und', 's', '▁in', 'er', "▁&'"]
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
Bptt 70
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1/models/sp15k/qrnn_rnd-nl4.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.411651    1.386593    0.265000
2         1.287022    1.488027    0.361000
3         1.084153    2.390431    0.370000
4         0.904227    0.936213    0.769000
5         0.740495    1.311880    0.538000
6         0.642756    0.754690    0.833000
7         0.582816    0.661088    0.892000
8         0.548893    0.683635    0.875000
Total time: 02:13
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1/models/sp15k/qrnn_rnd-nl4.m
Loss and accuracy using (cls_best): [0.33525154, tensor(0.9025)]
Processing data/wiki/en-100/models/sp15k/qrnn_rnd-nl4.m
../mldoc/en-1
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/en-1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/en-1/models/sp15k/qrnn_rnd-nl4.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/en-1/en.dev.csv
Data lm, trn: 13500, val: 1500
Data cls, trn: 1000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁the', '▁,', 's', '▁.', '▁of', '▁and', '▁in', '▁to', '▁a', 'ed']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
Bptt 70
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/en-1/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/en-1/models/sp15k/qrnn_rnd-nl4.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.369466    1.356636    0.374000
2         1.263357    2.515120    0.320000
3         1.119744    1.250081    0.569000
4         0.949653    1.033515    0.666000
5         0.802069    0.875799    0.779000
6         0.676997    0.842525    0.807000
7         0.613777    0.794573    0.826000
8         0.571342    0.781615    0.837000
Total time: 02:26
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/en-1/models/sp15k/qrnn_rnd-nl4.m
Loss and accuracy using (cls_best): [0.5295334, tensor(0.8150)]
Processing data/wiki/es-100/models/sp15k/qrnn_rnd-nl4.m
../mldoc/es-1
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp15k/qrnn_rnd-nl4.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/es.dev.csv
Data lm, trn: 13013, val: 1445
Data cls, trn: 1000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁la', '▁.', 's', '▁en', '▁el', '▁y', '▁a', '▁que']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 2621, first 100: ['▁sa', '▁i', 'ncia', '▁ka', '▁k', '▁tra', '▁fi', '▁volvi', '▁g', '▁man', '▁pasó', '▁tropas', 'pon', 'tuvieron', '▁x', '▁les', '▁empez', 'ieron', '▁bas', 'sco', '▁cam', '▁adapta', 'sion', '▁mol', 'pico', 'siones', '▁obstante', '▁!', '▁w', 'cular', 'puesta', '▁inten', '▁produj', 'clu', 'simismo', '▁pas', 'fla', '▁amerindio', 'aje', '▁deja', '▁fre', '▁jo', '▁2.', 'american', '▁cre', 'bajo', '▁medi', 'gla', '▁dirigi', 'hol', '▁aparición', 'aciones', 'vivi', 'eras', 'spe', '▁continu', '▁permaneci', '▁ber', 'usa', 'bió', '▁permitió', '▁municipios', '▁regres', 'rt', 'mbi', '▁pr', '▁ofreci', 'emi', 'misiones', '▁cap', '▁ram', 'icio', '▁wal', 'fru', '▁gen', '▁originalmente', '▁eva', '▁ferr', '▁descubri', '▁aparecen', '▁fon', 'capi', 'estre', 'pec', '▁vendi', 'iéndose', 'eja', 'liber', 'nsa', 'ológico', 'ío', 'blo', '▁tro', '▁aviones', 'cara', '▁activo', 'mostró', 'disciplina', '▁ara', 'estra']
Bptt 70
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp15k/qrnn_rnd-nl4.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.378275    1.354129    0.314000
2         1.209560    1.333649    0.507000
3         1.022200    0.820093    0.801000
4         0.854187    1.782254    0.389000
5         0.722861    1.031932    0.692000
6         0.640640    0.762994    0.853000
7         0.583225    0.677089    0.901000
8         0.556481    0.652575    0.904000
Total time: 01:59
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp15k/qrnn_rnd-nl4.m
Loss and accuracy using (cls_best): [0.35487488, tensor(0.8965)]
Processing data/wiki/fr-100/models/sp15k/qrnn_rnd-nl4.m
../mldoc/fr-1
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/models/sp15k/qrnn_rnd-nl4.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/fr.dev.csv
Data lm, trn: 13500, val: 1500
Data cls, trn: 1000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', 's', '▁.', "'", '▁la', '▁le', '▁et', '▁l', '▁à']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
Bptt 70
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/models/sp15k/qrnn_rnd-nl4.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.360408    1.298057    0.444000
2         1.251207    2.454086    0.393000
3         1.098488    1.152682    0.544000
4         0.926239    1.256870    0.622000
5         0.806994    0.911339    0.732000
6         0.717139    0.945148    0.726000
7         0.635195    0.781772    0.825000
8         0.602214    0.763521    0.825000
Total time: 02:17
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/models/sp15k/qrnn_rnd-nl4.m
Loss and accuracy using (cls_best): [0.52210134, tensor(0.8220)]
Processing data/wiki/it-100/models/sp15k/qrnn_rnd-nl4.m
../mldoc/it-1
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1/models/sp15k/qrnn_rnd-nl4.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1/it.dev.csv
Data lm, trn: 13500, val: 1500
Data cls, trn: 1000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁.', '▁di', '▁e', "▁&'", "'", '▁il', '▁la', '▁in', 'e']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
Bptt 70
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1/models/sp15k/qrnn_rnd-nl4.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.357973    1.353393    0.293000
2         1.282061    1.535401    0.390000
3         1.144333    1.480346    0.533000
4         0.985930    1.360542    0.540000
5         0.862014    1.285450    0.661000
6         0.720891    1.140574    0.629000
7         0.625376    0.840085    0.791000
8         0.572435    0.828283    0.793000
Total time: 01:21
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1/models/sp15k/qrnn_rnd-nl4.m
Loss and accuracy using (cls_best): [0.5931235, tensor(0.7890)]
Processing data/wiki/ja-100/models/sp15k/qrnn_rnd-nl4.m
../mldoc/ja-1
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ja-1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ja-1/models/sp15k/qrnn_rnd-nl4.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ja-1/ja.dev.csv
Data lm, trn: 13500, val: 1500
Data cls, trn: 1000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁、', '▁。', '▁の', '▁に', '▁を', '▁年', 'の', '▁は', '▁・', '▁)']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
Bptt 70
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ja-1/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ja-1/models/sp15k/qrnn_rnd-nl4.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.377756    1.402221    0.254000
2         1.243837    7.998792    0.254000
3         1.066383    2.645358    0.354000
4         0.903686    1.348676    0.541000
5         0.843216    0.945152    0.743000
6         0.759674    0.801283    0.810000
7         0.689767    0.786832    0.820000
8         0.674777    0.778615    0.818000
Total time: 02:48
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ja-1/models/sp15k/qrnn_rnd-nl4.m
Loss and accuracy using (cls_best): [0.5008588, tensor(0.8303)]
Processing data/wiki/ru-100/models/sp15k/qrnn_rnd-nl4.m
../mldoc/ru-1
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp15k/qrnn_rnd-nl4.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1/ru.dev.csv
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
Bptt 70
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp15k/qrnn_rnd-nl4.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.394592    1.393761    0.265000
2         1.381886    1.476836    0.293000
3         1.258016    1.153065    0.555000
4         1.105392    1.323574    0.556000
5         0.948704    1.049486    0.703000
6         0.848964    1.480141    0.605000
7         0.757975    1.001765    0.723000
8         0.684587    0.982136    0.741000
Total time: 03:07
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp15k/qrnn_rnd-nl4.m
Loss and accuracy using (cls_best): [0.7138667, tensor(0.7320)]
Processing data/wiki/zh-100/models/sp15k/qrnn_rnd-nl4.m
../mldoc/zh-1
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp15k/qrnn_rnd-nl4.m
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
Bptt 70
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp15k/qrnn_rnd-nl4.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.341714    1.208123    0.569000
2         1.059330    1.169385    0.662000
3         0.926222    0.771242    0.824000
4         0.843994    1.997928    0.524000
5         0.800537    0.874480    0.756000
6         0.710552    0.909481    0.758000
7         0.657595    0.719883    0.854000
8         0.617662    0.727267    0.852000
Total time: 02:20
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp15k/qrnn_rnd-nl4.m
Loss and accuracy using (cls_best): [0.48266637, tensor(0.8453)]
OrderedDict([('data/mldoc/de-1/models/sp15k/qrnn_rnd-nl4.m',
              0.9024999737739563),
             ('data/mldoc/en-1/models/sp15k/qrnn_rnd-nl4.m',
              0.8149999976158142),
             ('data/mldoc/es-1/models/sp15k/qrnn_rnd-nl4.m',
              0.8964999914169312),
             ('data/mldoc/fr-1/models/sp15k/qrnn_rnd-nl4.m',
              0.8220000267028809),
             ('data/mldoc/it-1/models/sp15k/qrnn_rnd-nl4.m',
              0.7889999747276306),
             ('data/mldoc/ja-1/models/sp15k/qrnn_rnd-nl4.m',
              0.8302500247955322),
             ('data/mldoc/ru-1/models/sp15k/qrnn_rnd-nl4.m',
              0.7319999933242798),
             ('data/mldoc/zh-1/models/sp15k/qrnn_rnd-nl4.m',
              0.8452500104904175)])
data/mldoc/de-1/models/sp15k/qrnn_rnd-nl4.m: 0.9024999737739563
data/mldoc/en-1/models/sp15k/qrnn_rnd-nl4.m: 0.8149999976158142
data/mldoc/es-1/models/sp15k/qrnn_rnd-nl4.m: 0.8964999914169312
data/mldoc/fr-1/models/sp15k/qrnn_rnd-nl4.m: 0.8220000267028809
data/mldoc/it-1/models/sp15k/qrnn_rnd-nl4.m: 0.7889999747276306
data/mldoc/ja-1/models/sp15k/qrnn_rnd-nl4.m: 0.8302500247955322
data/mldoc/ru-1/models/sp15k/qrnn_rnd-nl4.m: 0.7319999933242798
data/mldoc/zh-1/models/sp15k/qrnn_rnd-nl4.m: 0.8452500104904175
```