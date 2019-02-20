# QRNN ES

## SP15k nl 4
``
export CUDA_VISIBLE_DEVICES=1
LANG=es
python -m ulmfit lm --dataset-path data/wiki/${LANG}-100 --tokenizer='sp' --nl 4 --name 'nl4' --max-vocab 15000 --lang ${LANG} --qrnn=True - train 10 --bs=50 --drop_mult=0

Wiki text was split to 161509 articles
Wiki text was split to 78 articles
Running tokenization lm...
Data lm, trn: 161509, val: 78
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁.', '▁la', 's', '▁el', '▁en', '▁y', '▁a', "▁&'"]
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
/home/pczapla/anaconda3/envs/fastaiv1/lib/python3.7/site-packages/torch/utils/cpp_extension.py:152: UserWarning:

Training lm from random weights
epoch     train_loss  valid_loss  accuracy
1         2.851575    3.398695    0.372940
2         2.801543    3.353015    0.372648
3         2.807216    3.290132    0.380787
4         2.696361    3.220115    0.388937
5         2.668488    3.132770    0.399528
6         2.565685    3.062742    0.408880
7         2.503054    2.985069    0.419262
8         2.448338    2.895266    0.431797
9         2.411213    2.829787    0.441973
10        2.403536    2.811063    0.445468
Total time: 11:52:32
data/wiki/es-100/models/sp15k
Saving info data/wiki/es-100/models/sp15k/qrnn_nl4.m/info.json
``

```bash
export CUDA_VISIBLE_DEVICES=1
LANG=es
python -m ulmfit cls --dataset-path data/mldoc/${LANG}-1  --base-lm-path data/wiki/${LANG}-100/models/sp15k/qrnn_nl4.m  --lang=${LANG} --name 'nl4' - train 20 --bs 20 --num-cls-epochs=8
```


## SP30k nl 4
### LM
```
python -m ulmfit lm --dataset-path data/wiki/es-100 --bidir=False --qrnn=True --nl 4 --tokenizer='sp' --max-vocab 30000 --lang es --name 'nl4' --cuda-id=0  -  train 10 --drop-mult=0 --bs=50

Wiki text was split to 161509 articles
Wiki text was split to 78 articles
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁.', '▁la', '▁el', '▁en', '▁y', 's', '▁a', "▁&'"]
Training args:  {'clip': 0.12, 'drop_mult': 0} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Training lm from random weights
epoch     train_loss  valid_loss  accuracy
1         3.067289    3.640350    0.357276
2         2.958243    3.619773    0.358111
3         3.033412    3.587700    0.359495
4         2.933573    3.525202    0.367685
5         2.904549    3.467990    0.372583
6         2.798806    3.409506    0.380045
7         2.733132    3.303108    0.391922
8         2.675272    3.224150    0.401143
9         2.635299    3.166430    0.410160
10        2.656724    3.145599    0.413176
```

### MLDocs
```
python -m ulmfit cls --dataset-path data/mldoc/es-1  --cuda-id=0 --base-lm-path data-filtered/data/wiki/es-100/models/sp30k/qrnn_nl4.m  --lang=es --name 'nl4' - train 20 --bs 40 --cls-max-len 700

Max vocab: 30000
Cache dir: /home/marcin/github/n-waves/ulmfit-multilingual/data/mldoc/es-1/models/sp30k
Model dir: /home/marcin/github/n-waves/ulmfit-multilingual/data/mldoc/es-1/models/sp30k/qrnn_nl4.m
Loading validation /home/marcin/github/n-waves/ulmfit-multilingual/data/mldoc/es-1/es.dev.csv
Running tokenization...
Saving tokenized: cls.trn 13013, cls.val 1445
Running tokenization...
Saving tokenized: cls.trn 1000, cls.val 1000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁.', '▁la', '▁el', '▁en', '▁y', 's', '▁a', "▁&'"]
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
Training lm from:  [PosixPath('/home/marcin/github/n-waves/ulmfit-multilingual/data-filtered/data/wiki/es-100/models/sp30k/qrnn_nl4.m/lm_best'), PosixPath('/home/marcin/github/n-waves/ulmfit-multilingual/data-filtered/data/wiki/es-100/models/sp30k/qrnn_nl4.m/../itos')]
epoch     train_loss  valid_loss  accuracy
1         3.352874    2.367255    0.514858
epoch     train_loss  valid_loss  accuracy
1         2.796090    2.203233    0.536513
2         2.515840    1.970145    0.576640
3         2.198857    1.774013    0.610990
4         2.035614    1.633484    0.633450
5         1.944539    1.535505    0.649110
6         1.848854    1.451618    0.661764
7         1.788579    1.382675    0.673166
8         1.675414    1.320675    0.683617
9         1.614536    1.264944    0.694086
10        1.618723    1.215493    0.702936
11        1.504875    1.164356    0.712921
12        1.411316    1.126858    0.721374
13        1.421174    1.079897    0.731196
14        1.352116    1.044965    0.738148
15        1.318876    1.013755    0.745312
16        1.268569    0.986391    0.751383
17        1.273424    0.971129    0.754643
18        1.256196    0.960661    0.757439
19        1.233202    0.955790    0.758405
20        1.230536    0.955070    0.758496
/home/marcin/github/n-waves/ulmfit-multilingual/data/mldoc/es-1/models/sp30k
Saving info /home/marcin/github/n-waves/ulmfit-multilingual/data/mldoc/es-1/models/sp30k/qrnn_nl4.m/info.json
Starting classifier training
epoch     train_loss  valid_loss  accuracy
1         0.739119    0.438338    0.867000
Better model found at epoch 1 with val_loss value: 0.438338041305542.
epoch     train_loss  valid_loss  accuracy
1         0.425376    0.207067    0.950000
Better model found at epoch 1 with val_loss value: 0.20706671476364136.
epoch     train_loss  valid_loss  accuracy
1         0.311269    0.172416    0.956000
Better model found at epoch 1 with val_loss value: 0.17241604626178741.
epoch     train_loss  valid_loss  accuracy
1         0.226164    0.166543    0.958000
Better model found at epoch 1 with val_loss value: 0.1665433794260025.
2         0.199775    0.167683    0.956000
Saving models at /home/marcin/github/n-waves/ulmfit-multilingual/data/mldoc/es-1/models/sp30k/qrnn_nl4.m
Loss and accuracy using (cls_best): [0.18184493, tensor(0.9448)]
0.18184493482112885
0.9447500109672546
```
