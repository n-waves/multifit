# FR 
## SP30k LSTM nl 4 
### LM 
```
python -m ulmfit lm --dataset-path data/wiki/it-100 --lang=it --bidir=False --qrnn=False --max-vocab 30000 --nl 4 --tokenizer=sp --name 'nl4bs100'  -  train 10 --bs 100 --dropout-mult=0
Wiki text was split to 164583 articles
Wiki text was split to 98 articles
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁.', '▁di', "▁&'", "'", '▁e', '▁il', '▁la', 'e', '▁in']
Training args:  {'tie_weights': True, 'clip': 0.12, 'bptt': 70, 'pretrained_fnames': None, 'pretrained_model': None, 'drop_mult': 0.0} dps:  [0.25 0.1  0.2  0.02 0.15]
Training lm from random weights
epoch     train_loss  valid_loss  accuracy
1         3.306743    3.717148    0.353641
2         3.126413    3.606443    0.360839
3         3.062586    3.545493    0.365721
4         3.055600    3.474823    0.373451
5         2.927211    3.406635    0.380311
6         2.924096    3.321370    0.389487
7         2.779998    3.233350    0.399968
8         2.722100    3.147745    0.410365
9         2.615910    3.087420    0.419097
10        2.565747    3.075364    0.420906
data/wiki/it-100/models/sp30k
Saving info data/wiki/it-100/models/sp30k/lstm_nl4bs100.m/info.json
```


### MLDoc
MultiCCA: 85.55%, ULMFiT 88.42%
```
python -m ulmfit cls --dataset-path data/mldoc/it-1  --base-lm-path data/wiki/it-100/models/sp30k/lstm_nl4bs100.m  --lang=it --name 'nl4bs100' --cuda-id=1 - train 20 --bs 40 --num-cls-epochs=2
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1/models/sp30k/lstm_nl4bs100.m
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1/it.dev.csv
Running tokenization...
Saving tokenized: cls.trn 13500, cls.val 1500
Running tokenization...
Saving tokenized: cls.trn 1000, cls.val 1000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁.', '▁di', "▁&'", "'", '▁e', '▁il', '▁la', 'e', '▁in']
Training args:  {'tie_weights': True, 'clip': 0.12, 'bptt': 70, 'pretrained_fnames': [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/wiki/it-100/models/sp30k/lstm_nl4bs100.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/wiki/it-100/models/sp30k/lstm_nl4bs100.m/../itos')], 'pretrained_model': None, 'drop_mult': 0.3} dps:  [0.25 0.1  0.2  0.02 0.15]
Unknown tokens 0, first 100: []
Training lm from:  [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/wiki/it-100/models/sp30k/lstm_nl4bs100.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/wiki/it-100/models/sp30k/lstm_nl4bs100.m/../itos')]
epoch     train_loss  valid_loss  accuracy
1         2.826957    2.518636    0.492175
epoch     train_loss  valid_loss  accuracy
1         2.606302    2.397623    0.509596
2         2.470586    2.260363    0.531301
3         2.334087    2.113640    0.554089
4         2.176830    1.988222    0.572687
5         2.123101    1.869944    0.591537
6         2.011187    1.770606    0.606682
7         1.934953    1.676852    0.622504
8         1.889363    1.592609    0.637525
9         1.774590    1.517665    0.652233
10        1.725905    1.435543    0.666759
11        1.670903    1.365167    0.681168
12        1.610080    1.302561    0.694462
13        1.522876    1.242124    0.708201
14        1.478528    1.193259    0.718366
15        1.423993    1.150854    0.728324
16        1.389901    1.115550    0.735836
17        1.365959    1.094267    0.740730
18        1.347579    1.079465    0.744019
19        1.321906    1.074090    0.745281
20        1.332676    1.073143    0.745453
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1/models/sp30k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1/models/sp30k/lstm_nl4bs100.m/info.json
Starting classifier training
epoch     train_loss  valid_loss  accuracy
1         0.632703    0.463210    0.831000
epoch     train_loss  valid_loss  accuracy
1         0.527650    0.390041    0.858000
epoch     train_loss  valid_loss  accuracy
1         0.436223    0.326409    0.871000
epoch     train_loss  valid_loss  accuracy
1         0.361738    0.321380    0.875000
2         0.340658    0.315946    0.877000
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1/models/sp30k/lstm_nl4bs100.m
Loss and accuracy using (cls_best): [0.32998973, tensor(0.8842)]
```