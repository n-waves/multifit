# QRNN EN
## SP30k nl 4
### LM

```
python -m ulmfit lm --dataset-path data/wiki/wikitext-103 --bidir=False --qrnn=True --nl 4 --tokenizer='sp' --max-vocab 30000 --lang en --name 'nl4' --cuda-id=1  -  train 10 --drop-mult=0 --bs=50

Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁the', '▁,', '▁.', 's', '▁of', '▁and', '▁in', '▁to', '▁a', 'ed']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.5} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Training lm from random weights
epoch     train_loss  valid_loss  accuracy
1         3.184221    3.256314    0.438527
2         3.084555    3.241498    0.435628
3         3.099060    3.258447    0.435060
4         3.119621    3.220939    0.437597
5         3.073662    3.165012    0.445108
6         2.938047    3.086962    0.452921
7         2.920506    2.998151    0.462940
8         2.920506    2.899240    0.474378
9         2.862836    2.835098    0.485305
10        2.891070    2.810929    0.489867
```

### LM, BS=128, drop-mult=0.5
```
python -m ulmfit lm --dataset-path data/wiki/wikitext-103 --bidir=False --qrnn=True --nl 4 --tokenizer='sp' --max-vocab 30000 --lang en --name 'nl4-bs128' --cuda-id=1  -  train 10 --drop-mult=0.5 --bs=128

Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁the', '▁,', '▁.', 's', '▁of', '▁and', '▁in', '▁to', '▁a', 'ed']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.5} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Training lm from random weights
epoch     train_loss  valid_loss  accuracy
1         3.413345    3.280860    0.433011
2         3.219606    3.129479    0.444172
3         3.136091    3.094905    0.448493
4         3.145281    3.033001    0.452830
5         3.100366    2.980189    0.458984
6         3.062894    2.923044    0.464841
7         3.001627    2.834753    0.475316
8         2.979051    2.792044    0.480915
9         2.933140    2.733279    0.488346
10        2.964397    2.720861    0.490423
```

### MLDocs
```
python -m ulmfit cls --dataset-path data/mldoc/en-1  --cuda-id=0 --base-lm-path data-filtered/data/wiki/wikitext-103/models/sp30k/qrnn_nl4.m  --lang=en --name 'nl4' - train 20 --bs 40 --cls-max-len 700

Max vocab: 30000
Cache dir: /home/marcin/github/n-waves/ulmfit-multilingual/data/mldoc/en-1/models/sp30k
Model dir: /home/marcin/github/n-waves/ulmfit-multilingual/data/mldoc/en-1/models/sp30k/qrnn_nl4.m
Loading validation /home/marcin/github/n-waves/ulmfit-multilingual/data/mldoc/en-1/en.dev.csv
Running tokenization...
Saving tokenized: cls.trn 13500, cls.val 1500
Running tokenization...
Saving tokenized: cls.trn 1000, cls.val 1000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁the', '▁,', '▁.', 's', '▁of', '▁and', '▁in', '▁to', '▁a', 'ed']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
Training lm from:  [PosixPath('/home/marcin/github/n-waves/ulmfit-multilingual/data-filtered/data/wiki/wikitext-103/models/sp30k/qrnn_nl4.m/lm_best'), PosixPath('/home/marcin/github/n-waves/ulmfit-multilingual/data-filtered/data/wiki/wikitext-103/models/sp30k/qrnn_nl4.m/.
./itos')]
epoch     train_loss  valid_loss  accuracy
1         4.459886    3.692770    0.364677
epoch     train_loss  valid_loss  accuracy
1         3.962907    3.560222    0.379027
2         3.673292    3.378484    0.402066
3         3.460093    3.191662    0.424295
4         3.296515    3.030681    0.442995
5         3.161650    2.891829    0.459052
6         3.022674    2.776469    0.473280
7         2.974365    2.686321    0.484403
8         2.869587    2.593854    0.496297
9         2.785321    2.509093    0.506853
10        2.677728    2.440328    0.516178
11        2.641243    2.371950    0.525810
12        2.652385    2.320008    0.533105
13        2.547195    2.261057    0.542046
14        2.491570    2.216933    0.548810
15        2.454437    2.179364    0.555077
16        2.414449    2.147612    0.559972
17        2.358593    2.125351    0.563405
18        2.362696    2.111580    0.565614
19        2.341626    2.104268    0.566749
20        2.342680    2.102918    0.566966
/home/marcin/github/n-waves/ulmfit-multilingual/data/mldoc/en-1/models/sp30k
Saving info /home/marcin/github/n-waves/ulmfit-multilingual/data/mldoc/en-1/models/sp30k/qrnn_nl4.m/info.json
Starting classifier training
epoch     train_loss  valid_loss  accuracy
1         0.622379    0.422002    0.882000
Better model found at epoch 1 with val_loss value: 0.42200201749801636.
epoch     train_loss  valid_loss  accuracy
1         0.313018    0.275563    0.908000
Better model found at epoch 1 with val_loss value: 0.27556276321411133.
epoch     train_loss  valid_loss  accuracy
1         0.241521    0.174606    0.933000
Better model found at epoch 1 with val_loss value: 0.1746061146259308.
epoch     train_loss  valid_loss  accuracy
1         0.125556    0.170286    0.940000
Better model found at epoch 1 with val_loss value: 0.17028628289699554.
2         0.107322    0.181366    0.939000
Saving models at /home/marcin/github/n-waves/ulmfit-multilingual/data/mldoc/en-1/models/sp30k/qrnn_nl4.m
Loss and accuracy using (cls_best): [0.18917121, tensor(0.9388)]
0.1891712099313736
0.9387500286102295
```
