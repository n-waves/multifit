# QRNN DE
## SP30k nl
### LM
```
python -m ulmfit lm --dataset-path data/wiki/de-100 --bidir=False --qrnn=True --nl 4 --tokenizer='sp' --max-vocab 30000 --lang de --name 'nl4' --cuda-id=0 - train 10 --drop-mult=0 --bs=50

Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁.', '▁,', '▁der', '▁die', '▁und', '▁in', "▁&'", 'en', 's', '-']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Training lm from random weights
epoch     train_loss  valid_loss  accuracy
1         2.790653    2.867094    0.511392
2         2.742032    2.843288    0.510885
3         2.696114    2.833874    0.512062
4         2.671780    2.786312    0.516448
5         2.611292    2.725993    0.522723
6         2.542737    2.655713    0.530968
7         2.572076    2.582141    0.539928
8         2.465960    2.509654    0.549987
9         2.405682    2.448580    0.558674
10        2.339395    2.428111    0.562502
```

### MLDocs
```
python -m ulmfit cls --dataset-path data/mldoc/de-1  --cuda-id=0 --base-lm-path data-filtered/data/wiki/de-100/models/sp30k/qrnn_nl4.m  --lang=de --name 'nl4' - train 20 --bs 40 --cls-max-len 700

Max vocab: 30000
Cache dir: /home/marcin/github/n-waves/ulmfit-multilingual/data/mldoc/de-1/models/sp30k
Model dir: /home/marcin/github/n-waves/ulmfit-multilingual/data/mldoc/de-1/models/sp30k/qrnn_nl4.m
Loading validation /home/marcin/github/n-waves/ulmfit-multilingual/data/mldoc/de-1/de.dev.csv
Running tokenization...
Saving tokenized: cls.trn 13500, cls.val 1500
Running tokenization...
Saving tokenized: cls.trn 1000, cls.val 1000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁.', '▁,', '▁der', '▁die', '▁und', '▁in', "▁&'", 'en', 's', '-']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
Training lm from:  [PosixPath('/home/marcin/github/n-waves/ulmfit-multilingual/data-filtered/data/wiki/de-100/models/sp30k/qrnn_nl4.m/lm_best'), PosixPath('/home/marcin/github/n-waves/ulmfit-multilingual/data-filtered/data/wiki/de-100/models/sp30k/qrnn_nl4.m/../itos')]
epoch     train_loss  valid_loss  accuracy
1         3.450698    2.601732    0.527671
epoch     train_loss  valid_loss  accuracy
1         2.888087    2.477170    0.542949
2         2.621279    2.300024    0.568743
3         2.313220    2.120824    0.592728
4         2.176746    1.973596    0.613343
5         2.114441    1.857317    0.628628
6         2.022593    1.765069    0.642017
7         1.936942    1.696150    0.651549
8         1.860200    1.622848    0.661923
9         1.795039    1.549579    0.673416
10        1.740739    1.500053    0.681305
11        1.695835    1.448141    0.689201
12        1.605702    1.402924    0.697096
13        1.582328    1.354327    0.706123
14        1.548034    1.316290    0.712870
15        1.496170    1.282155    0.719413
16        1.514243    1.255556    0.724801
17        1.482411    1.236461    0.728380
18        1.458308    1.223498    0.730708
19        1.422691    1.218288    0.731713
20        1.380592    1.217068    0.731893
/home/marcin/github/n-waves/ulmfit-multilingual/data/mldoc/de-1/models/sp30k
Saving info /home/marcin/github/n-waves/ulmfit-multilingual/data/mldoc/de-1/models/sp30k/qrnn_nl4.m/info.json
Starting classifier training
epoch     train_loss  valid_loss  accuracy
1         0.529402    0.376163    0.900000
Better model found at epoch 1 with val_loss value: 0.3761630356311798.
epoch     train_loss  valid_loss  accuracy
1         0.290838    0.252989    0.916000
Better model found at epoch 1 with val_loss value: 0.25298893451690674.
epoch     train_loss  valid_loss  accuracy
1         0.184352    0.204892    0.941000
Better model found at epoch 1 with val_loss value: 0.20489171147346497.
epoch     train_loss  valid_loss  accuracy
1         0.113328    0.204136    0.947000
Better model found at epoch 1 with val_loss value: 0.20413607358932495.
2         0.106220    0.200674    0.949000
Better model found at epoch 2 with val_loss value: 0.20067360997200012.
Saving models at /home/marcin/github/n-waves/ulmfit-multilingual/data/mldoc/de-1/models/sp30k/qrnn_nl4.m
Loss and accuracy using (cls_best): [0.15208693, tensor(0.9532)]
0.15208692848682404
0.953249990940094
```
