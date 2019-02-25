# without col merge
````
python -m ulmfit cls --dataset-path data/cls/${LANG}-books  --base-lm-path data/wiki-m/${LANG}-100/models/sp30k/lstm_nl4.m  --lang=${LANG} --name 'nl4' - train 20 --bs 20 --num-cls-epochs=8 --lr-sched=single
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/cls/fr-books/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/cls/fr-books/models/sp30k/lstm_nl4.m
Validation set not found using 10% of trn
Data lm, trn: 33183, val: 3687
Data cls, trn: 1800, val: 200
Running tokenization tst...
Data tst, trn: 200, val: 2000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁.', "'", 's', '▁la', '▁le', '▁et', '▁l', '▁à']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
Training lm from:  [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/wiki-m/fr-100/models/sp30k/lstm_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/wiki-m/fr-100/models/sp30k/lstm_nl4.m/../itos')]
epoch     train_loss  valid_loss  accuracy
1         3.356221    2.821012    0.518492
Total time: 00:21
epoch     train_loss  valid_loss  accuracy
1         3.041214    2.734799    0.524577
2         2.919576    2.648412    0.535661
3         2.822292    2.542236    0.549206
4         2.721790    2.414110    0.561852
5         2.596515    2.276732    0.579841
6         2.453715    2.140479    0.600370
7         2.333764    2.000186    0.621349
8         2.231092    1.873927    0.644259
9         2.101130    1.765473    0.660529
10        2.006949    1.666797    0.682196
11        1.905025    1.584023    0.696058
12        1.820798    1.513958    0.709841
13        1.751217    1.456632    0.720846
14        1.689076    1.410359    0.729947
15        1.646113    1.371438    0.739868
16        1.594153    1.346142    0.744577
17        1.564375    1.332298    0.746693
18        1.536557    1.322925    0.748995
19        1.532926    1.319159    0.749444
20        1.525449    1.318028    0.749815
Total time: 08:51
/home/pczapla/workspace/ulmfit-multilingual/data/cls/fr-books/models/sp30k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/cls/fr-books/models/sp30k/lstm_nl4.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.588118    0.581087    0.700000
2         0.504373    0.583527    0.720000
3         0.412651    0.538866    0.750000
4         0.295401    0.658459    0.750000
5         0.212442    1.054068    0.720000
6         0.126090    1.302099    0.745000
7         0.078312    1.307932    0.760000
8         0.050346    1.339740    0.745000
Total time: 00:35
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/cls/fr-books/models/sp30k/lstm_nl4.m
Loss and accuracy using (cls_best): [1.3513571, tensor(0.7700)]
1.351357102394104
0.7699999809265137
````

### FR books 

````bash
python -m ulmfit cls --dataset-path data/cls/${LANG}-books  --base-lm-path data/wiki-m/${LANG}-100/models/sp30k/lstm_nl4.m  --lang=${LANG} --name 'nl4' - train 20 --bs 20 --num-cls-epochs=8 --lr-sched=single
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/cls/fr-books/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/cls/fr-books/models/sp30k/lstm_nl4.m
Validation set not found using 10% of trn
Running tokenization lm...
Data lm, trn: 33183, val: 3687
Running tokenization cls...
Data cls, trn: 1800, val: 200
Running tokenization tst...
Data tst, trn: 200, val: 2000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁.', "'", 's', '▁la', '▁le', '▁et', '▁l', '▁à']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
Training lm from:  [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/wiki-m/fr-100/models/sp30k/lstm_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/wiki-m/fr-100/models/sp30k/lstm_nl4.m/../itos')]
epoch     train_loss  valid_loss  accuracy
1         3.790325    3.409294    0.367234
Total time: 06:02
epoch     train_loss  valid_loss  accuracy
1         3.526303    3.326439    0.378936
2         3.466923    3.226977    0.392378
3         3.342312    3.111874    0.406997
4         3.244619    2.992510    0.422330
5         3.156150    2.877498    0.437467
6         3.070326    2.762509    0.453874
7         2.956969    2.651613    0.471552
8         2.878008    2.535935    0.491058
9         2.790110    2.438724    0.508560
10        2.684145    2.323467    0.528415
11        2.633781    2.231418    0.547093
12        2.535126    2.143523    0.564889
13        2.464436    2.055402    0.582077
14        2.330094    1.989257    0.596582
15        2.372371    1.924338    0.610048
16        2.190224    1.866912    0.621738
17        2.176868    1.834098    0.629221
18        2.168293    1.809196    0.633879
19        2.151132    1.797144    0.636382
20        2.130476    1.793351    0.637044
Total time: 2:30:05
/home/pczapla/workspace/ulmfit-multilingual/data/cls/fr-books/models/sp30k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/cls/fr-books/models/sp30k/lstm_nl4.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.314315    0.530879    0.865000
2         0.336746    0.468635    0.865000
3         0.255810    0.324242    0.870000
4         0.149121    0.480570    0.885000
5         0.093909    0.613743    0.890000
6         0.091678    0.660452    0.885000
7         0.049993    0.649642    0.910000
8         0.034218    0.640008    0.910000
Total time: 04:19
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/cls/fr-books/models/sp30k/lstm_nl4.m
Loss and accuracy using (cls_best): [0.5418505, tensor(0.9100)]
0.5418505072593689
0.9100000262260437
````

```
Loading pretrained model
Unknown tokens 0, first 100: []
Training lm from:  [PosixPath('/home/n-waves/workspace/ulmfit-multilingual/data/wiki/fr-100/models/sp15k/qrnn_nl4.m/lm_best'), PosixPath('/home/n-waves/workspace/ulmfit-multilingual/data/wiki/fr-100/models/sp15k/qrnn_nl4.m/../itos')]
epoch     train_loss  valid_loss  accuracy
1         4.037580    3.312217    0.364410
Total time: 01:44
epoch     train_loss  valid_loss  accuracy
1         3.709982    3.256320    0.371825
2         3.459413    3.150574    0.386972
3         3.296628    3.037327    0.402039
4         3.186458    2.914899    0.418413
5         3.092632    2.817097    0.431216
6         2.966957    2.726081    0.442906
7         2.924824    2.647339    0.453871
8         2.818279    2.561596    0.466795
9         2.773893    2.501994    0.475877
10        2.736084    2.438490    0.485978
11        2.688937    2.370927    0.496899
12        2.615245    2.314875    0.506508
13        2.583292    2.260717    0.515725
14        2.535631    2.220295    0.522666
15        2.466035    2.179093    0.530148
16        2.461427    2.151952    0.535315
17        2.390641    2.131065    0.538749
18        2.376235    2.116927    0.541430
19        2.407630    2.115370    0.542039
20        2.391378    2.112687    0.542522
Total time: 46:33
/home/n-waves/workspace/ulmfit-multilingual/data/cls/fr-books/models/sp15k
Saving info /home/n-waves/workspace/ulmfit-multilingual/data/cls/fr-books/models/sp15k/qrnn_nl4.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.466671    0.534682    0.745000
2         0.358965    0.372612    0.875000
3         0.251557    0.311034    0.900000
4         0.166484    0.585425    0.865000
5         0.101803    0.726341    0.900000
6         0.072025    0.587875    0.885000
7         0.045328    0.760989    0.890000
8         0.027765    0.727203    0.890000
Total time: 01:17
Saving models at /home/n-waves/workspace/ulmfit-multilingual/data/cls/fr-books/models/sp15k/qrnn_nl4.m
Loss and accuracy using (cls_best): [0.55982095, tensor(0.8970)]
0.5598209500312805
0.8970000147819519
```