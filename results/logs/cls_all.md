# Overall

## DE BOOKS LSTM

```bash
(fastai) julian@dl-box-spot:~/ulmfit-multilingual$ python -m ulmfit cls --dataset-path data/cls/de-books  --base-lm-path ../data/wiki/de-100/models/sp30k/lstm_nl4.m --tokenizer='sp' --lang=de --name 'nl4' - train 20 --bs 40 --lr-sched=1cycle --num-cls-epochs=8
Max vocab: 30000
Cache dir: /home/julian/ulmfit-multilingual/data/cls/de-books/models/sp30k
Model dir: /home/julian/ulmfit-multilingual/data/cls/de-books/models/sp30k/lstm_nl4.m
Validation set not found using 10% of trn
Running tokenization lm...
Data lm, trn: 30600, val: 3400                                                                                                                                       
Running tokenization cls...
Data cls, trn: 1800, val: 200                                                                                                                                        
Running tokenization tst...
Data tst, trn: 200, val: 2000                                                                                                                                        
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁.', '▁,', '▁der', '▁die', '▁und', '▁in', 'en', "▁&'", 's', '-']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
Training lm from:  [PosixPath('/home/julian/data/wiki/de-100/models/sp30k/lstm_nl4.m/lm_best'), PosixPath('/home/julian/data/wiki/de-100/models/sp30k/lstm_nl4.m/../itos')]
epoch     train_loss  valid_loss  accuracy
1         3.553707    3.077508    0.458865     
Total time: 09:49
epoch     train_loss  valid_loss  accuracy
1         3.233023    2.998775    0.469265
2         3.130155    2.900699    0.481559
3         3.038409    2.793384    0.494638
4         2.970666    2.693831    0.506061
5         2.899437    2.601532    0.515972
6         2.803581    2.516531    0.526783
7         2.732246    2.443080    0.536339
8         2.675900    2.375012    0.544895
9         2.636490    2.313508    0.553726
10        2.604711    2.253531    0.562466
11        2.550045    2.202728    0.570852
12        2.501192    2.145478    0.579989
13        2.484679    2.092014    0.588614
14        2.409206    2.044224    0.596671
15        2.344645    2.008057    0.603097
16        2.346225    1.976991    0.608867
17        2.313172    1.954794    0.612839
18        2.269678    1.937210    0.615802
19        2.277551    1.930322    0.617028
20        2.246812    1.928822    0.617285
Total time: 3:38:47
/home/julian/ulmfit-multilingual/data/cls/de-books/models/sp30k
Saving info /home/julian/ulmfit-multilingual/data/cls/de-books/models/sp30k/lstm_nl4.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.406082    0.238703    0.905000
2         0.313938    0.477311    0.865000                                                                          3         0.255491    0.228014    0.890000
4         0.155123    0.384204    0.900000
5         0.107264    0.374567    0.905000
6         0.070755    0.468389    0.900000
7         0.035681    0.243386    0.945000
8         0.022694    0.242060    0.920000
Total time: 05:49
Saving models at /home/julian/ulmfit-multilingual/data/cls/de-books/models/sp30k/lstm_nl4.m
Loss and accuracy using (cls_best): [0.43779454, tensor(0.9170)]         
0.4377945363521576
0.9169999957084656
```

## DE BOOKS QRNN

```bash
(fastai) julian@dl-box-spot:~/ulmfit-multilingual$ python -m ulmfit cls --dataset-path data/cls/de-books  --base-lm-path ../data/wiki/de-100/models/sp15k/qrnn_nl4.m --tokenizer='sp' --lang=de --name 'nl4' - train 20 --bs 20 --lr-sched=1cycle --num-cls-epochs=8

Loading pretrained model
Unknown tokens 0, first 100: []
Training lm from:  [PosixPath('/home/julian/data/wiki/de-100/models/sp15k/qrnn_nl4.m/lm_best'), PosixPath('/home/julian/data/wiki/de-100/models/sp15k/qrnn_nl4.m/../itos')]
epoch     train_loss  valid_loss  accuracy
1         3.865708    3.093886    0.446700                                                                                                                                                                                                   
Total time: 03:56
epoch     train_loss  valid_loss  accuracy
1         3.355898    2.985934    0.461723
2         3.105310    2.841031    0.480453
3         2.973803    2.708443    0.496344
4         2.846477    2.598246    0.509789
5         2.772356    2.528287    0.517500
6         2.664962    2.439845    0.528694
7         2.635260    2.375823    0.536753
8         2.593652    2.320358    0.544354
9         2.555928    2.274414    0.550421
10        2.509033    2.230369    0.556924
11        2.474522    2.189939    0.562351
12        2.450867    2.152792    0.567782
13        2.430727    2.118766    0.573289
14        2.386325    2.086252    0.578170
15        2.345515    2.061625    0.582117
16        2.348084    2.041799    0.585584
17        2.322886    2.025411    0.588263
18        2.309043    2.016290    0.589943
19        2.273257    2.011463    0.590589
20        2.293612    2.010254    0.590786
Total time: 1:44:20
/home/julian/ulmfit-multilingual/data/cls/de-books/models/sp15k
Saving info /home/julian/ulmfit-multilingual/data/cls/de-books/models/sp15k/qrnn_nl4.m/info.json

Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.432808    0.280206    0.890000
2         0.321715    0.225741    0.930000
3         0.226988    0.583458    0.765000
4         0.155420    0.281753    0.925000
5         0.095285    0.330694    0.910000
6         0.066643    0.484801    0.885000
7         0.035414    0.373012    0.920000
8         0.018784    0.392293    0.920000
Total time: 02:43
Saving models at /home/julian/ulmfit-multilingual/data/cls/de-books/models/sp15k/qrnn_nl4.m
Loss and accuracy using (cls_best): [0.6044611, tensor(0.9175)]                     
0.6044610738754272
0.9175000190734863

```

## FR BOOKS LSTM

```bash
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
```

## FR BOOKS QRNN

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

## JA BOOKS LSTM

```bash
(fastai) julian@dl-box-spot:~/ulmfit-multilingual$ python -m ulmfit cls --dataset-path data/cls/ja-books  --base-lm-path ../data/wiki/ja-100/models/sp30k/lstm_nl4.m --tokenizer='sp' --lang=ja --name 'nl4' - train 20 --bs 40 --lr-sched=1cycle --num-cls-epochs=8

Max vocab: 30000
Cache dir: /home/julian/ulmfit-multilingual/data/cls/ja-books/models/sp30k
Model dir: /home/julian/ulmfit-multilingual/data/cls/ja-books/models/sp30k/lstm_nl4.m
Validation set not found using 10% of trn
Running tokenization lm...
Data lm, trn: 30600, val: 3399 
Running tokenization cls...
Data cls, trn: 1800, val: 200
Running tokenization tst...
Data tst, trn: 200, val: 1999
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁、', '▁の', '▁。', '▁に', '▁を', '▁は', '▁年', '▁が', '▁)', '▁(']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
Training lm from:  [PosixPath('/home/julian/data/wiki/ja-100/models/sp30k/lstm_nl4.m/lm_best'), PosixPath('/home/julian/data/wiki/ja-100/models/sp30k/lstm_nl4.m/../itos')]
epoch     train_loss  valid_loss  accuracy
1         3.847485    3.422460    0.376126
                                         
Total time: 06:28
epoch     train_loss  valid_loss  accuracy
1         3.588070    3.347799    0.386644
2         3.481361    3.262498    0.398140
3         3.393415    3.164333    0.410120
4         3.305471    3.069857    0.421340
5         3.246775    2.972239    0.433235
6         3.127283    2.886817    0.443638
7         3.085512    2.806101    0.454796
8         3.016604    2.738343    0.463713
9         2.947214    2.667280    0.473756
10        2.919253    2.602734    0.483177  
12        2.799891    2.488073    0.501841
13        2.772961    2.432371    0.511213
14        2.706188    2.389203    0.518911
15        2.653137    2.346985    0.526103
16        2.624165    2.316532    0.531397
17        2.578764    2.293599    0.535356
18        2.568077    2.279164    0.537922
19        2.529823    2.271825    0.539241
20        2.558044    2.270341    0.539438

Total time: 2:35:18
/home/julian/ulmfit-multilingual/data/cls/ja-books/models/sp30k
Saving info /home/julian/ulmfit-multilingual/data/cls/ja-books/models/sp30k/lstm_nl4.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.492803    0.584423    0.770000
2         0.415298    0.697332    0.675000
3         0.321692    0.742086    0.705000
4         0.281904    1.092880    0.730000
5         0.168274    1.050856    0.820000
6         0.112483    0.895169    0.795000
7         0.065752    1.082333    0.795000
8         0.038848    1.138289    0.805000
Total time: 05:46
Saving models at /home/julian/ulmfit-multilingual/data/cls/ja-books/models/sp30k/lstm_nl4.m
Loss and accuracy using (cls_best): [0.7825211, tensor(0.8514)]
0.78252112865448
0.8514257073402405
```

## JA BOOKS QRNN
```
(fbashastai) julian@dl-box-spot:~/ulmfit-multilingual$ python -m ulmfit cls --dataset-path data/cls/ja-books  --base-lm-path ../data/wiki/ja-100/models/sp15k/qrnn_nl4.m/ --tokenizer='sp' --lang=ja --name 'nl4' - train 20 --bs 20 --lr-sched=1cycle --num-cls-epochs=8
Max vocab: 15000
Cache dir: /home/julian/ulmfit-multilingual/data/cls/ja-books/models/sp15k
Model dir: /home/julian/ulmfit-multilingual/data/cls/ja-books/models/sp15k/qrnn_nl4.m
Validation set not found using 10% of trn
Running tokenization lm...
Data lm, trn: 30600, val: 3399                                                                                                                                       
Running tokenization cls...
Data cls, trn: 1800, val: 200                                                                                                                                        
Running tokenization tst...
Data tst, trn: 200, val: 1999                                                                                                                                        
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁、', '▁。', '▁の', '▁に', '▁を', '▁年', 'の', '▁は', '▁・', '▁)']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
Training lm from:  [PosixPath('/home/julian/data/wiki/ja-100/models/sp15k/qrnn_nl4.m/lm_best'), PosixPath('/home/julian/data/wiki/ja-100/models/sp15k/qrnn_nl4.m/../itos')]
epoch     train_loss  valid_loss  accuracy
1         4.507299    3.523476    0.375921
Total time: 02:56
epoch     train_loss  valid_loss  accuracy
1         3.743898    3.405970    0.388302
2         3.470031    3.225125    0.410167
3         3.257088    3.058917    0.429464
4         3.124567    2.932191    0.443150
5         3.024665    2.830841    0.454890
6         2.945257    2.753084    0.463231
8         2.825861    2.637334    0.477286
9         2.799877    2.594100    0.482632
10        2.727558    2.548748    0.488096
11        2.709687    2.512753    0.492869
12        2.677212    2.475403    0.498646
13        2.635228    2.447274    0.502558
14        2.604527    2.418465    0.506572
15        2.593657    2.394549    0.510170
16        2.563881    2.378627    0.512493
17        2.547412    2.363772    0.514894
18        2.523795    2.355130    0.516159
19        2.524199    2.351448    0.516690
20        2.504680    2.350773    0.516817
Total time: 1:19:10
/home/julian/ulmfit-multilingual/data/cls/ja-books/models/sp15k
Saving info /home/julian/ulmfit-multilingual/data/cls/ja-books/models/sp15k/qrnn_nl4.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.541095    0.580246    0.715000
2         0.459145    0.636574    0.720000
3         0.370347    0.669777    0.710000
4         0.271052    0.781472    0.740000
5         0.198006    1.307968    0.750000
6         0.139997    1.702155    0.715000
7         0.112580    1.866208    0.735000
8         0.065695    2.046943    0.740000
Total time: 02:11
Saving models at /home/julian/ulmfit-multilingual/data/cls/ja-books/models/sp15k/qrnn_nl4.m
Loss and accuracy using (cls_best): [1.0670499, tensor(0.8349)]                                                                                                                                                          
1.0670498609542847
0.8349174857139587
```


## DE DVD LSTM

```bash
(fastai) julian@dl-box-spot:~/ulmfit-multilingual$ python -m ulmfit cls --dataset-path data/cls/de-dvd  --base-lm-path ../data/wiki/de-100/models/sp30k/lstm_nl4.m --tokenizer='sp' --lang=de --name 'nl4' - train 20 --bs 40  --lr-sched=1cycle --num-cls-epochs=8

Max vocab: 30000                                                                                  
Cache dir: /home/julian/ulmfit-multilingual/data/cls/de-dvd/models/sp30k
Model dir: /home/julian/ulmfit-multilingual/data/cls/de-dvd/models/sp30k/lstm_nl4.m
Validation set not found using 10% of trn
Running tokenization lm...
Data lm, trn: 30600, val: 3400                                                                    

Running tokenization cls...                                                                       
Data cls, trn: 1800, val: 200                                                                     

Running tokenization tst...                                                                       
Data tst, trn: 200, val: 2000                                                                     

Size of vocabulary: 30000                                                                         
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', 
'<unk>', '▁', '▁.', '▁,', '▁der', '▁die', '▁und', '▁in', 'en', "▁&'", 's', '-']                   
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, '
hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}                                
Loading pretrained model
Unknown tokens 0, first 100: []                                             
Training lm from:  [PosixPath('/home/julian/data/wiki/de-100/models/sp30k/lstm_nl4.m/lm_best'), Po
sixPath('/home/julian/data/wiki/de-100/models/sp30k/lstm_nl4.m/../itos')]                         
epoch     train_loss  valid_loss  accuracy                                                        
1         3.564329    3.126153    0.456060
Total time: 09:33
epoch     train_loss  valid_loss  accuracy
1         3.270905    3.030011    0.467729
2         3.127578    2.923014    0.481434
3         3.027712    2.804503    0.495315
4         2.922042    2.698406    0.507350
5         2.833169    2.605587    0.518419
6         2.765501    2.521508    0.528258
7         2.684195    2.443519    0.538367
8         2.644001    2.373817    0.547404
9         2.586362    2.309439    0.556455
10        2.554237    2.253083    0.564804
11        2.500762    2.196377    0.573611
12        2.469062    2.144791    0.581450
13        2.423278    2.093090    0.590096
14        2.343388    2.043406    0.598047
15        2.321417    2.008692    0.604748
16        2.265463    1.972947    0.610544
17        2.248210    1.948689    0.615224
18        2.222042    1.934402    0.617739
19        2.184187    1.926367    0.619161
20        2.225068    1.925283    0.619336  
Total time: 3:47:59

Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.494345    0.347763    0.875000
2         0.424967    0.466457    0.810000
3         0.321692    0.440244    0.870000
4         0.212389    0.323907    0.895000
5         0.142327    0.532973    0.900000
6         0.080535    0.452185    0.885000
7         0.039367    0.456267    0.895000
8         0.021793    0.470200    0.890000
Total time: 06:18
Saving models at /home/julian/ulmfit-multilingual/data/cls/de-dvd/models/sp30k/lstm_nl4.m
Loss and accuracy using (cls_best): [0.56412625, tensor(0.8835)]
0.5641262531280518
0.8834999799728394
```

## DE DVD QRNN

```bash
(fastai) julian@dl-box-spot:~/ulmfit-multilingual$ python -m ulmfit cls --dataset-path data/cls/de-dvd  --base-lm-path ../data/wiki/de-100/models/sp15k/qrnn_nl4.m --tokenizer='sp' --lang=de --name 'nl4' - train 20 --bs 20  --lr-sched=1cycle --num-cls-epochs=8

Max vocab: 15000
Cache dir: /home/julian/ulmfit-multilingual/data/cls/de-dvd/models/sp15k
Model dir: /home/julian/ulmfit-multilingual/data/cls/de-dvd/models/sp15k/qrnn_nl4.m
Validation set not found using 10% of trn
Running tokenization lm...
Data lm, trn: 30600, val: 3400       
Running tokenization cls...
Data cls, trn: 1800, val: 200        
Running tokenization tst...
Data tst, trn: 200, val: 2000        
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁.', '▁,', '▁der', '▁die', 'en', '▁und', 's', '▁in', 'er', "▁&'"]
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
Training lm from:  [PosixPath('/home/julian/data/wiki/de-100/models/sp15k/qrnn_nl4.m/lm_best'), PosixPath('/home/julian/data/wiki/de-100/models/sp15k/qrnn_nl4.m/../itos')]
epoch     train_loss  valid_loss  accuracy
1         3.849395    3.198631    0.440844                   
Total time: 04:35
epoch     train_loss  valid_loss  accuracy
1         3.342962    3.041377    0.460921
2         3.085583    2.859099    0.483095
3         2.897264    2.715346    0.500916
4         2.801560    2.604200    0.513484
5         2.740901    2.519784    0.523829
6         2.680627    2.449723    0.532085
7         2.621618    2.391428    0.539286
8         2.558427    2.341838    0.546166
9         2.531594    2.295952    0.552346
10        2.507932    2.255107    0.557783
11        2.473801    2.214491    0.563843
12        2.475930    2.174546    0.569883
13        2.377409    2.133925    0.576207
14        2.384114    2.097871    0.582316
15        2.341124    2.069540    0.586640
16        2.296975    2.042327    0.591220
17        2.310770    2.020913    0.594684
18        2.253782    2.009811    0.596532
19        2.243294    2.002954    0.597705
20        2.220592    2.001262    0.597922
Total time: 2:06:03
/home/julian/ulmfit-multilingual/data/cls/de-dvd/models/sp15k
Saving info /home/julian/ulmfit-multilingual/data/cls/de-dvd/models/sp15k/qrnn_nl4.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.404121    0.396830    0.840000
2         0.340004    0.301888    0.890000
3         0.231930    0.635746    0.815000
4         0.173531    0.369872    0.915000
5         0.112006    0.450329    0.930000
6         0.068657    0.580160    0.920000
7         0.025398    0.703478    0.925000
8         0.015986    0.684468    0.920000
Total time: 02:54
Saving models at /home/julian/ulmfit-multilingual/data/cls/de-dvd/models/sp15k/qrnn_nl4.m
Loss and accuracy using (cls_best): [0.6911931, tensor(0.8840)]                                                                                                                                                          
0.6911931037902832
0.8840000033378601
```

## FR DVD LSTM

```bash
(fastai) julian@dl-box-spot:~/ulmfit-multilingual$ python -m ulmfit cls --dataset-path data/cls/fr-dvd  --base-lm-path ../data/wiki/fr-100/models/sp30k/lstm_nl4.m --tokenizer='sp' --lang=fr --name 'nl4' - train 20 --bs 40  --lr-sched=1cycle --num-cls-epochs=8

Max vocab: 30000
Cache dir: /home/julian/ulmfit-multilingual/data/cls/fr-dvd/models/sp30k
Model dir: /home/julian/ulmfit-multilingual/data/cls/fr-dvd/models/sp30k/lstm_nl4.m
Validation set not found using 10% of trn
Running tokenization lm...
Data lm, trn: 12021, val: 1335                                                                                                                                                                                         
Running tokenization cls...
Data cls, trn: 1800, val: 200                                                                                                                                                                                          
Running tokenization tst...
Data tst, trn: 200, val: 2000                                                                                                                                                                                          
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁.', "'", 's', '▁la', '▁le', '▁et', '▁l', '▁à']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
Training lm from:  [PosixPath('/home/julian/data/wiki/fr-100/models/sp30k/lstm_nl4.m/lm_best'), PosixPath('/home/julian/data/wiki/fr-100/models/sp30k/lstm_nl4.m/../itos')]
epoch     train_loss  valid_loss  accuracy
1         3.825279    3.341784    0.382891
Total time: 01:36
epoch     train_loss  valid_loss  accuracy
1         3.548075    3.280657    0.390619
2         3.489078    3.206246    0.400057
3         3.378426    3.099167    0.414316
4         3.278241    2.985728    0.428153
5         3.164112    2.868728    0.442793
6         3.081279    2.740803    0.459863
7         2.951635    2.615327    0.477424
8         2.860259    2.511515    0.493157
9         2.761055    2.386526    0.512397
10        2.628587    2.277270    0.531014
11        2.572315    2.181750    0.548689
12        2.452535    2.083487    0.566041
13        2.389231    1.998139    0.581409
14        2.313358    1.927491    0.594620
15        2.263673    1.873754    0.605384
16        2.196958    1.827021    0.614506
17        2.169217    1.797702    0.619863
18        2.126882    1.777056    0.623906
19        2.116131    1.767786    0.625270
20        2.090418    1.765665    0.625703
Total time: 37:20
/home/julian/ulmfit-multilingual/data/cls/fr-dvd/models/sp30k
Saving info /home/julian/ulmfit-multilingual/data/cls/fr-dvd/models/sp30k/lstm_nl4.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.586308    0.504827    0.750000
2         0.478227    0.411526    0.860000
3         0.417158    0.314054    0.890000
4         0.286224    0.263725    0.900000
5         0.163930    0.387664    0.880000
6         0.095715    0.282535    0.930000
7         0.051098    0.294014    0.930000
8         0.028741    0.301007    0.930000
Total time: 02:57
Saving models at /home/julian/ulmfit-multilingual/data/cls/fr-dvd/models/sp30k/lstm_nl4.m
Loss and accuracy using (cls_best): [0.5228756, tensor(0.8920)]                                                                                                                                                          
0.5228756070137024
0.8920000195503235
```

## FR DVD QRNN

```bash
(fastai) julian@dl-box-spot:~/ulmfit-multilingual$ python -m ulmfit cls --dataset-path data/cls/fr-dvd  --base-lm-path ../data/wiki/fr-100/models/sp15k/qrnn_nl4.m --tokenizer='sp' --lang=fr --name 'nl4' - train 20 --bs 20  --lr-sched=1cycle --num-cls-epochs=8

Max vocab: 15000
Cache dir: /home/julian/ulmfit-multilingual/data/cls/fr-dvd/models/sp15k
Model dir: /home/julian/ulmfit-multilingual/data/cls/fr-dvd/models/sp15k/qrnn_nl4.m
Validation set not found using 10% of trn
Running tokenization lm...
Data lm, trn: 12021, val: 1335                                                                                                                                                                                         
Running tokenization cls...
Data cls, trn: 1800, val: 200                                                                                                                                                                                          
Running tokenization tst...
Data tst, trn: 200, val: 2000                                                                                                                                                                                          
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', 's', '▁.', "'", '▁la', '▁le', '▁et', '▁l', '▁à']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
Training lm from:  [PosixPath('/home/julian/data/wiki/fr-100/models/sp15k/qrnn_nl4.m/lm_best'), PosixPath('/home/julian/data/wiki/fr-100/models/sp15k/qrnn_nl4.m/../itos')]
epoch     train_loss  valid_loss  accuracy
1         4.098866    3.325340    0.371045
Total time: 00:38
epoch     train_loss  valid_loss  accuracy
1         3.766878    3.274258    0.378042
2         3.552927    3.168638    0.392944
3         3.344454    3.048812    0.409167
4         3.237171    2.934779    0.422993
5         3.116725    2.819759    0.438676
6         3.054311    2.695264    0.454934
7         2.959086    2.596110    0.469993
8         2.854205    2.502099    0.482847
9         2.764695    2.409211    0.497850
10        2.717763    2.330320    0.510213
11        2.625616    2.237966    0.525547
12        2.569468    2.172560    0.536735
13        2.478079    2.097768    0.550401
14        2.392206    2.028578    0.562645
15        2.354875    1.981342    0.571711
16        2.323998    1.942189    0.578875
17        2.289521    1.915991    0.584164
18        2.245144    1.899356    0.587052
19        2.261464    1.892248    0.588192
20        2.215216    1.889717    0.588575
Total time: 17:34
/home/julian/ulmfit-multilingual/data/cls/fr-dvd/models/sp15k
Saving info /home/julian/ulmfit-multilingual/data/cls/fr-dvd/models/sp15k/qrnn_nl4.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.483799    0.268736    0.895000
2         0.401154    0.192629    0.930000
3         0.269437    0.262472    0.905000
4         0.200453    0.246643    0.915000
5         0.119864    0.434285    0.900000
6         0.073772    0.423398    0.950000
7         0.039172    0.456106    0.945000
8         0.026730    0.401908    0.945000                                                                                                                                                                                                 
Total time: 01:13
Saving models at /home/julian/ulmfit-multilingual/data/cls/fr-dvd/models/sp15k/qrnn_nl4.m
Loss and accuracy using (cls_best): [0.75294757, tensor(0.8885)]                                                                                                                                                           
0.7529475688934326
```

## JA DVD LSTM

```bash
(fastai) julian@dl-box-spot:~/ulmfit-multilingual$ python -m ulmfit cls --dataset-path data/cls/ja-dvd  --base-lm-path ../data/wiki/ja-100/models/sp30k/lstm_nl4.m --tokenizer='sp' --lang=ja --name 'nl4' - train 20 --bs 40 --lr-sched=1cycle --num-cls-epochs=8

Max vocab: 30000
Cache dir: /home/julian/ulmfit-multilingual/data/cls/ja-dvd/models/sp30k
Model dir: /home/julian/ulmfit-multilingual/data/cls/ja-dvd/models/sp30k/lstm_nl4.m
Validation set not found using 10% of trn
Running tokenization lm...
Data lm, trn: 30600, val: 3400                                                                                                                                                                                         
Running tokenization cls...
Data cls, trn: 1800, val: 200                                                                                                                                                                                          
Running tokenization tst...
Data tst, trn: 200, val: 2000                                                                                                                                                                                          
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁、', '▁の', '▁。', '▁に', '▁を', '▁は', '▁年', '▁が', '▁)', '▁(']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
Training lm from:  [PosixPath('/home/julian/data/wiki/ja-100/models/sp30k/lstm_nl4.m/lm_best'), PosixPath('/home/julian/data/wiki/ja-100/models/sp30k/lstm_nl4.m/../itos')]
epoch     train_loss  valid_loss  accuracy
1         3.904242    3.480109    0.375665
Total time: 04:37
epoch     train_loss  valid_loss  accuracy
1         3.671750    3.403708    0.386209
2         3.565445    3.310030    0.398628
3         3.443218    3.204042    0.411568
4         3.374648    3.098158    0.424107
5         3.288731    3.005343    0.435621
6         3.187409    2.913090    0.446806
7         3.135114    2.829881    0.457120
8         3.073141    2.753949    0.467695
9         2.996589    2.682856    0.478041
10        2.909743    2.613899    0.487629
11        2.859827    2.550690    0.497565
12        2.818285    2.492902    0.507112
13        2.779268    2.435685    0.516448
14        2.718145    2.387462    0.525241
15        2.664007    2.346267    0.532140
16        2.641343    2.312850    0.537994
17        2.599257    2.288488    0.542193
18        2.579481    2.274002    0.544809
19        2.571687    2.267283    0.545827
20        2.560343    2.265548    0.546052
Total time: 1:47:55
/home/julian/ulmfit-multilingual/data/cls/ja-dvd/models/sp30k
Saving info /home/julian/ulmfit-multilingual/data/cls/ja-dvd/models/sp30k/lstm_nl4.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.525819    0.402837    0.830000
2         0.453459    0.425072    0.820000
3         0.401383    0.482119    0.770000
4         0.337860    0.502686    0.775000
5         0.234284    0.805287    0.805000
6         0.134409    0.729153    0.815000
7         0.072370    0.895428    0.805000
8         0.040945    0.832303    0.800000
Total time: 03:09
Saving models at /home/julian/ulmfit-multilingual/data/cls/ja-dvd/models/sp30k/lstm_nl4.m
Loss and accuracy using (cls_best): [0.72511286, tensor(0.8395)]                                                                                                                                                         
0.7251128554344177
0.8395000100135803
```

## JA DVD QRNN

```bash
(fastai) julian@dl-box-spot:~/ulmfit-multilingual$ python -m ulmfit cls --dataset-path data/cls/ja-dvd  --base-lm-path ../data/wiki/ja-100/models/sp15k/qrnn_nl4.m --tokenizer='sp' --lang=ja --name 'nl4' - train 20 --bs 20 --lr-sched=1cycle --num-cls-epochs=8

Max vocab: 15000
Cache dir: /home/julian/ulmfit-multilingual/data/cls/ja-dvd/models/sp15k
Model dir: /home/julian/ulmfit-multilingual/data/cls/ja-dvd/models/sp15k/qrnn_nl4.m
Validation set not found using 10% of trn
Running tokenization lm...
Data lm, trn: 30600, val: 3400                                                                                                                                                                                       
Running tokenization cls...
Data cls, trn: 1800, val: 200                                                                                                                                                                                        
Running tokenization tst...
Data tst, trn: 200, val: 2000                                                                                                                                                                                        
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁、', '▁。', '▁の', '▁に', '▁を', '▁年', 'の', '▁は', '▁・', '▁)']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
Training lm from:  [PosixPath('/home/julian/data/wiki/ja-100/models/sp15k/qrnn_nl4.m/lm_best'), PosixPath('/home/julian/data/wiki/ja-100/models/sp15k/qrnn_nl4.m/../itos')]
epoch     train_loss  valid_loss  accuracy
1         4.426582    3.629627    0.375135
Total time: 03:07
epoch     train_loss  valid_loss  accuracy
1         3.805171    3.477644    0.391030
2         3.488298    3.267601    0.414995
3         3.285561    3.090409    0.434407
4         3.132227    2.956780    0.448842
5         3.038989    2.863759    0.458349
6         2.945142    2.775436    0.468323
7         2.899916    2.711322    0.476454
8         2.868184    2.657508    0.482740
9         2.826075    2.605312    0.489217
10        2.785546    2.566676    0.494568
11        2.702636    2.519315    0.500817
12        2.713773    2.482188    0.505794
13        2.672333    2.451759    0.510435
14        2.644996    2.420648    0.515061
15        2.613210    2.397427    0.518870
16        2.569429    2.370109    0.522590
17        2.570986    2.355417    0.525179
18        2.537443    2.344698    0.527021
19        2.520152    2.340001    0.527741
20        2.538636    2.338896    0.527936
Total time: 1:25:44
/home/julian/ulmfit-multilingual/data/cls/ja-dvd/models/sp15k
Saving info /home/julian/ulmfit-multilingual/data/cls/ja-dvd/models/sp15k/qrnn_nl4.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.523285    0.524549    0.745000
2         0.455993    0.435848    0.800000
3         0.330351    0.636708    0.815000
4         0.240356    0.732489    0.820000
5         0.148504    0.834724    0.820000
6         0.095455    1.066170    0.810000
7         0.057415    1.286681    0.815000
8         0.041691    1.246479    0.825000
Total time: 02:09
Saving models at /home/julian/ulmfit-multilingual/data/cls/ja-dvd/models/sp15k/qrnn_nl4.m
Loss and accuracy using (cls_best): [1.0560615, tensor(0.8405)]                                                                                                                                                            
1.0560615062713623
0.840499997138977
```

## DE MUSIC QRNN

```bash
(fastai) julian@dl-box-spot:~/ulmfit-multilingual$ python -m ulmfit cls --dataset-path data/cls/de-music  --base-lm-path ../data/wiki/de-100/models/sp15k/qrnn_nl4.m --tokenizer='sp' --lang=de --name 'nl4' - train 20 --bs 20 --lr-sched=1cycle --num-cls-epochs=8

Max vocab: 15000
Cache dir: /home/julian/ulmfit-multilingual/data/cls/de-music/models/sp15k
Model dir: /home/julian/ulmfit-multilingual/data/cls/de-music/models/sp15k/qrnn_nl4.m
Validation set not found using 10% of trn
Running tokenization lm...
Data lm, trn: 30600, val: 3400                                                                                                                                                                                         
Running tokenization cls...
Data cls, trn: 1800, val: 200                                                                                                                                                                                          
Running tokenization tst...
Data tst, trn: 200, val: 2000                                                                                                                                                                                          
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁.', '▁,', '▁der', '▁die', 'en', '▁und', 's', '▁in', 'er', "▁&'"]
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
Training lm from:  [PosixPath('/home/julian/data/wiki/de-100/models/sp15k/qrnn_nl4.m/lm_best'), PosixPath('/home/julian/data/wiki/de-100/models/sp15k/qrnn_nl4.m/../itos')]
epoch     train_loss  valid_loss  accuracy
1         3.911617    3.177610    0.439701
Total time: 02:34
epoch     train_loss  valid_loss  accuracy
1         3.346166    3.029566    0.459678
2         3.049080    2.844766    0.481775
3         2.894477    2.690567    0.499936
4         2.759127    2.573302    0.513566
5         2.697380    2.489542    0.523565
6         2.656406    2.407121    0.533571
7         2.627402    2.345922    0.542248
8         2.536563    2.284745    0.550096
9         2.526928    2.235896    0.557415
10        2.438895    2.186658    0.564642
11        2.407938    2.137030    0.572150
12        2.372165    2.094725    0.578804
13        2.396612    2.055748    0.585109
14        2.288415    2.010951    0.592412
15        2.261529    1.980380    0.597678
16        2.253213    1.954077    0.602550
17        2.205776    1.932198    0.606216
18        2.215024    1.921017    0.608381
19        2.202204    1.913123    0.609632
20        2.191336    1.911685    0.609869
Total time: 1:13:39
/home/julian/ulmfit-multilingual/data/cls/de-music/models/sp15k
Saving info /home/julian/ulmfit-multilingual/data/cls/de-music/models/sp15k/qrnn_nl4.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.431190    0.313631    0.845000
2         0.332442    0.370172    0.855000
3         0.231262    0.320309    0.885000
4         0.169519    0.521468    0.850000
5         0.119292    0.451171    0.885000
6         0.090633    0.377672    0.905000
7         0.040820    0.412475    0.895000
8         0.020507    0.423591    0.890000
Total time: 01:44
Saving models at /home/julian/ulmfit-multilingual/data/cls/de-music/models/sp15k/qrnn_nl4.m
Loss and accuracy using (cls_best): [0.5168868, tensor(0.9145)]                                                                                                                                                            
0.5168867707252502
0.9144999980926514
```

## JA MUSIC QRNN

```bash
(fastai) julian@dl-box-spot:~/ulmfit-multilingual$ python -m ulmfit cls --dataset-path data/cls/ja-music  --base-lm-path ../data/wiki/ja-100/models/sp15k/qrnn_nl4.m --tokenizer='sp' --lang=ja --name 'nl4' - train 20 --bs 20 --lr-sched=1cycle --num-cls-epochs=8

Max vocab: 15000
Cache dir: /home/julian/ulmfit-multilingual/data/cls/ja-music/models/sp15k
Model dir: /home/julian/ulmfit-multilingual/data/cls/ja-music/models/sp15k/qrnn_nl4.m
Validation set not found using 10% of trn
Running tokenization lm...
Data lm, trn: 30600, val: 3399                                                                                                                                                                                         
Running tokenization cls...
Data cls, trn: 1800, val: 200                                                                                                                                                                                          
Running tokenization tst...
Data tst, trn: 200, val: 1999                                                                                                                                                                                          
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁、', '▁。', '▁の', '▁に', '▁を', '▁年', 'の', '▁は', '▁・', '▁)']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
Training lm from:  [PosixPath('/home/julian/data/wiki/ja-100/models/sp15k/qrnn_nl4.m/lm_best'), PosixPath('/home/julian/data/wiki/ja-100/models/sp15k/qrnn_nl4.m/../itos')]
epoch     train_loss  valid_loss  accuracy
1         4.556378    3.658723    0.373246
Total time: 02:37
epoch     train_loss  valid_loss  accuracy
1         3.817752    3.491593    0.390537
2         3.540558    3.294715    0.413274
3         3.334061    3.111620    0.433772
4         3.125960    2.963380    0.450106
5         3.041799    2.858605    0.461084
6         2.971459    2.762077    0.472102
7         2.871483    2.690570    0.480306
8         2.835200    2.623168    0.488692
9         2.770324    2.581520    0.493827
10        2.742213    2.516730    0.502326
11        2.710455    2.471337    0.509197
12        2.644516    2.412517    0.518069
13        2.621891    2.378112    0.522952
14        2.571613    2.346067    0.527796
15        2.571703    2.311945    0.533637
16        2.518094    2.287681    0.537693
17        2.500024    2.266639    0.540884
18        2.506000    2.254209    0.543006
19        2.479864    2.249063    0.543790
20        2.442601    2.247427    0.544025
Total time: 1:11:55
/home/julian/ulmfit-multilingual/data/cls/ja-music/models/sp15k
Saving info /home/julian/ulmfit-multilingual/data/cls/ja-music/models/sp15k/qrnn_nl4.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.506969    0.522131    0.770000
2         0.441373    0.474673    0.830000
3         0.353533    0.529668    0.820000
4         0.254606    1.358136    0.750000
5         0.234732    1.085676    0.800000
6         0.150228    1.233079    0.820000
7         0.075722    1.429222    0.825000
8         0.076240    1.380577    0.815000
Total time: 01:48
Saving models at /home/julian/ulmfit-multilingual/data/cls/ja-music/models/sp15k/qrnn_nl4.m
Loss and accuracy using (cls_best): [0.9297137, tensor(0.8619)]                                                                                                                                                            
0.9297137260437012
0.8619309663772583
```

## FR MUSIC QRNN

```bash
(fastai) julian@dl-box-spot:~/ulmfit-multilingual$ python -m ulmfit cls --dataset-path data/cls/fr-music  --base-lm-path ../data/wiki/fr-100/models/sp15k/qrnn_nl4.m --tokenizer='sp' --lang=fr --name 'nl4' - train 20 --bs 20 --lr-sched=1cycle --num-cls-epochs=8

Max vocab: 15000
Cache dir: /home/julian/ulmfit-multilingual/data/cls/fr-music/models/sp15k
Model dir: /home/julian/ulmfit-multilingual/data/cls/fr-music/models/sp15k/qrnn_nl4.m
Validation set not found using 10% of trn
Running tokenization lm...
Data lm, trn: 17946, val: 1993                                                                                                                                                                                         
Running tokenization cls...
Data cls, trn: 1800, val: 200                                                                                                                                                                                          
Running tokenization tst...
Data tst, trn: 200, val: 2000                                                                                                                                                                                          
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', 's', '▁.', "'", '▁la', '▁le', '▁et', '▁l', '▁à']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
Training lm from:  [PosixPath('/home/julian/data/wiki/fr-100/models/sp15k/qrnn_nl4.m/lm_best'), PosixPath('/home/julian/data/wiki/fr-100/models/sp15k/qrnn_nl4.m/../itos')]
epoch     train_loss  valid_loss  accuracy
1         4.065422    3.303533    0.375714
Total time: 01:45
epoch     train_loss  valid_loss  accuracy
1         3.680984    3.222789    0.387669
2         3.401064    3.078735    0.408392
3         3.229715    2.933169    0.427501
4         3.068981    2.768277    0.449853
5         2.927764    2.633488    0.468013
6         2.836827    2.490237    0.489544
7         2.658033    2.333551    0.515255
8         2.586924    2.215265    0.536613
9         2.499067    2.099867    0.558580
10        2.428197    1.985598    0.581145
11        2.279977    1.884761    0.602102
12        2.205373    1.801624    0.618907
13        2.160284    1.727773    0.634058
14        2.004652    1.659159    0.647294
15        1.994962    1.622999    0.655824
16        1.908620    1.570057    0.666497
17        1.871175    1.541314    0.672019
18        1.873397    1.524069    0.675352
19        1.843536    1.519989    0.676418
20        1.922381    1.517763    0.676880
Total time: 48:16
/home/julian/ulmfit-multilingual/data/cls/fr-music/models/sp15k
Saving info /home/julian/ulmfit-multilingual/data/cls/fr-music/models/sp15k/qrnn_nl4.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.391277    0.377619    0.870000
2         0.293999    0.289887    0.900000
3         0.199309    0.408675    0.915000
4         0.134351    0.413012    0.915000
5         0.100232    0.504149    0.925000
6         0.060509    0.593078    0.925000
7         0.044179    0.499701    0.930000
8         0.050736    0.508113    0.920000
Total time: 02:07
Saving models at /home/julian/ulmfit-multilingual/data/cls/fr-music/models/sp15k/qrnn_nl4.m
Loss and accuracy using (cls_best): [0.41011345, tensor(0.9285)]                                                                                                                                                           
0.41011345386505127
0.9284999966621399
```

## DE MUSIC LSTM

```bash
(fastai) julian@dl-box-spot:~/ulmfit-multilingual$ python -m ulmfit cls --dataset-path data/cls/de-music  --base-lm-path ../data/wiki/de-100/models/sp30k/lstm_nl4.m --tokenizer='sp' --lang=de --name 'nl4' - train 20 --bs 20 --lr-sched=1cycle --num-cls-epochs=8

Max vocab: 30000
Cache dir: /home/julian/ulmfit-multilingual/data/cls/de-music/models/sp30k
Model dir: /home/julian/ulmfit-multilingual/data/cls/de-music/models/sp30k/lstm_nl4.m
Validation set not found using 10% of trn
Running tokenization lm...
Data lm, trn: 30600, val: 3400
Running tokenization cls...
Data cls, trn: 1800, val: 200
Running tokenization tst...
Data tst, trn: 200, val: 2000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁.', '▁,', '▁der', '▁die', '▁und', '▁in', 'en', "▁&'", 's', '-']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
Training lm from:  [PosixPath('/home/julian/data/wiki/de-100/models/sp30k/lstm_nl4.m/lm_best'), PosixPath('/home/julian/data/wiki/de-100/models/sp30k/lstm_nl4.m/../itos')]
epoch     train_loss  valid_loss  accuracy
1         4.002521    3.526757    0.376006  
Total time: 06:32
epoch     train_loss  valid_loss  accuracy
1         3.738713    3.425875    0.389128   
2         3.607711    3.316313    0.403729   
3         3.441798    3.188569    0.418975   
4         3.347294    3.070090    0.432434   
5         3.261581    2.959181    0.447016   
6         3.161360    2.850135    0.460998   
7         3.092968    2.747133    0.475559   
8         2.999860    2.661642    0.488256   
9         2.946614    2.572933    0.502634   
10        2.834917    2.477213    0.517815    
11        2.768194    2.394673    0.532000    
12        2.743433    2.325108    0.545050    
13        2.596613    2.255300    0.557315    
14        2.347057    1.965440    0.616312    
15        2.289672    1.907880    0.626826    
16        2.238047    1.870341    0.634285
17        2.169917    1.829818    0.641958
18        2.145997    1.811395    0.645616
19        2.111888    1.800622    0.647706
20        2.067247    1.797927    0.648219
Total time: 3:50:18
/home/julian/ulmfit-multilingual/data/cls/de-music/models/sp30k
Saving info /home/julian/ulmfit-multilingual/data/cls/de-music/models/sp30k/lstm_nl4.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.392259    0.319764    0.875000
2         0.373723    0.401961    0.850000
3         0.315902    0.415566    0.850000
4         0.185113    0.312382    0.890000
5         0.122869    0.399712    0.865000
6         0.084130    0.435429    0.910000
7         0.057294    0.394715    0.890000
8         0.028046    0.391238    0.900000
Total time: 06:34
Saving models at /home/julian/ulmfit-multilingual/data/cls/de-music/models/sp30k/lstm_nl4.m
Loss and accuracy using (cls_best): [0.4154407, tensor(0.9210)]                                                                                                         
0.41544070839881897
0.9210000038146973
```

## JA MUSIC LSTM

```bash
(fastai) julian@dl-box-spot:~/ulmfit-multilingual$ python -m ulmfit cls --dataset-path data/cls/ja-music  --base-lm-path ../data/wiki/ja-100/models/sp30k/lstm_nl4.m --tokenizer='sp' --lang=ja --name 'nl4' - train 20 --bs 20 --lr-sched=1cycle --num-cls-epochs=8

Data lm, trn: 30600, val: 3399
Running tokenization cls...
Data cls, trn: 1800, val: 200 
Running tokenization tst...
Data tst, trn: 200, val: 1999
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁、', '▁の', '▁。', '▁に', '▁を', '▁は', '▁年', '▁が', '▁)', '▁(']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
Training lm from:  [PosixPath('/home/julian/data/wiki/ja-100/models/sp30k/lstm_nl4.m/lm_best'), PosixPath('/home/julian/data/wiki/ja-100/models/sp30k/lstm_nl4.m/../itos')]

epoch     train_loss  valid_loss  accuracy
1         3.680415    3.162607    0.451358  
Total time: 09:43
epoch     train_loss  valid_loss  accuracy
1         3.258039    3.043514    0.467028   
2         3.082901    2.909904    0.482319   
3         3.009017    2.784519    0.496660   
4         2.900604    2.671213    0.510406   
5         2.776977    2.570581    0.522192   
6         2.789654    2.488552    0.532739   
7         2.710876    2.407913    0.543447   
8         2.655036    2.342028    0.553344   
9         2.571593    2.281001    0.562552   
10        2.539299    2.207963    0.574177    
11        2.466461    2.139726    0.585225
12        2.441152    2.081656    0.595266
13        2.434502    2.018719    0.606514
14        2.576859    2.190329    0.569373
15        2.543341    2.137856    0.579508
16        2.467283    2.092796    0.587677
17        2.417593    2.061508    0.593782
18        2.375962    2.038786    0.598027
19        2.391491    2.029075    0.599871
20        2.352595    2.026604    0.600235
Total time: 2:41:18
/home/julian/ulmfit-multilingual/data/cls/ja-music/models/sp30k
Saving info /home/julian/ulmfit-multilingual/data/cls/ja-music/models/sp30k/lstm_nl4.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.458052    0.371875    0.860000
2         0.473817    0.539201    0.730000
3         0.423880    0.390433    0.845000
4         0.310703    0.402607    0.855000
5         0.211760    0.607136    0.865000
6         0.108535    0.845904    0.860000
7         0.053125    0.897018    0.860000
8         0.024544    0.891663    0.855000
Total time: 04:29
Saving models at /home/julian/ulmfit-multilingual/data/cls/ja-music/models/sp30k/lstm_nl4.m
Loss and accuracy using (cls_best): [0.777393, tensor(0.8644)]                                                                                                          
0.7773929834365845
0.8644322156906128
```

## FR MUSIC LSTM

```bash
(fastai) julian@dl-box-spot:~/ulmfit-multilingual$ python -m ulmfit cls --dataset-path data/cls/fr-music  --base-lm-path ../data/wiki/fr-100/models/sp30k/lstm_nl4.m --tokenizer='sp' --lang=fr --name 'nl4' - train 20 --bs 40 --lr-sched=1cycle --num-cls-epochs=8

Running tokenization tst...
Data tst, trn: 200, val: 2000                                                                                                                                       
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁.', "'", 's', '▁la', '▁le', '▁et', '▁l', '▁à']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
Training lm from:  [PosixPath('/home/julian/data/wiki/fr-100/models/sp30k/lstm_nl4.m/lm_best'), PosixPath('/home/julian/data/wiki/fr-100/models/sp30k/lstm_nl4.m/../itos')]
epoch     train_loss  valid_loss  accuracy
1         3.686174    3.271906    0.394277
Total time: 02:47
epoch     train_loss  valid_loss  accuracy
1         3.490278    3.197725    0.403639
2         3.356067    3.100815    0.415938
3         3.269037    2.970971    0.433533
4         3.102959    2.819620    0.453009
5         2.957009    2.647919    0.478246
6         2.793691    2.481614    0.504928
7         2.646251    2.316360    0.534853
8         2.532166    2.140370    0.566817
9         2.361445    1.982554    0.596333
10        2.258446    1.855159    0.621765
11        2.155252    1.772740    0.640348
12        2.071291    1.668775    0.660405
13        1.887608    1.579711    0.677771
14        1.873631    1.493046    0.694815
15        1.824689    1.438728    0.705296
16        1.766544    1.398732    0.714093
17        1.646138    1.372478    0.719408
18        1.684073    1.350950    0.723633
19        1.650602    1.344994    0.724889
20        1.602114    1.341957    0.725314
Total time: 1:04:53
/home/julian/ulmfit-multilingual/data/cls/fr-music/models/sp30k
Saving info /home/julian/ulmfit-multilingual/data/cls/fr-music/models/sp30k/lstm_nl4.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.415938    0.214479    0.930000
2         0.373839    0.334263    0.880000
3         0.317772    0.660272    0.795000
4         0.207807    0.440546    0.880000
5         0.146999    0.377026    0.890000
6         0.095834    0.288273    0.925000
7         0.048218    0.350355    0.895000
8         0.023682    0.325598    0.915000
Total time: 03:18
Saving models at /home/julian/ulmfit-multilingual/data/cls/fr-music/models/sp30k/lstm_nl4.m
Loss and accuracy using (cls_best): [0.32345334, tensor(0.9295)]                                                                                                      
0.3234533369541168
0.9294999837875366
```