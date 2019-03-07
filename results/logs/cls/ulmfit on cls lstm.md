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