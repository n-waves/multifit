
## BS=18, lr_mult=1.0

epoch     train_loss  valid_loss  accuracy
1         4.427713    3.693394    0.484268
Total time: 01:32
epoch     train_loss  valid_loss  accuracy
1         3.758918    3.446661    0.529820
2         3.394254    3.199054    0.577411
3         3.235364    3.014517    0.610520
4         3.125459    2.871101    0.637153
5         2.994313    2.773862    0.654470
6         2.915075    2.693080    0.669942
7         2.855732    2.622858    0.683629
8         2.755074    2.572147    0.694145
9         2.697898    2.517524    0.704816
10        2.689881    2.468190    0.715927
11        2.579573    2.432807    0.723324
12        2.659464    2.387878    0.733931
13        2.520637    2.344804    0.744233
14        2.482952    2.315014    0.751855
15        2.564730    2.279045    0.761163
16        2.552707    2.255916    0.766971
17        2.511244    2.240169    0.770991
18        2.461429    2.228213    0.774309
19        2.426440    2.222140    0.775745
20        2.425955    2.221128    0.775836
Total time: 1:14:17

## BS=500, lr_mult=1.0
epoch     train_loss  valid_loss  accuracy
1         5.536769    3.850831    0.444662
Total time: 01:10
epoch     train_loss  valid_loss  accuracy
1         4.845898    3.781763    0.461471
2         4.388605    3.643141    0.491225
3         4.038255    3.464554    0.526143

## BS=500, lr_mult=27
epoch     train_loss  valid_loss  accuracy
1         7.234749    5.868155    0.312675
Total time: 01:41

## BS=500, lr_mult=10 + BS=50 lr_mult=10 for cls
/data/wiki/ru-100/models/sp15k/qrnn_nl4sl.m/../itos')]
epoch     train_loss  valid_loss  accuracy
1         5.052441    4.082105    0.439539
Total time: 02:27
epoch     train_loss  valid_loss  accuracy
1         4.120197    3.712686    0.498216
2         3.727043    3.373258    0.557896
3         3.383009    3.109635    0.598970
4         3.180799    2.938478    0.626816
5         3.048913    2.812639    0.647257
6         2.943903    2.727179    0.661784
7         2.864300    2.650275    0.674248
8         2.773810    2.583594    0.687063
9         2.724850    2.529445    0.697573
10        2.673996    2.473824    0.708698
11        2.657637    2.431461    0.716904
12        2.591277    2.372668    0.730318
13        2.537707    2.323294    0.741157
14        2.486507    2.280270    0.751768
15        2.435933    2.238660    0.762545
16        2.401303    2.208848    0.769561
17        2.374117    2.184253    0.776400
18        2.341421    2.169156    0.780388
19        2.328202    2.163922    0.781700
20        2.315462    2.161784    0.782105
Total time: 1:07:09
------------------- Checking the influence of number of epochs on the accuracy  
(multifit) test@test:~/workspace/ulmfit-multilingual$ rm /home/test/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp15k/qrnn_nl4sl-bs500.m/cls*
(multifit) test@test:~/workspace/ulmfit-multilingual$ python -m ulmfit cls --dataset-path data/mldoc/${LANG}-1 --base-lm-path data/wiki/${LANG}-100/models/sp15k/qrnn_${NAME}.m  --lang=${LANG} --name ${NAME}-bs500 - train 0 --bs 50 --num-cls-epochs=4 --lr_sched=1cycle --label-smoothing-eps=0.1 --lr_mult=1
Max vocab: 15000
Cache dir: /home/test/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp15k
Model dir: /home/test/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp15k/qrnn_nl4sl-bs500.m
Loading validation /home/test/workspace/ulmfit-multilingual/data/mldoc/ru-1/ru.dev.csv
Data lm, trn: 9195, val: 1021
Data cls, trn: 1000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁.', '▁в', 'а', 'и', 'е', '▁и', 'й', '▁на', 'х']
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.063845    1.111960    0.601000
2         0.902245    0.766871    0.817000
3         0.766261    0.707502    0.861000
4         0.680053    0.694492    0.866000
Total time: 01:22
Saving models at /home/test/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp15k/qrnn_nl4sl-bs500.m
Loss and accuracy using (cls_best): [0.41532615, tensor(0.8630)]
0.41532614827156067
0.8629999756813049
(multifit) test@test:~/workspace/ulmfit-multilingual$ python -m ulmfit cls --dataset-path data/mldoc/${LANG}-1 --base-lm-path data/wiki/${LANG}-100/models/sp15k/qrnn_${NAME}.m  --lang=${LANG} --name ${NAME}-bs500 - train 0 --bs 50 --num-cls-epochs=16 --lr_sched=1cycle --label-smoothing-eps=0.1 --lr_mult=1
Max vocab: 15000
Cache dir: /home/test/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp15k
Model dir: /home/test/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp15k/qrnn_nl4sl-bs500.m
Loading validation /home/test/workspace/ulmfit-multilingual/data/mldoc/ru-1/ru.dev.csv
Data lm, trn: 9195, val: 1021
Data cls, trn: 1000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁.', '▁в', 'а', 'и', 'е', '▁и', 'й', '▁на', 'х']
Loading last classifier
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.556688    0.706000    0.873000
2         0.537578    0.717411    0.865000
3         0.532326    0.775549    0.854000
4         0.529178    0.767506    0.861000
5         0.521306    0.797604    0.860000
6         0.527344    0.736225    0.868000
7         0.516393    0.724941    0.878000
8         0.510422    0.716110    0.873000
9         0.504320    0.701886    0.869000
10        0.500323    0.676577    0.878000
11        0.493490    0.682657    0.873000
12        0.484450    0.682047    0.878000
13        0.479248    0.682782    0.880000
14        0.474778    0.688019    0.873000
15        0.472664    0.685304    0.874000
16        0.470747    0.677925    0.878000
Total time: 07:57
