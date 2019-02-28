## LM


### MLDoc 1 
```
python -m ulmfit cls --dataset-path data/mldoc/${LANG}-1 --base-lm-path data/wiki/${LANG}-100/models/sp15k/qrnn_${NAME}.m  --lang=${LANG} --name ${NAME} - train 20 --bs 18 --num-cls-epochs=4 --lr_sched=1cycle --label-smoothing-eps=0.1
Max vocab: 15000
Cache dir: /home/test/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp15k
Model dir: /home/test/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp15k/qrnn_nl4-merity.m
Loading validation /home/test/workspace/ulmfit-multilingual/data/mldoc/ru-1/ru.dev.csv
Data lm, trn: 9195, val: 1021
Data cls, trn: 1000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁.', '▁в', 'а', 'и', 'е', '▁и', 'й', '▁на', 'х']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
Training lm from:  [PosixPath('/home/test/workspace/ulmfit-multilingual/data/wiki/ru-100/models/sp15k/qrnn_nl4-merity.m/lm_best'), PosixPath('/home/test/workspace/ulmfit-multilingual/data/wiki/ru-100/models/sp15k/qrnn_nl4-merity.m/../itos')]
epoch     train_loss  valid_loss  accuracy
1         4.411345    3.660489    0.486965
Total time: 02:43
epoch     train_loss  valid_loss  accuracy
1         3.613334    3.372079    0.544188
2         3.325245    3.100234    0.596406
3         3.181919    2.906442    0.631586
4         3.010830    2.767429    0.656378
5         2.880418    2.663865    0.676339
6         2.825526    2.571074    0.694140
7         2.766901    2.483362    0.711652
8         2.601965    2.417213    0.726853
9         2.569160    2.341699    0.744193
10        2.588142    2.272457    0.760294
11        2.494011    2.198197    0.779175
12        2.421921    2.135517    0.795854
13        2.396429    2.075012    0.812815
14        2.306572    2.019140    0.828851
15        2.281730    1.966554    0.843595
16        2.206670    1.927567    0.854515
17        2.143836    1.901352    0.862114
18        2.141715    1.884954    0.867003
19        2.070353    1.876935    0.869214
20        2.066195    1.874844    0.869665
Total time: 2:12:21
/home/test/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp15k
Saving info /home/test/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp15k/qrnn_nl4-merity.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.994393    0.755689    0.844000
2         0.859871    0.822650    0.856000
3         0.678185    0.721333    0.859000
4         0.586906    0.693618    0.878000
Total time: 04:17
Saving models at /home/test/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp15k/qrnn_nl4-merity.m
Loss and accuracy using (cls_best): [0.3872361, tensor(0.8777)]
0.387236088514328
0.8777499794960022
```

### MLDoc 2
```
(multifit) test@test:~/workspace/ulmfit-multilingual$ python -m ulmfit cls --dataset-path data/mldoc/${LANG}-1 --base-lm-path data/mldoc/${LANG}-1/models/sp15k/qrnn_${NAME}.
m  --lang=${LANG} --name ${NAME}-16 - train 0 --bs 18 --num-cls-epochs=16 --lr_sched=1cycle --label-smoothing-eps=0.1
Max vocab: 15000
Cache dir: /home/test/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp15k
Model dir: /home/test/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp15k/qrnn_nl4-merity-16.m
Loading validation /home/test/workspace/ulmfit-multilingual/data/mldoc/ru-1/ru.dev.csv
Data lm, trn: 9195, val: 1021
Data cls, trn: 1000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁.', '▁в', 'а', 'и', 'е', '▁и', 'й', '▁на', 'х']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
/home/test/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp15k
Saving info /home/test/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp15k/qrnn_nl4-merity-16.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.081481    0.837775    0.781000
2         0.901621    0.798574    0.858000
3         0.778870    0.826576    0.859000
4         0.693465    0.787875    0.833000
5         0.639763    0.841092    0.861000
6         0.595044    0.731504    0.853000
7         0.576115    0.796013    0.819000
8         0.544098    0.744034    0.875000
9         0.531359    0.699035    0.879000
10        0.513886    0.698310    0.879000
11        0.495473    0.686897    0.864000
12        0.489863    0.688584    0.881000
13        0.481086    0.675660    0.881000
14        0.479960    0.684917    0.883000
15        0.490157    0.687865    0.882000
16        0.486081    0.679104    0.882000
Total time: 15:26
Saving models at /home/test/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp15k/qrnn_nl4-merity-16.m
Loss and accuracy using (cls_best): [0.4047818, tensor(0.8737)]
0.4047817885875702
0.8737499713897705
```