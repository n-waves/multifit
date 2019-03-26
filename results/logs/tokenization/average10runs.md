
TOK=sp15k
NAME=e8avg
for LANG in ru fr; do 
    for i in 1 2 3 4 5 6 7 8 9; do python -m ulmfit eval --glob="mldoc/${LANG}-1/models/${TOK}/qrnn_nl4.m" --name ${NAME}_$i --num-cls-epochs=8 --label-smoothing-eps=0.1 --lr_sched=1cycle --bs=18; done
done
TOK=sp15k
NAME=avg
for LANG in ru fr; do 
    for i in 1 2 3 4 5 6 7 8 9; do python -m ulmfit eval --glob="mldoc/${LANG}-1/models/${TOK}/qrnn_nl4.m" --name ${NAME}_$i --num-cls-epochs=4 --label-smoothing-eps=0.1 --lr_sched=1cycle --bs=18; done
done

NAME=e8avg
TOK=vf60k
for LANG in ru fr; do 
    for i in 1 2 3 4 5 6 7 8 9; do python -m ulmfit eval --glob="mldoc/${LANG}-1/models/${TOK}/qrnn_nl4.m" --name ${NAME}_$i --num-cls-epochs=8 --label-smoothing-eps=0.1 --lr_sched=1cycle --bs=18; done
done
NAME=avg
TOK=vf60k
for LANG in ru fr; do 
    for i in 1 2 3 4 5 6 7 8 9; do python -m ulmfit eval --glob="mldoc/${LANG}-1/models/${TOK}/qrnn_nl4.m" --name ${NAME}_$i --num-cls-epochs=4 --label-smoothing-eps=0.1 --lr_sched=1cycle --bs=18; done
done


for TOK in vf60k sp15k; do
for LANG in ru fr; do 
    NAME=e8avg 
    for i in 1 2 3 4 5 6 7 8 9; do python -m ulmfit eval --glob="mldoc/${LANG}-1/models/${TOK}/qrnn_nl4.m" --name ${NAME}_$i --num-cls-epochs=8 --label-smoothing-eps=0.1 --lr_sched=1cycle --bs=18; done
done
done

for TOK in vf60k sp15k; do
for LANG in ru fr; do
    NAME=avg  
    for i in 1 2 3 4 5 6 7 8 9; do python -m ulmfit eval --glob="mldoc/${LANG}-1/models/${TOK}/qrnn_nl4.m" --name ${NAME}_$i --num-cls-epochs=4 --label-smoothing-eps=0.1 --lr_sched=1cycle --bs=18; done
done
done
LANG=es
for i in 1 2 3 4 5 6 7 9; do python -m ulmfit eval --glob="mldoc/${LANG}-1/models/sp15k/qrnn_nl4.m" --name avg_$i --num-cls-epochs=4 --label-smoothing-eps=0.1 --lr_sched=1cycle --bs=18; done

LANG=de
for i in 1 2 3 4 5 6 7 9; do python -m ulmfit eval --glob="mldoc/${LANG}-1/models/sp15k/qrnn_nl4.m" --name avg_$i --num-cls-epochs=4 --label-smoothing-eps=0.1 --lr_sched=1cycle --bs=18; done

## epoch 8
vf60k
ds     de-1   es-1   fr-1   it-1   ru-1 
best  95.45  96.17  94.77  90.72  87.85
max   95.63  96.43  95.32  91.05  88.30
avg   95.42  96.05  95.07  90.59  87.80

sp15k
ds     de-1   es-1   fr-1   it-1   ru-1
best  96.17  95.92  94.55  90.45  86.95
max   96.28  96.03  95.10  90.72  87.45
avg   96.01  95.72  94.63  90.37  86.95


-0--
0   data/mldoc/fr-1/models/vf60k/qrnn_e8avg_1.m       0.95325  0.211328         0.951  0.211664
1   data/mldoc/fr-1/models/vf60k/qrnn_e8avg_2.m       0.95075  0.199939         0.947  0.198606
2   data/mldoc/fr-1/models/vf60k/qrnn_e8avg_3.m       0.95125  0.217569         0.952  0.215529
3   data/mldoc/fr-1/models/vf60k/qrnn_e8avg_4.m       0.95225  0.208047         0.951  0.203784
4   data/mldoc/fr-1/models/vf60k/qrnn_e8avg_5.m       0.95025  0.206937         0.946  0.206194
5   data/mldoc/fr-1/models/vf60k/qrnn_e8avg_6.m       0.95075  0.203967         0.951  0.204809
6   data/mldoc/fr-1/models/vf60k/qrnn_e8avg_7.m       0.94775  0.211408         0.954  0.201543
7   data/mldoc/fr-1/models/vf60k/qrnn_e8avg_8.m       0.95075  0.202703         0.952  0.197218
8   data/mldoc/fr-1/models/vf60k/qrnn_e8avg_9.m       0.94925  0.207698         0.947  0.204896
9     data/mldoc/ru-1/models/vf60k/qrnn_avg_1.m       0.88300  0.375823         0.876  0.364029
10    data/mldoc/ru-1/models/vf60k/qrnn_avg_2.m       0.87675  0.386695         0.883  0.356660
11    data/mldoc/ru-1/models/vf60k/qrnn_avg_3.m       0.87750  0.372321         0.879  0.374368
12    data/mldoc/ru-1/models/vf60k/qrnn_avg_4.m       0.87400  0.379490         0.875  0.370343
13    data/mldoc/ru-1/models/vf60k/qrnn_avg_5.m       0.87725  0.380067         0.877  0.367522
14    data/mldoc/ru-1/models/vf60k/qrnn_avg_6.m       0.87525  0.393280         0.874  0.368825
15    data/mldoc/ru-1/models/vf60k/qrnn_avg_7.m       0.87900  0.380393         0.882  0.373376
16    data/mldoc/ru-1/models/vf60k/qrnn_avg_8.m       0.88025  0.376825         0.875  0.375059
17    data/mldoc/ru-1/models/vf60k/qrnn_avg_9.m       0.88125  0.380887         0.884  0.367705
18  data/mldoc/ru-1/models/vf60k/qrnn_e8avg_1.m       0.87850  0.385976         0.888  0.385302
19  data/mldoc/ru-1/models/vf60k/qrnn_e8avg_2.m       0.87600  0.384469         0.879  0.384878
20  data/mldoc/ru-1/models/vf60k/qrnn_e8avg_4.m       0.87600  0.386223         0.870  0.391646
21  data/mldoc/ru-1/models/vf60k/qrnn_e8avg_5.m       0.87950  0.385152         0.885  0.371122
22  data/mldoc/ru-1/models/vf60k/qrnn_e8avg_6.m       0.88175  0.383746         0.875  0.391054
23  data/mldoc/ru-1/models/vf60k/qrnn_e8avg_7.m       0.87850  0.392120         0.874  0.382000
24  data/mldoc/ru-1/models/vf60k/qrnn_e8avg_8.m       0.87250  0.394342         0.881  0.378663
25  data/mldoc/ru-1/models/vf60k/qrnn_e8avg_9.m       0.87950  0.387625         0.881  0.380548
ds     fr-1   ru-1
best  94.77  87.85
max   95.32  88.30
avg   95.07  87.80
---

# epoch 4
## SP15k
                                         name  tst_accuracy  tst_loss  val_accuracy  val_loss
0   data/mldoc/de-1/models/sp15k/qrnn_avg_1.m       0.95700  0.142444         0.945  0.206329
1   data/mldoc/de-1/models/sp15k/qrnn_avg_2.m       0.95975  0.141225         0.955  0.189209
2   data/mldoc/de-1/models/sp15k/qrnn_avg_3.m       0.95925  0.140172         0.946  0.198159
3   data/mldoc/de-1/models/sp15k/qrnn_avg_4.m       0.95650  0.145889         0.942  0.193202
4   data/mldoc/de-1/models/sp15k/qrnn_avg_5.m       0.96000  0.137710         0.953  0.192368
5   data/mldoc/de-1/models/sp15k/qrnn_avg_6.m       0.95975  0.145318         0.954  0.196413
6   data/mldoc/de-1/models/sp15k/qrnn_avg_7.m       0.95925  0.141719         0.943  0.199226
7   data/mldoc/de-1/models/sp15k/qrnn_avg_8.m       0.96100  0.140644         0.949  0.204398
8   data/mldoc/de-1/models/sp15k/qrnn_avg_9.m       0.96050  0.143330         0.949  0.189472
9   data/mldoc/es-1/models/sp15k/qrnn_avg_1.m       0.95700  0.149081         0.963  0.162808
10  data/mldoc/es-1/models/sp15k/qrnn_avg_2.m       0.95875  0.141572         0.963  0.150970
11  data/mldoc/es-1/models/sp15k/qrnn_avg_3.m       0.95900  0.151387         0.962  0.157047
12  data/mldoc/es-1/models/sp15k/qrnn_avg_4.m       0.95375  0.165935         0.956  0.182161
13  data/mldoc/es-1/models/sp15k/qrnn_avg_5.m       0.95850  0.151109         0.960  0.156376
14  data/mldoc/es-1/models/sp15k/qrnn_avg_6.m       0.95800  0.150724         0.961  0.152761
15  data/mldoc/es-1/models/sp15k/qrnn_avg_7.m       0.95875  0.142476         0.963  0.151585
16  data/mldoc/es-1/models/sp15k/qrnn_avg_8.m       0.95525  0.165120         0.957  0.164723
17  data/mldoc/es-1/models/sp15k/qrnn_avg_9.m       0.95725  0.151323         0.960  0.156123
18  data/mldoc/it-1/models/sp15k/qrnn_avg_1.m       0.90175  0.312996         0.900  0.297118
19  data/mldoc/it-1/models/sp15k/qrnn_avg_2.m       0.90250  0.316763         0.903  0.274423
20  data/mldoc/it-1/models/sp15k/qrnn_avg_3.m       0.89900  0.329157         0.915  0.290098
21  data/mldoc/it-1/models/sp15k/qrnn_avg_4.m       0.90100  0.322112         0.907  0.285727
22  data/mldoc/it-1/models/sp15k/qrnn_avg_5.m       0.90100  0.308545         0.910  0.276683
23  data/mldoc/it-1/models/sp15k/qrnn_avg_6.m       0.90275  0.323594         0.915  0.287004
24  data/mldoc/it-1/models/sp15k/qrnn_avg_7.m       0.89925  0.295158         0.910  0.269167
25  data/mldoc/it-1/models/sp15k/qrnn_avg_9.m       0.90325  0.312664         0.908  0.297471
ds     de-1   es-1   it-1
best  95.97  95.70  89.90
max   96.10  95.90  90.32
avg   95.92  95.74  90.13

## VF60k
                                         name  tst_accuracy  tst_loss  val_accuracy  val_loss
0     data/mldoc/de-1/models/vf60k/qrnn_avg.m       0.95250  0.193797         0.946  0.225316
1   data/mldoc/de-1/models/vf60k/qrnn_avg_1.m       0.95575  0.157327         0.947  0.189885
2   data/mldoc/de-1/models/vf60k/qrnn_avg_2.m       0.95400  0.174519         0.947  0.201792
3   data/mldoc/de-1/models/vf60k/qrnn_avg_3.m       0.95325  0.180489         0.947  0.208106
4   data/mldoc/de-1/models/vf60k/qrnn_avg_4.m       0.95425  0.161056         0.949  0.199169
5   data/mldoc/de-1/models/vf60k/qrnn_avg_5.m       0.94775  0.182012         0.941  0.210262
6   data/mldoc/de-1/models/vf60k/qrnn_avg_6.m       0.95375  0.164578         0.947  0.198632
7   data/mldoc/de-1/models/vf60k/qrnn_avg_7.m       0.95575  0.152596         0.947  0.196844
8   data/mldoc/de-1/models/vf60k/qrnn_avg_8.m       0.95350  0.167661         0.942  0.203538
9     data/mldoc/es-1/models/vf60k/qrnn_avg.m       0.95950  0.146121         0.961  0.161852
10  data/mldoc/es-1/models/vf60k/qrnn_avg_1.m       0.95500  0.154836         0.960  0.176217
11  data/mldoc/es-1/models/vf60k/qrnn_avg_2.m       0.95850  0.154539         0.961  0.163008
12  data/mldoc/es-1/models/vf60k/qrnn_avg_3.m       0.96100  0.151916         0.966  0.169869
13  data/mldoc/es-1/models/vf60k/qrnn_avg_4.m       0.95825  0.144630         0.962  0.144410
14  data/mldoc/es-1/models/vf60k/qrnn_avg_5.m       0.95675  0.155685         0.960  0.175439
15  data/mldoc/es-1/models/vf60k/qrnn_avg_6.m       0.95900  0.143995         0.959  0.164156
16  data/mldoc/es-1/models/vf60k/qrnn_avg_7.m       0.95800  0.144662         0.962  0.162957
17  data/mldoc/es-1/models/vf60k/qrnn_avg_8.m       0.95850  0.149185         0.962  0.163159
18    data/mldoc/it-1/models/vf60k/qrnn_avg.m       0.89925  0.320389         0.912  0.272104
19  data/mldoc/it-1/models/vf60k/qrnn_avg_1.m       0.90525  0.305978         0.920  0.255507
20  data/mldoc/it-1/models/vf60k/qrnn_avg_2.m       0.90725  0.287647         0.917  0.245568
21  data/mldoc/it-1/models/vf60k/qrnn_avg_3.m       0.89925  0.313870         0.910  0.271480
22  data/mldoc/it-1/models/vf60k/qrnn_avg_4.m       0.91125  0.285618         0.915  0.255942
23  data/mldoc/it-1/models/vf60k/qrnn_avg_5.m       0.91100  0.288841         0.911  0.255724
24  data/mldoc/it-1/models/vf60k/qrnn_avg_6.m       0.90525  0.287412         0.914  0.253394
25  data/mldoc/it-1/models/vf60k/qrnn_avg_7.m       0.90000  0.308104         0.910  0.256991
26  data/mldoc/it-1/models/vf60k/qrnn_avg_8.m       0.90450  0.301262         0.918  0.251368
ds     de-1   es-1   it-1
best  95.42  96.10  90.53
max   95.57  96.10  91.12
avg   95.34  95.83  90.48



# IT 
## VF60k - 9 runs eval 
for i in 1 2 3 4 5 6 7 8; do python -m ulmfit eval --glob="mldoc/${LANG}-1/models/vf60k/qrnn_nl4.m" --name avg_$i --num-cls-epochs=4 --label-smoothing-eps=0.1 --lr_sched=1cycle --bs=10; done
python -m ulmfit eval --glob="mldoc/${LANG}-1/models/vf60k/qrnn_nl4*.m" --train=False

                                        name  tst_accuracy  tst_loss  val_accuracy  val_loss
0    data/mldoc/it-1/models/vf60k/qrnn_nl4.m       0.89925  0.320389         0.912  0.272104
1  data/mldoc/it-1/models/vf60k/qrnn_nl4_1.m       0.90525  0.305978         0.920  0.255507
2  data/mldoc/it-1/models/vf60k/qrnn_nl4_2.m       0.90725  0.287647         0.917  0.245568
3  data/mldoc/it-1/models/vf60k/qrnn_nl4_3.m       0.89925  0.313870         0.910  0.271480
4  data/mldoc/it-1/models/vf60k/qrnn_nl4_4.m       0.91125  0.285618         0.915  0.255942
5  data/mldoc/it-1/models/vf60k/qrnn_nl4_5.m       0.91100  0.288841         0.911  0.255724
6  data/mldoc/it-1/models/vf60k/qrnn_nl4_6.m       0.90525  0.287412         0.914  0.253394
7  data/mldoc/it-1/models/vf60k/qrnn_nl4_7.m       0.90000  0.308104         0.910  0.256991
8  data/mldoc/it-1/models/vf60k/qrnn_nl4_8.m       0.90450  0.301262         0.918  0.251368
ds     it-1
best  90.53
max   91.12
avg   90.48

## sp15k - 9 runs eval
LANG=it
for i in 1 2 3 4 5 6 7 9; do python -m ulmfit eval --glob="mldoc/${LANG}-1/models/sp15k/qrnn_nl4.m" --name avg_$i --num-cls-epochs=4 --label-smoothing-eps=0.1 --lr_sched=1cycle --bs=18; done
python -m ulmfit eval --glob="mldoc/${LANG}-1/models/sp15k/qrnn_nl4_a*.m" --train=False
                                        name  tst_accuracy  tst_loss  val_accuracy  val_loss
0  data/mldoc/it-1/models/sp15k/qrnn_avg_1.m       0.90175  0.312996         0.900  0.297118
1  data/mldoc/it-1/models/sp15k/qrnn_avg_2.m       0.90250  0.316763         0.903  0.274423
2  data/mldoc/it-1/models/sp15k/qrnn_avg_3.m       0.89900  0.329157         0.915  0.290098
3  data/mldoc/it-1/models/sp15k/qrnn_avg_4.m       0.90100  0.322112         0.907  0.285727
4  data/mldoc/it-1/models/sp15k/qrnn_avg_5.m       0.90100  0.308545         0.910  0.276683
5  data/mldoc/it-1/models/sp15k/qrnn_avg_6.m       0.90275  0.323594         0.915  0.287004
6  data/mldoc/it-1/models/sp15k/qrnn_avg_7.m       0.89925  0.295158         0.910  0.269167
7  data/mldoc/it-1/models/sp15k/qrnn_avg_9.m       0.90325  0.312664         0.908  0.297471
ds     it-1
best  89.90
max   90.32
avg   90.13


# ES
## VF60k - 8 runs eval
for i in 1 2 3 4 5 6 7 8; do python -m ulmfit eval --glob="mldoc/${LANG}-1/models/vf60k/qrnn_nl4.m" --name avg_$i --num-cls-epochs=4 --label-smoothing-eps=0.1 --lr_sched=1cycle --bs=10; done
python -m ulmfit eval --glob="mldoc/${LANG}-1/models/vf60k/qrnn_nl4*.m" --train=False

                                        name  tst_accuracy  tst_loss  val_accuracy  val_loss
0    data/mldoc/es-1/models/vf60k/qrnn_nl4.m       0.95950  0.146121         0.961  0.161852
1  data/mldoc/es-1/models/vf60k/qrnn_nl4_1.m       0.95500  0.154836         0.960  0.176217
2  data/mldoc/es-1/models/vf60k/qrnn_nl4_2.m       0.95850  0.154539         0.961  0.163008
3  data/mldoc/es-1/models/vf60k/qrnn_nl4_3.m       0.96100  0.151916         0.966  0.169869
4  data/mldoc/es-1/models/vf60k/qrnn_nl4_4.m       0.95825  0.144630         0.962  0.144410
5  data/mldoc/es-1/models/vf60k/qrnn_nl4_5.m       0.95675  0.155685         0.960  0.175439
6  data/mldoc/es-1/models/vf60k/qrnn_nl4_6.m       0.95900  0.143995         0.959  0.164156
7  data/mldoc/es-1/models/vf60k/qrnn_nl4_7.m       0.95800  0.144662         0.962  0.162957
8  data/mldoc/es-1/models/vf60k/qrnn_nl4_8.m       0.95850  0.149185         0.962  0.163159
ds     es-1
best  96.10
max   96.10
avg   95.83


## sp15k - 8 runs eval 
LANG=es
for i in 1 2 3 4 5 6 7 9; do python -m ulmfit eval --glob="mldoc/${LANG}-1/models/sp15k/qrnn_nl4.m" --name avg_$i --num-cls-epochs=4 --label-smoothing-eps=0.1 --lr_sched=1cycle --bs=18; done
python -m ulmfit eval --glob="mldoc/${LANG}-1/models/sp15k/qrnn_avg*.m" --train=False
                                        name  tst_accuracy  tst_loss  val_accuracy  val_loss
0  data/mldoc/es-1/models/sp15k/qrnn_avg_1.m       0.95700  0.149081         0.963  0.162808
1  data/mldoc/es-1/models/sp15k/qrnn_avg_2.m       0.95875  0.141572         0.963  0.150970
2  data/mldoc/es-1/models/sp15k/qrnn_avg_3.m       0.95900  0.151387         0.962  0.157047
3  data/mldoc/es-1/models/sp15k/qrnn_avg_4.m       0.95375  0.165935         0.956  0.182161
4  data/mldoc/es-1/models/sp15k/qrnn_avg_5.m       0.95850  0.151109         0.960  0.156376
5  data/mldoc/es-1/models/sp15k/qrnn_avg_6.m       0.95800  0.150724         0.961  0.152761
6  data/mldoc/es-1/models/sp15k/qrnn_avg_7.m       0.95875  0.142476         0.963  0.151585
7  data/mldoc/es-1/models/sp15k/qrnn_avg_8.m       0.95525  0.165120         0.957  0.164723
8  data/mldoc/es-1/models/sp15k/qrnn_avg_9.m       0.95725  0.151323         0.960  0.156123
ds     es-1
best  95.70
max   95.90
avg   95.74

# DE
## VF60k - 9 runs eval
for i in 1 2 3 4 5 6 7 8; do python -m ulmfit eval --glob="mldoc/${LANG}-1/models/vf60k/qrnn_nl4.m" --name avg_$i --num-cls-epochs=4 --label-smoothing-eps=0.1 --lr_sched=1cycle --bs=10; done
python -m ulmfit eval --glob="mldoc/${LANG}-1/models/vf60k/qrnn_nl4*.m" --train=False
                                        name  tst_accuracy  tst_loss  val_accuracy  val_loss
0    data/mldoc/de-1/models/vf60k/qrnn_nl4.m       0.95250  0.193797         0.946  0.225316
1  data/mldoc/de-1/models/vf60k/qrnn_nl4_1.m       0.95575  0.157327         0.947  0.189885
2  data/mldoc/de-1/models/vf60k/qrnn_nl4_2.m       0.95400  0.174519         0.947  0.201792
3  data/mldoc/de-1/models/vf60k/qrnn_nl4_3.m       0.95325  0.180489         0.947  0.208106
4  data/mldoc/de-1/models/vf60k/qrnn_nl4_4.m       0.95425  0.161056         0.949  0.199169
5  data/mldoc/de-1/models/vf60k/qrnn_nl4_5.m       0.94775  0.182012         0.941  0.210262
6  data/mldoc/de-1/models/vf60k/qrnn_nl4_6.m       0.95375  0.164578         0.947  0.198632
7  data/mldoc/de-1/models/vf60k/qrnn_nl4_7.m       0.95575  0.152596         0.947  0.196844
8  data/mldoc/de-1/models/vf60k/qrnn_nl4_8.m       0.95350  0.167661         0.942  0.203538
ds     de-1
best  95.42
max   95.57
avg   95.34

## sp15k - 8 runs eval 
for i in 1 2 3 4 5 6 7 9; do python -m ulmfit eval --glob="mldoc/${LANG}-1/models/sp15k/qrnn_nl4.m" --name avg_$i --num-cls-epochs=4 --label-smoothing-eps=0.1 --lr_sched=1cycle --bs=18; done
python -m ulmfit eval --glob="mldoc/${LANG}-1/models/sp15k/qrnn_avg*.m" --train=False
                                        name  tst_accuracy  tst_loss  val_accuracy  val_loss
0  data/mldoc/de-1/models/sp15k/qrnn_avg_1.m       0.95700  0.142444         0.945  0.206329
1  data/mldoc/de-1/models/sp15k/qrnn_avg_2.m       0.95975  0.141225         0.955  0.189209
2  data/mldoc/de-1/models/sp15k/qrnn_avg_3.m       0.95925  0.140172         0.946  0.198159
3  data/mldoc/de-1/models/sp15k/qrnn_avg_4.m       0.95650  0.145889         0.942  0.193202
4  data/mldoc/de-1/models/sp15k/qrnn_avg_5.m       0.96000  0.137710         0.953  0.192368
5  data/mldoc/de-1/models/sp15k/qrnn_avg_6.m       0.95975  0.145318         0.954  0.196413
6  data/mldoc/de-1/models/sp15k/qrnn_avg_7.m       0.95925  0.141719         0.943  0.199226
7  data/mldoc/de-1/models/sp15k/qrnn_avg_8.m       0.96100  0.140644         0.949  0.204398
8  data/mldoc/de-1/models/sp15k/qrnn_avg_9.m       0.96050  0.143330         0.949  0.189472
ds     de-1
best  95.97
max   96.10
avg   95.92

# RU
## VF60k - 9 runs eval
for i in 1 2 3 4 5 6 7 8; do python -m ulmfit eval --glob="mldoc/${LANG}-1/models/vf60k/qrnn_nl4.m" --name nl4_$i --num-cls-epochs=4 --label-smoothing-eps=0.1 --lr_sched=1cycle --bs=10; done
python -m ulmfit eval --glob="mldoc/${LANG}-1/models/vf60k/qrnn_nl4*.m" --train=False
