python -m ulmfit cls --dataset-path data/mldoc/${LANG}-1 --base-lm-path data/wiki/${LANG}-100/models/sp15k/qrnn_${NAME}.m  --lang=${LANG} --name ${NAME}-2 - train 20 --bs 18 --num-cls-epochs=4 --lr_sched=1cycle --label-smoothing-eps=0.1
Max vocab: 15000
Cache dir: /home/n-waves/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp15k
Model dir: /home/n-waves/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp15k/qrnn_nl8-2.m
Loading validation /home/n-waves/workspace/ulmfit-multilingual/data/mldoc/ru-1/ru.dev.csv
Data lm, trn:  9195, val: 1021
Data cls, trn: 1000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁.', '▁в', 'а', 'и', 'е', '▁и', 'й', '▁на', 'х']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
Training lm from:  [PosixPath('/home/n-waves/workspace/ulmfit-multilingual/data/wiki/ru-100/models/sp15k/qrnn_nl8.m/lm_best'), PosixPath('/home/n-waves/workspace/ulmfit-multilingual/data/wiki/ru-100/models/sp15k/qrnn_nl8.m/../itos')]
epoch     train_loss  valid_loss  accuracy
1         3.966292    3.450758    0.527065
Total time: 02:58
epoch     train_loss  valid_loss  accuracy
1         3.495761    3.276329    0.560047
2         3.319947    3.102911    0.593742
3         3.137904    2.955317    0.620171
4         3.040286    2.839161    0.642270
5         2.869962    2.753622    0.658331
6         2.905739    2.680881    0.672860
7         2.836454    2.620925    0.685026
8         2.857271    2.569716    0.695722
9         2.702872    2.520050    0.705589
10        2.701559    2.473591    0.715346
11        2.740815    2.429558    0.725597
12        2.646513    2.389550    0.735010
13        2.587685    2.349614    0.744885
14        2.546527    2.311087    0.754463
15        2.568136    2.278581    0.762980
16        2.492115    2.252367    0.769275
17        2.338561    2.230529    0.775072
18        2.447506    2.218215    0.778437
19        2.364424    2.212115    0.780085
20        2.367132    2.210520    0.780424
Total time: 1:30:47
/home/n-waves/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp15k
Saving info /home/n-waves/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp15k/qrnn_nl8-2.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.969255    0.769736    0.843000
2         0.846340    0.813483    0.839000
3         0.718175    0.705339    0.867000
4         0.609513    0.726442    0.875000
Total time: 02:54
Saving models at /home/n-waves/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp15k/qrnn_nl8-2.m
Loss and accuracy using (cls_best): [0.4056449, tensor(0.8698)]
0.40564489364624023



(fastaiv1) n-waves@GV100:~/workspace/ulmfit-multilingual$ CUDA_VISIBLE_DEVICES=0 python -m ulmfit lm --dataset-path data/wiki/${LANG}-100 --tokenizer='sp' --nl 4 --name 'nl4-merity-wide2' --max-vocab 15000 --lang ${LANG} --qrnn=True --bptt=140 --nh 3100 - train 10 --bs=50 --drop_mult=0  --label-smoothing-eps=0.1
Max vocab: 15000
Cache dir: data/wiki/ru-100/models/sp15k
Model dir: data/wiki/ru-100/models/sp15k/qrnn_nl4-merity-wide2.m
Wiki text was split to 193047 articles
Wiki text was split to 460 articles
Data lm, trn: 193047, val: 460
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁.', '▁в', 'а', 'и', 'е', '▁и', 'й', '▁на', 'х']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Training lm from random weights
epoch     train_loss  valid_loss  accuracy
1         3.871650    3.908779    0.467000
2         3.834093    3.884629    0.467916
3         3.741005    3.870331    0.469612
4         3.785444    3.818511    0.476906
5         3.741888    3.752743    0.486148
6         3.678481    3.672177    0.499054
7         3.570398    3.581498    0.512801
8         3.455193    3.482614    0.530569
9         3.379779    3.409405    0.543477
10        3.384574    3.387195    0.548881
Total time: 27:24:33
data/wiki/ru-100/models/sp15k