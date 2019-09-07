```
LANG=ja                                                                                 ✘ 130
python -m ulmfit lm --dataset-path data/wiki/${LANG}-100 --tokenizer='fsp' --nl 4 --name 'nl4' --max-vocab 30000 --lang ${LANG} --qrnn=True --lmseed=1  --nh=1552 - train 10 --bs=50 --drop_mult=0 --label-smoothing-eps=0.1
Training lm
Max vocab: 30000
Cache dir: data/wiki/ja-100/models/fsp30k
Model dir: data/wiki/ja-100/models/fsp30k/qrnn_nl4_lmseed-1.m
Setting LM seed to 1
Wiki text was split to 120037 articles
Wiki text was split to 63 articles
Data lm, trn: 120037, val: 63
Size of vocabulary: 30000
First 20 words in vocab: ['▁xxunk', '▁xxpad', '▁xxbos', '▁xxeos', '▁xxfld', '▁xxmaj', '▁xxup', '▁xxrep', '▁xxwrep', '<unk>', '▁', '▁、', '▁。', '▁の', '▁に', '▁を', '▁年', '▁は', '▁・', '▁(']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0} dps:  {'input_p': 0.25, 'output_p': 0.1, 'weight_p': 0.2, 'embed_p': 0.02, 'hidden_p': 0.15}
Bptt 70
Training lm from random weights
epoch     train_loss  valid_loss  accuracy  time
0         4.346260    4.408598    0.370535  1:12:49
1         4.253693    4.355113    0.372354  1:12:40
2         4.190918    4.288729    0.383211  1:12:15
3         4.148739    4.242265    0.389964  1:12:12
4         4.136361    4.190885    0.398423  1:12:23
5         4.051008    4.119476    0.409002  1:12:15
6         3.966213    4.052222    0.419292  1:12:20
7         3.928247    3.979634    0.431336  1:12:18
8         3.840935    3.929402    0.442688  1:12:39
9         3.909105    3.911067    0.446342  1:12:51
Total time: 12:04:47
data/wiki/ja-100/models/fsp30k
Saving info data/wiki/ja-100/models/fsp30k/qrnn_nl4_lmseed-1.m/info.json
```
-------

```
python -m ulmfit cls --dataset-path data/mldoc/${LANG}-1 --base-lm-path data/wiki/${LANG}-100/models/fsp30k/qrnn_nl4_lmseed-1.m --lang=${LANG} --name 'nl4' --clsweightseed=0 - train 20 --bs 20 --lr_sched=1cycle --label-smoothing-eps=0.1
data/mldoc/ja-1 'data/wiki/ja-100/models/fsp30k/qrnn_nl4_lmseed-1.m'
Training CLS
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ja-1/models/fsp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ja-1/models/fsp30k/qrnn_nl4_lmseed-1-clsweightseed-0.m
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ja-1/ja.dev.csv
Running tokenization lm-notst...
Data lm-notst, trn: 13500, val: 1500
Running tokenization cls...
Data cls, trn: 1000, val: 1000
Running tokenization tst...
Data tst, trn: 1000, val: 4000
Size of vocabulary: 30000
First 20 words in vocab: ['▁xxunk', '▁xxpad', '▁xxbos', '▁xxeos', '▁xxfld', '▁xxmaj', '▁xxup', '▁xxrep', '▁xxwrep', '<unk>', '▁', '▁、', '▁。', '▁の', '▁に', '▁を', '▁年', '▁は', '▁・', '▁(']
Training lm
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ja-1/models/fsp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ja-1/models/fsp30k/qrnn_nl4_lmseed-1-clsweightseed-0.m
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'input_p': 0.25, 'output_p': 0.1, 'weight_p': 0.2, 'embed_p': 0.02, 'hidden_p': 0.15}
Loading pretrained model
Bptt 70
Training lm from:  [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/wiki/ja-100/models/fsp30k/qrnn_nl4_lmseed-1.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/wiki/ja-100/models/fsp30k/qrnn_nl4_lmseed-1.m/../itos')]
epoch     train_loss  valid_loss  accuracy  time
0         4.921511    4.071093    0.449957  02:49
Total time: 02:49
epoch     train_loss  valid_loss  accuracy  time
0         4.078950    3.772346    0.490980  03:45
1         3.727190    3.439734    0.550465  03:47
2         3.359830    3.201281    0.590776  03:47
3         3.319000    3.039330    0.618913  03:48
4         3.127082    2.923124    0.637883  03:49
5         3.004416    2.842367    0.652876  03:49
6         2.940764    2.775950    0.664768  03:49
7         2.961673    2.723808    0.672876  03:48
8         2.895331    2.680753    0.681351  03:49
9         2.804141    2.642330    0.688681  03:48
10        2.867590    2.607323    0.695910  03:49
11        2.755013    2.570714    0.703812  03:48
12        2.747998    2.539443    0.710536  03:49
13        2.710947    2.512321    0.716777  03:49
14        2.675630    2.486792    0.721944  03:49
15        2.654088    2.466209    0.726944  03:50
16        2.668594    2.452018    0.730059  03:49
17        2.640188    2.444180    0.731973  03:47
18        2.562034    2.439116    0.732988  03:43
19        2.641966    2.438401    0.733259  03:42
Total time: 1:16:04
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ja-1/models/fsp30k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ja-1/models/fsp30k/qrnn_nl4_lmseed-1-clsweightseed-0.m/info.json
Setting classifier weights seed to 0
Single training schedule
epoch     train_loss  valid_loss  f_beta    precision  recall    kappa_score  matthews_correff  accuracy  time
/home/pczapla/workspace/_oss/fastai/fastai/fastai/metrics.py:189: UserWarning: average=`binary` was selected for a non binary case. Value for average has now been set to `macro` instead.
  warn("average=`binary` was selected for a non binary case. Value for average has now been set to `macro` instead.")
0         0.866883    0.690708    0.826320  0.859738   0.829345  0.770440     0.781947          0.828000  00:13
Better model found at epoch 0 with f_beta value: 0.826319694519043.
/home/pczapla/workspace/_oss/fastai/fastai/fastai/metrics.py:189: UserWarning: average=`binary` was selected for a non binary case. Value for average has now been set to `macro` instead.
  warn("average=`binary` was selected for a non binary case. Value for average has now been set to `macro` instead.")
1         0.676495    0.609408    0.881395  0.888526   0.874378  0.831782     0.835147          0.874000  00:13
Better model found at epoch 1 with f_beta value: 0.8813954591751099.
Total time: 00:27
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ja-1/models/fsp30k/qrnn_nl4_lmseed-1-clsweightseed-0.m
/home/pczapla/workspace/_oss/fastai/fastai/fastai/metrics.py:189: UserWarning: average=`binary` was selected for a non binary case. Value for average has now been set to `macro` instead.
  warn("average=`binary` was selected for a non binary case. Value for average has now been set to `macro` instead.")
Model: nl4
Evaluation on: test
F1 score bin: 0.8911986351013184
Loss: 0.32485464215278625
Precision: 0.8969309329986572
Recall: 0.8911643028259277
Accuracy: 0.8914999961853027
test F1 score bin:     0.8911986351013184
test Loss:             0.32485464
test Precision:        0.8969309329986572
test Recall:           0.8911643028259277
test Kappa Linear:     0.8553637266159058
test Matthews Correff: 0.8571781516075134
test Accuracy:         0.8914999961853027
```
 
----
```
for seed in  1 2 3 4 5 ; do 
    python -m multifit eval --glob="mldoc/${LANG}-1/models/fsp30k/qrnn_nl4_lmseed-1-clsweightseed-0.m" --name nl4  --clsweightseed=$seed --num-cls-epochs=8 --bs=18 --lr_sched=1cycle --label-smoothing-eps=0.1
done
```