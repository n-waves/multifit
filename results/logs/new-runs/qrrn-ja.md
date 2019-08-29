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

python -m ulmfit cls --dataset-path data/mldoc/${LANG}-1 --base-lm-path data/wiki/${LANG}-100/models/fsp30k/qrnn_nl4_lmseed-1.m --lang=${LANG} --name 'nl4' --clsweightseed=0 - train 20 --bs 20 --lr_sched=1cycle --label-smoothing-eps=0.1 