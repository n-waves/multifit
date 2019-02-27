## LM
```bash
CUDA_VISIBLE_DEVICES=0 python -m ulmfit lm --dataset-path data/wiki/${LANG}-100 --tokenizer='sp' --nl 5 --name 'nl5-mer
ity' --max-vocab 15000 --lang ${LANG} --qrnn=True --bptt=140 --nh 2500 - train 14 --bs=50 --drop_mult=0  --label-smoothing-eps=0.1
Max vocab: 15000
Cache dir: data/wiki/ru-100/models/sp15k
Model dir: data/wiki/ru-100/models/sp15k/qrnn_nl5-merity.m
Wiki text was split to 193047 articles
Wiki text was split to 460 articles
Data lm, trn: 193047, val: 460
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁.', '▁в', 'а', 'и', 'е', '▁и', 'й', '▁на', 'х']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Training lm from random weights
epoch     train_loss  valid_loss  accuracy
1         3.969639    4.024787    0.452887
2         3.814622    3.834142    0.476612
3         3.798372    3.846118    0.473666
4         3.742609    3.835311    0.474612
5         3.715114    3.790690    0.480469
6         3.652987    3.748408    0.486146
7         3.573350    3.697325    0.493774
8         3.589853    3.637134    0.504189
9         3.558110    3.583030    0.512137
10        3.501382    3.510491    0.524148
11        3.408982    3.437177    0.536634
12        3.402717    3.373548    0.548113
13        3.293624    3.331311    0.556288
14        3.309859    3.322777    0.558426
Total time: 68:05:15
data/wiki/ru-100/models/sp15k
Saving info data/wiki/ru-100/models/sp15k/qrnn_nl5-merity.m/info.json
```

### MLDoc 1 
```

```