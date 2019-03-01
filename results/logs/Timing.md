
## QRNN sp15k timing
```
time python -m ulmfit lm --dataset-path data/wiki/${LANG}-100 --tokenizer='sp' --nl 4 --name 'nl4-2' --max-vocab 15000 --lang ${LANG} --qrnn=True - train 1 --bs=50 --drop_mult=0  --label-smoothing-eps=0.1
Max vocab: 15000
Cache dir: data/wiki/ru-100/models/sp15k
Model dir: data/wiki/ru-100/models/sp15k/qrnn_nl4-2.m
Wiki text was split to 193047 articles
Wiki text was split to 460 articles
Data lm, trn: 193047, val: 460
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁.', '▁в', 'а', 'и', 'е', '▁и', 'й', '▁на', 'х']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Training lm from random weights
epoch     train_loss  valid_loss  accuracy
1         3.943105    3.860063    0.477620
Total time: 1:05:03
data/wiki/ru-100/models/sp15k
Saving info data/wiki/ru-100/models/sp15k/qrnn_nl4-2.m/info.json

real    65m30,341s
user    48m49,047s
sys     16m40,688s
```