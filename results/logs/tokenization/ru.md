```
LANG=ru
python -m ulmfit lm --dataset-path data/wiki/${LANG}-100 --tokenizer='vf' --nl 4 --name 'nl4' --max-vocab 60000 --lang ${LANG} --qrnn=True - train 10 --bs=50 --drop_mult=0  --label-smoothing-eps=0.1
Max vocab: 60000
Cache dir: data/wiki/ru-100/models/vf60k
Model dir: data/wiki/ru-100/models/vf60k/qrnn_nl4.m
Wiki text was split to 193047 articles
Wiki text was split to 460 articles
Running tokenization lm...
Data lm, trn: 193047, val: 460
Size of vocabulary: 60003
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', ',', '\n', '.', 'в', 'и', ')', '(', 'на', '—', '«', '»', 'с']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Training lm from random weights
epoch     train_loss  valid_loss  accuracy
1         4.586900    4.478803    0.413023
2         4.483496    4.400461    0.418495
3         4.484620    4.390928    0.418422
4         4.373594    4.350045    0.422567
5         4.350337    4.307665    0.427411
6         4.314571    4.249700    0.436324
7         4.232540    4.183857    0.446341
8         4.252573    4.119820    0.455522
9         4.136978    4.088805    0.462345
10        4.116755    4.079840    0.465394
Total time: 11:24:03
data/wiki/ru-100/models/vf60k
```

