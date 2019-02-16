# QRNN RU
## SP30k nl4
### LM
```
python -m ulmfit lm --dataset-path data/wiki/ru-100 --bidir=False --qrnn=True --nl 4 --tokenizer='sp' --max-vocab 30000 --lang ru --name 'nl4' --cuda-id=0  -  train 10 --drop-mult=0 --bs=50

Wiki text was split to 193047 articles
Wiki text was split to 460 articles
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁.', '▁в', 'а', '▁и', 'е', 'и', 'й', '▁на', '▁с']
Training args:  {'clip': 0.12, 'drop_mult': 0} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Training lm from random weights
epoch     train_loss  valid_loss  accuracy
1         3.273207    3.350111    0.429702
2         3.169897    3.274238    0.433682
3         3.162197    3.247077    0.435900
4         3.131630    3.168798    0.445252
5         3.042942    3.096774    0.453532
6         2.950550    3.002989    0.465113
7         2.833593    2.902871    0.478954
8         2.829737    2.805592    0.492138
9         2.746991    2.733609    0.503711
10        2.687201    2.708546    0.508050
```
