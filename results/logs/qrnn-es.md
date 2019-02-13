# QRNN ES

## SP30k nl 4
### LM
```
python -m ulmfit lm --dataset-path data/wiki/es-100 --bidir=False --qrnn=True --nl 4 --tokenizer='sp' --max-vocab 30000 --lang es --name 'nl4' --cuda-id=0  -  train 10 --drop-mult=0 --bs=50

Wiki text was split to 161509 articles
Wiki text was split to 78 articles
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁.', '▁la', '▁el', '▁en', '▁y', 's', '▁a', "▁&'"]
Training args:  {'clip': 0.12, 'drop_mult': 0} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Training lm from random weights
epoch     train_loss  valid_loss  accuracy
1         3.067289    3.640350    0.357276
2         2.958243    3.619773    0.358111
3         3.033412    3.587700    0.359495
4         2.933573    3.525202    0.367685
5         2.904549    3.467990    0.372583
6         2.798806    3.409506    0.380045
7         2.733132    3.303108    0.391922
8         2.675272    3.224150    0.401143
9         2.635299    3.166430    0.410160
10        2.656724    3.145599    0.413176
```
