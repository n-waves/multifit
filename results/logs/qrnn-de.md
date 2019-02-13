# QRNN DE
## SP30k nl
### LM
```
python -m ulmfit lm --dataset-path data/wiki/de-100 --bidir=False --qrnn=True --nl 4 --tokenizer='sp' --max-vocab 30000 --lang de --name 'nl4' --cuda-id=0 - train 10 --drop-mult=0 --bs=50

Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁.', '▁,', '▁der', '▁die', '▁und', '▁in', "▁&'", 'en', 's', '-']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Training lm from random weights
epoch     train_loss  valid_loss  accuracy
1         2.790653    2.867094    0.511392
2         2.742032    2.843288    0.510885
3         2.696114    2.833874    0.512062
4         2.671780    2.786312    0.516448
5         2.611292    2.725993    0.522723
6         2.542737    2.655713    0.530968
7         2.572076    2.582141    0.539928
8         2.465960    2.509654    0.549987
9         2.405682    2.448580    0.558674
10        2.339395    2.428111    0.562502
```
