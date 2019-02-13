# QRNN EN
## SP30k nl 4
### LM

```
python -m ulmfit lm --dataset-path data/wiki/wikitext-103 --bidir=False --qrnn=True --nl 4 --tokenizer='sp' --max-vocab 30000 --lang en --name 'nl4' --cuda-id=1  -  train 10 --drop-mult=0 --bs=50

Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁the', '▁,', '▁.', 's', '▁of', '▁and', '▁in', '▁to', '▁a', 'ed']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.5} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Training lm from random weights
epoch     train_loss  valid_loss  accuracy
1         3.184221    3.256314    0.438527
2         3.084555    3.241498    0.435628
3         3.099060    3.258447    0.435060
4         3.119621    3.220939    0.437597
5         3.073662    3.165012    0.445108
6         2.938047    3.086962    0.452921
7         2.920506    2.998151    0.462940
8         2.920506    2.899240    0.474378
9         2.862836    2.835098    0.485305
10        2.891070    2.810929    0.489867
```

### LM, BS=128, drop-mult=0.5
```
python -m ulmfit lm --dataset-path data/wiki/wikitext-103 --bidir=False --qrnn=True --nl 4 --tokenizer='sp' --max-vocab 30000 --lang en --name 'nl4-bs128' --cuda-id=1  -  train 10 --drop-mult=0.5 --bs=128

Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁the', '▁,', '▁.', 's', '▁of', '▁and', '▁in', '▁to', '▁a', 'ed']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.5} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Training lm from random weights
epoch     train_loss  valid_loss  accuracy
1         3.413345    3.280860    0.433011
2         3.219606    3.129479    0.444172
3         3.136091    3.094905    0.448493
4         3.145281    3.033001    0.452830
5         3.100366    2.980189    0.458984
6         3.062894    2.923044    0.464841
7         3.001627    2.834753    0.475316
8         2.979051    2.792044    0.480915
9         2.933140    2.733279    0.488346
10        2.964397    2.720861    0.490423
```
