# IT 
## SP30k QRNN nl 4 
### LM 
```
python -m ulmfit lm --dataset-path data/wiki/it-100/ --cuda-id=0 --tokenizer='sp' --nl 4 --name 'nl4' --max-vocab 30000 --lang it --qrnn=True - train 10 --bs=50 --drop_mult=0
Max vocab: 30000
Cache dir: data/wiki/it-100/models/sp30k
Model dir: data/wiki/it-100/models/sp30k/qrnn_nl4.m
Tokenized data loaded
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁.', '▁di', "▁&'", "'", '▁e', '▁il', '▁la', 'e', '▁in']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Training lm from random weights
epoch     train_loss  valid_loss  accuracy
1         3.354224    3.749085    0.350236
2         3.274838    3.697026    0.351104
3         3.222462    3.680071    0.352152
4         3.217652    3.628976    0.357922
5         3.117965    3.563592    0.364370
6         3.075397    3.483997    0.372794
7         3.002098    3.394749    0.383217
8         2.936974    3.316284    0.393616
9         2.843549    3.258448    0.401605
10        2.818070    3.240303    0.404684
Total time: 10:49:44
data/wiki/it-100/models/sp30k
Saving info data/wiki/it-100/models/sp30k/qrnn_nl4.m/info.json
```

### MLDoc
