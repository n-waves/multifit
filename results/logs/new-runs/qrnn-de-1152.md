```
LANG=de
python -m multifit lm --dataset-path data/wiki/${LANG}-100 --tokenizer='fsp' --nl 4 --name '1152' --max-vocab 30000 --lang ${LANG} --qrnn=True --lmseed=1  --nh=1152 - train 10 --bs=50 --drop_mult=0 --label-smoothing-eps=0.1
Training lm
Max vocab: 30000
Cache dir: data/wiki/de-100/models/fsp30k
Model dir: data/wiki/de-100/models/fsp30k/qrnn_1152_lmseed-1.m
Setting LM seed to 1
Wiki text was split to 191112 articles
Wiki text was split to 431 articles
Data lm, trn: 191112, val: 431
Size of vocabulary: 30000
First 20 words in vocab: ['▁xxunk', '▁xxpad', '▁xxbos', '▁xxeos', '▁xxfld', '▁xxmaj', '▁xxup', '▁xxrep', '▁xxwrep', '<unk>', '▁', '▁.', '▁,', '▁der', '▁die', '▁und', '▁in', "▁&'", 's', '-']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0} config:  {'emb_sz': 400, 'n_hid': 1152, 'n_layers': 4, 'pad_token': 1, 'qrnn': True, 'bidir': False, 'output_p': 0.1, 'hidden_p': 0.15, 'input_p': 0.25, 'embed_p': 0.02, 'weight_p': 0.2, 'tie_weights': True, 'out_bias': True}
Bptt 70
Training lm from random weights
epoch     train_loss  valid_loss  accuracy  time
0         4.452324    4.517938    0.397063  1:12:10
1         4.354969    4.473063    0.399108  1:12:08
2         4.375927    4.460905    0.400854  1:11:47
3         4.279962    4.406254    0.406279  1:11:47
4         4.229251    4.358620    0.413019  1:11:51
5         4.194514    4.380852    0.409330  1:11:54
6         4.151179    4.204870    0.431857  1:11:57
7         4.085239    4.131142    0.442781  1:12:28
8         4.026665    4.080124    0.451819  1:13:08
9         4.015738    4.063066    0.454933  1:13:28
Total time: 12:02:42
data/wiki/de-100/models/fsp30k
Saving info data/wiki/de-100/models/fsp30k/qrnn_1152_lmseed-1.m/info.json
```