```
LANG=de                                                                                   
python -m ulmfit lm --dataset-path data/wiki/${LANG}-100 --tokenizer='fsp' --nl 4 --name 'nl4' --max-vocab 30000 --lang ${LANG} --qrnn=True --lmseed=1  --nh=1552 - train 10 --bs=50 --drop_mult=0 --label-smoothing-eps=0.1
Training lm
Max vocab: 30000
Cache dir: data/wiki/de-100/models/fsp30k
Model dir: data/wiki/de-100/models/fsp30k/qrnn_nl4_lmseed-1.m
Setting LM seed to 1
Wiki text was split to 191112 articles
Wiki text was split to 431 articles
Data lm, trn: 191112, val: 431
Size of vocabulary: 30000
First 20 words in vocab: ['▁xxunk', '▁xxpad', '▁xxbos', '▁xxeos', '▁xxfld', '▁xxmaj', '▁xxup', '▁xxrep', '▁xxwrep', '<unk>', '▁', '▁.', '▁,', '▁der', '▁die', '▁und', '▁in', "▁&'", 's', '-']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0} dps:  {'input_p': 0.25, 'output_p': 0.1, 'weight_p': 0.2, 'embed_p': 0.02, 'hidden_p': 0.15}
Bptt 70
Training lm from random weights
epoch     train_loss  valid_loss  accuracy  time
0         4.403168    4.469939    0.403841  1:30:29
1         4.312332    4.432437    0.404200  1:30:30
2         4.334051    4.423142    0.405613  1:30:03
3         4.239668    4.376138    0.411162  1:30:05
4         4.193250    4.324453    0.416959  1:30:04
5         4.151386    4.248287    0.427130  1:30:16
6         4.103295    4.160756    0.438965  1:30:32
7         4.033971    4.086159    0.450292  1:30:16
8         3.967596    4.029943    0.459819  1:31:20
9         3.954203    4.013039    0.463228  1:31:14
Total time: 15:04:52
data/wiki/de-100/models/fsp30k
Saving info data/wiki/de-100/models/fsp30k/qrnn_nl4_lmseed-1.m/info.json
```
-------
