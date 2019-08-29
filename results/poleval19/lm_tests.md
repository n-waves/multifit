```
for seed in 3 4 5; do 
    python -m ulmfit lm --dataset-path data/reddit/pl-100 --bidir=False --qrnn=False --nl 4 --tokenizer='sp' --max-vocab 25000 --lang pl --name 'nl4' --lmseed=$seed - train 10 --drop-mult=0 --bs=100
done


for seed in 6 7 8; do 
    python -m ulmfit lm --dataset-path data/reddit/pl-100 --bidir=False --qrnn=False --nl 4 --tokenizer='sp' --max-vocab 25000 --lang pl --name 'nl4' --lmseed=$seed - train 10 --drop-mult=0 --bs=100
done
```

```
for seed in 3 4 5; do
    python -m ulmfit lm --dataset-path data/reddit/pl-100 --bidir=False --qrnn=False --nl 4 --tokenizer='sp' --max-vocab 25000 --lang pl --name 'nl4' --lmseed=$seed - train 10 --drop-mult=0 --bs=100
done

Training lm
Max vocab: 25000
Cache dir: data/reddit/pl-100/models/sp25k
Model dir: data/reddit/pl-100/models/sp25k/lstm_nl4_lmseed-3.m
Setting LM seed to 3
Running tokenization lm...
Data lm, trn: 1083512, val: 27852
Size of vocabulary: 25000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxeos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', 'xxlink', 'xxuser', 'xxemoji', 'yyemoji', '<unk>', '▁', ',', '.', '▁"', '▁to', '▁nie']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0} dps:  {'input_p': 0.25, 'output_p': 0.1, 'weight_p': 0.2, 'embed_p': 0.02, 'hidden_p': 0.15}
Bptt 70
Training lm from random weights
epoch     train_loss  valid_loss  accuracy
1         4.552017    4.536615    0.291548
2         4.136435    4.131667    0.325988
3         4.033742    4.043243    0.333318
4         3.946634    3.958406    0.342276
5         3.842665    3.883467    0.350498
6         3.763694    3.812498    0.358469
7         3.667258    3.751945    0.365176
8         3.546397    3.704516    0.372318
9         3.418958    3.694572    0.374632
10        3.317018    3.707627    0.373888
Total time: 3:51:03
data/reddit/pl-100/models/sp25k
Saving info data/reddit/pl-100/models/sp25k/lstm_nl4_lmseed-3.m/info.json
Training lm
Max vocab: 25000
Cache dir: data/reddit/pl-100/models/sp25k
Model dir: data/reddit/pl-100/models/sp25k/lstm_nl4_lmseed-4.m
Setting LM seed to 4
Data lm, trn: 1083512, val: 27852
Size of vocabulary: 25000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxeos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', 'xxlink', 'xxuser', 'xxemoji', 'yyemoji', '<unk>', '▁', ',', '.', '▁"', '▁to', '▁nie']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0} dps:  {'input_p': 0.25, 'output_p': 0.1, 'weight_p': 0.2, 'embed_p': 0.02, 'hidden_p': 0.15}
Bptt 70
Training lm from random weights
epoch     train_loss  valid_loss  accuracy
1         4.556551    4.538219    0.291983
2         4.141277    4.135662    0.325108
3         4.038784    4.040287    0.333952
4         3.954407    3.953955    0.342720
5         3.866849    3.881778    0.350481
6         3.758424    3.809757    0.358544
7         3.658111    3.746240    0.366357
8         3.529903    3.702708    0.372116
9         3.396303    3.692299    0.374357
10        3.318496    3.705198    0.373649
Total time: 3:50:38
data/reddit/pl-100/models/sp25k
Saving info data/reddit/pl-100/models/sp25k/lstm_nl4_lmseed-4.m/info.json
Training lm
Max vocab: 25000
Cache dir: data/reddit/pl-100/models/sp25k
Model dir: data/reddit/pl-100/models/sp25k/lstm_nl4_lmseed-5.m
Setting LM seed to 5
Data lm, trn: 1083512, val: 27852
Size of vocabulary: 25000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxeos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', 'xxlink', 'xxuser', 'xxemoji', 'yyemoji', '<unk>', '▁', ',', '.', '▁"', '▁to', '▁nie']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0} dps:  {'input_p': 0.25, 'output_p': 0.1, 'weight_p': 0.2, 'embed_p': 0.02, 'hidden_p': 0.15}
Bptt 70
Training lm from random weights
epoch     train_loss  valid_loss  accuracy
1         4.550366    4.531669    0.292171
2         4.135955    4.137668    0.325189
3         4.046186    4.042921    0.333102
4         3.953019    3.955529    0.341944
5         3.866866    3.883109    0.350230
6         3.757591    3.812064    0.358260
7         3.652359    3.748058    0.365905
8         3.544907    3.703726    0.371906
9         3.416362    3.694410    0.373976
10        3.317027    3.707936    0.372984
Total time: 3:50:58
data/reddit/pl-100/models/sp25k
```
```
for seed in 6 7 8; do
    python -m ulmfit lm --dataset-path data/reddit/pl-100 --bidir=False --qrnn=False --nl 4 --tokenizer='sp' --max-vocab 25000 --lang pl --name 'nl4' --lmseed=$seed - train 10 --drop-mult=0 --bs=100
done
Training lm
Max vocab: 25000
Cache dir: data/reddit/pl-100/models/sp25k
Model dir: data/reddit/pl-100/models/sp25k/lstm_nl4_lmseed-6.m
Setting LM seed to 6
Running tokenization lm...
Data lm, trn: 1083512, val: 27852
Size of vocabulary: 25000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxeos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', 'xxlink', 'xxuser', 'xxemoji', 'yyemoji', '<unk>', '▁', ',', '.', '▁"', '▁to', '▁nie']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0} dps:  {'input_p': 0.25, 'output_p': 0.1, 'weight_p': 0.2, 'embed_p': 0.02, 'hidden_p': 0.15}
Bptt 70
Training lm from random weights
epoch     train_loss  valid_loss  accuracy
1         4.534413    4.521482    0.293163
2         4.143852    4.135971    0.325955
3         4.032825    4.043230    0.333343
4         3.950969    3.955216    0.341929
5         3.852641    3.879689    0.350599
6         3.755054    3.808284    0.358533
7         3.663442    3.743742    0.366095
8         3.526133    3.699655    0.372137
9         3.407880    3.689233    0.374294
10        3.306818    3.702875    0.373423
Total time: 4:00:23
data/reddit/pl-100/models/sp25k
Saving info data/reddit/pl-100/models/sp25k/lstm_nl4_lmseed-6.m/info.json
Training lm
Max vocab: 25000
Cache dir: data/reddit/pl-100/models/sp25k
Model dir: data/reddit/pl-100/models/sp25k/lstm_nl4_lmseed-7.m
Setting LM seed to 7
Data lm, trn: 1083512, val: 27852
Size of vocabulary: 25000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxeos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', 'xxlink', 'xxuser', 'xxemoji', 'yyemoji', '<unk>', '▁', ',', '.', '▁"', '▁to', '▁nie']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0} dps:  {'input_p': 0.25, 'output_p': 0.1, 'weight_p': 0.2, 'embed_p': 0.02, 'hidden_p': 0.15}
Bptt 70
Training lm from random weights
epoch     train_loss  valid_loss  accuracy
1         4.588949    4.553423    0.291087
2         4.142656    4.144599    0.324350
3         4.048930    4.048537    0.332631
4         3.934464    3.961079    0.341731
5         3.869824    3.889143    0.349106
6         3.768741    3.819727    0.357520
7         3.678499    3.756810    0.364769
8         3.553171    3.711584    0.370588
9         3.426710    3.699724    0.373105
10        3.330709    3.711221    0.372492
Total time: 4:00:43
data/reddit/pl-100/models/sp25k
Saving info data/reddit/pl-100/models/sp25k/lstm_nl4_lmseed-7.m/info.json
Training lm
Max vocab: 25000
Cache dir: data/reddit/pl-100/models/sp25k
Model dir: data/reddit/pl-100/models/sp25k/lstm_nl4_lmseed-8.m
Setting LM seed to 8
Data lm, trn: 1083512, val: 27852
Size of vocabulary: 25000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxeos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', 'xxlink', 'xxuser', 'xxemoji', 'yyemoji', '<unk>', '▁', ',', '.', '▁"', '▁to', '▁nie']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0} dps:  {'input_p': 0.25, 'output_p': 0.1, 'weight_p': 0.2, 'embed_p': 0.02, 'hidden_p': 0.15}
Bptt 70
Training lm from random weights
epoch     train_loss  valid_loss  accuracy
1         4.567297    4.540712    0.292412
2         4.156515    4.136066    0.325115
3         4.042436    4.043569    0.333562
4         3.954511    3.959762    0.341943
5         3.857088    3.886994    0.349635
6         3.778073    3.816180    0.358092
7         3.672334    3.754274    0.365024
8         3.529533    3.708678    0.371669
9         3.423441    3.698037    0.373702
10        3.336955    3.709018    0.373042
Total time: 4:01:18
data/reddit/pl-100/models/sp25k
```
------------------

for seed in 3 4 5; do 
    python -m ulmfit poleval19_full data/reddit/pl-100/models/sp25k/lstm_nl4_lmseed-${seed}.m --early_stopping=False --skip_train_seed=True
done


for seed in 6 7 8; do
    python -m ulmfit poleval19_full data/reddit/pl-100/models/sp25k/lstm_nl4_lmseed-${seed}.m --early_stopping=False --skip_train_seed=True
done

------------------------------------------------------
## Results
```
python -m ulmfit ensemble  --glob "data/hate/pl-10-reddit/models/sp25k/lstm_ft6_cl6_lmseed-*" --key-template='${dataset_name}-${lmseed}'
{'Key': 'pl-10-reddit-6', 'Test Accuracy': 0.893, 'Test F1': tensor(0.5202), 'on': PosixPath('data/hate/pl-10-reddit/pl.test.csv'), 'files_count': 19}
{'File saved to': PosixPath('pl-10-reddit-6.ensemble.csv')}
{'Key': 'pl-10-reddit-5', 'Test Accuracy': 0.9, 'Test F1': tensor(0.5614), 'on': PosixPath('data/hate/pl-10-reddit/pl.test.csv'), 'files_count': 19}
{'File saved to': PosixPath('pl-10-reddit-5.ensemble.csv')}
{'Key': 'pl-10-reddit-3', 'Test Accuracy': 0.902, 'Test F1': tensor(0.5586), 'on': PosixPath('data/hate/pl-10-reddit/pl.test.csv'), 'files_count': 19}
{'File saved to': PosixPath('pl-10-reddit-3.ensemble.csv')}
{'Key': 'pl-10-reddit-7', 'Test Accuracy': 0.91, 'Test F1': tensor(0.6218), 'on': PosixPath('data/hate/pl-10-reddit/pl.test.csv'), 'files_count': 19}
{'File saved to': PosixPath('pl-10-reddit-7.ensemble.csv')}
{'Key': 'pl-10-reddit-4', 'Test Accuracy': 0.902, 'Test F1': tensor(0.5625), 'on': PosixPath('data/hate/pl-10-reddit/pl.test.csv'), 'files_count': 19}
{'File saved to': PosixPath('pl-10-reddit-4.ensemble.csv')}
{'Key': 'pl-10-reddit-8', 'Test Accuracy': 0.895, 'Test F1': tensor(0.5333), 'on': PosixPath('data/hate/pl-10-reddit/pl.test.csv'), 'files_count': 19}
{'File saved to': PosixPath('pl-10-reddit-8.ensemble.csv')}
```
-------------------------------------------------------

```
for seed in 3 4 5; do 
    python -m ulmfit lm --dataset-path data/wiki/pl-100 --bidir=False --qrnn=False --nl 4 --tokenizer='sp' --max-vocab 25000 --lang pl --name 'nl4' --lmseed=$seed - train 1 --drop-mult=0 --bs=100
done

for seed in 6 7 8; do 
    python -m ulmfit lm --dataset-path data/wiki/pl-100 --bidir=False --qrnn=False --nl 4 --tokenizer='sp' --max-vocab 25000 --lang pl --name 'nl4' --lmseed=$seed - train 1 --drop-mult=0 --bs=100
done




```
## Results
```
for seed in 3 4 5; do                                                                  ✘ 130
    python -m ulmfit lm --dataset-path data/wiki/pl-100 --bidir=False --qrnn=False --nl 4 --tokenizer='sp' --max-vocab 25000 --lang pl --name 'nl4' --lmseed=$seed - train 1 --drop-mult=0 --bs=100
done

Training lm
Max vocab: 25000
Cache dir: data/wiki/pl-100/models/sp25k
Model dir: data/wiki/pl-100/models/sp25k/lstm_nl4_lmseed-3.m
Setting LM seed to 3
Data lm, trn: 235357, val: 264
Size of vocabulary: 25000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', 'xxlink', 'xxuser', 'xxnumber', 'xxemoji', 'yyemoji', '<unk>', '▁', '▁.', '▁,', '▁w', 'a', 'e']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0} dps:  {'input_p': 0.25, 'output_p': 0.1, 'weight_p': 0.2, 'embed_p': 0.02, 'hidden_p': 0.15}
Bptt 70
Training lm from random weights
epoch     train_loss  valid_loss  accuracy
1         2.827361    3.228038    0.439213
Total time: 1:58:12
data/wiki/pl-100/models/sp25k
Saving info data/wiki/pl-100/models/sp25k/lstm_nl4_lmseed-3.m/info.json
Training lm
Max vocab: 25000
Cache dir: data/wiki/pl-100/models/sp25k
Model dir: data/wiki/pl-100/models/sp25k/lstm_nl4_lmseed-4.m
Setting LM seed to 4
Data lm, trn: 235357, val: 264
Size of vocabulary: 25000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', 'xxlink', 'xxuser', 'xxnumber', 'xxemoji', 'yyemoji', '<unk>', '▁', '▁.', '▁,', '▁w', 'a', 'e']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0} dps:  {'input_p': 0.25, 'output_p': 0.1, 'weight_p': 0.2, 'embed_p': 0.02, 'hidden_p': 0.15}
Bptt 70
Training lm from random weights
epoch     train_loss  valid_loss  accuracy
1         2.908606    3.238701    0.437213
Total time: 1:58:08
data/wiki/pl-100/models/sp25k
Saving info data/wiki/pl-100/models/sp25k/lstm_nl4_lmseed-4.m/info.json
Training lm
Max vocab: 25000
Cache dir: data/wiki/pl-100/models/sp25k
Model dir: data/wiki/pl-100/models/sp25k/lstm_nl4_lmseed-5.m
Setting LM seed to 5
Data lm, trn: 235357, val: 264
Size of vocabulary: 25000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', 'xxlink', 'xxuser', 'xxnumber', 'xxemoji', 'yyemoji', '<unk>', '▁', '▁.', '▁,', '▁w', 'a', 'e']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0} dps:  {'input_p': 0.25, 'output_p': 0.1, 'weight_p': 0.2, 'embed_p': 0.02, 'hidden_p': 0.15}
Bptt 70
Training lm from random weights
epoch     train_loss  valid_loss  accuracy
1         2.866939    3.221571    0.440584
Total time: 1:58:06
data/wiki/pl-100/models/sp25k



for seed in 6 7 8; do                                                                  ✘ 130
    python -m ulmfit lm --dataset-path data/wiki/pl-100 --bidir=False --qrnn=False --nl 4 --tokenizer='sp' --max-vocab 25000 --lang pl --name 'nl4' --lmseed=$seed - train 1 --drop-mult=0 --bs=100
done
Training lm
Max vocab: 25000
Cache dir: data/wiki/pl-100/models/sp25k
Model dir: data/wiki/pl-100/models/sp25k/lstm_nl4_lmseed-6.m
Setting LM seed to 6
Data lm, trn: 235357, val: 264
Size of vocabulary: 25000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', 'xxlink', 'xxuser', 'xxnumber', 'xxemoji', 'yyemoji', '<unk>', '▁', '▁.', '▁,', '▁w', 'a', 'e']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0} dps:  {'input_p': 0.25, 'output_p': 0.1, 'weight_p': 0.2, 'embed_p': 0.02, 'hidden_p': 0.15}
Bptt 70
Training lm from random weights
epoch     train_loss  valid_loss  accuracy
{'Key': 'pl-10-reddit-7', 'Test Accuracy': 0.91, 'Test F1': tensor(0.6218), 'on': PosixPath('data/hate/pl-10-reddit/pl.test.csv'), 'files_count': 19}
1         2.875308    3.222101    0.440292
Total time: 2:03:27
data/wiki/pl-100/models/sp25k
Saving info data/wiki/pl-100/models/sp25k/lstm_nl4_lmseed-6.m/info.json
Training lm
Max vocab: 25000
Cache dir: data/wiki/pl-100/models/sp25k
Model dir: data/wiki/pl-100/models/sp25k/lstm_nl4_lmseed-7.m
Setting LM seed to 7
Data lm, trn: 235357, val: 264
Size of vocabulary: 25000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', 'xxlink', 'xxuser', 'xxnumber', 'xxemoji', 'yyemoji', '<unk>', '▁', '▁.', '▁,', '▁w', 'a', 'e']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0} dps:  {'input_p': 0.25, 'output_p': 0.1, 'weight_p': 0.2, 'embed_p': 0.02, 'hidden_p': 0.15}
Bptt 70
Training lm from random weights
epoch     train_loss  valid_loss  accuracy
1         2.880711    3.230228    0.439213
Total time: 2:03:20
data/wiki/pl-100/models/sp25k
Saving info data/wiki/pl-100/models/sp25k/lstm_nl4_lmseed-7.m/info.json
Training lm
Max vocab: 25000
Cache dir: data/wiki/pl-100/models/sp25k
Model dir: data/wiki/pl-100/models/sp25k/lstm_nl4_lmseed-8.m
Setting LM seed to 8
Data lm, trn: 235357, val: 264
Size of vocabulary: 25000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', 'xxlink', 'xxuser', 'xxnumber', 'xxemoji', 'yyemoji', '<unk>', '▁', '▁.', '▁,', '▁w', 'a', 'e']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0} dps:  {'input_p': 0.25, 'output_p': 0.1, 'weight_p': 0.2, 'embed_p': 0.02, 'hidden_p': 0.15}
Bptt 70
Training lm from random weights
epoch     train_loss  valid_loss  accuracy
1         2.815793    3.215459    0.440529
Total time: 2:03:37
data/wiki/pl-100/models/sp25k
Saving info data/wiki/pl-100/models/sp25k/lstm_nl4_lmseed-8.m/info.json
```






## Wikipedia 1e tests with early stopping
```
for seed in 3 4 5; do 
    python -m ulmfit poleval19_full data/wiki/pl-100/models/sp25k/lstm_nl4_lmseed-${seed}.m --early_stopping=False --skip_train_seed=True --name "1ep"
done


for seed in 6 7 8; do
    python -m ulmfit poleval19_full data/wiki/pl-100/models/sp25k/lstm_nl4_lmseed-${seed}.m --early_stopping=False --skip_train_seed=True --name "1ep"
done

```

### Ensemble  dropout 0.3 wikipedia
```
{'Key': '6', 'Test Accuracy': 0.884, 'Test F1': tensor(0.4867), 'on': PosixPath('data/hate/pl-10-wiki/pl.test.csv'), 'files_count': 19}
{'File saved to': PosixPath('6.ensemble.csv')}
{'Key': '5', 'Test Accuracy': 0.885, 'Test F1': tensor(0.4700), 'on': PosixPath('data/hate/pl-10-wiki/pl.test.csv'), 'files_count': 19}
{'File saved to': PosixPath('5.ensemble.csv')}
{'Key': '3', 'Test Accuracy': 0.893, 'Test F1': tensor(0.5158), 'on': PosixPath('data/hate/pl-10-wiki/pl.test.csv'), 'files_count': 19}
{'File saved to': PosixPath('3.ensemble.csv')}
{'Key': '7', 'Test Accuracy': 0.888, 'Test F1': tensor(0.5172), 'on': PosixPath('data/hate/pl-10-wiki/pl.test.csv'), 'files_count': 19}
{'File saved to': PosixPath('7.ensemble.csv')}
{'Key': '8', 'Test Accuracy': 0.9, 'Test F1': tensor(0.5575), 'on': PosixPath('data/hate/pl-10-wiki/pl.test.csv'), 'files_count': 19}
{'File saved to': PosixPath('8.ensemble.csv')}
{'Key': '4', 'Test Accuracy': 0.891, 'Test F1': tensor(0.5240), 'on': PosixPath('data/hate/pl-10-wiki/pl.test.csv'), 'files_count': 19}
{'File saved to': PosixPath('4.ensemble.csv')}
```

# wikipedia with early stopping
for seed in 3 4 5 6 7 8; do 
    python -m ulmfit poleval19_full data/wiki/pl-100/models/sp25k/lstm_nl4_lmseed-${seed}.m --early_stopping=True --skip_train_seed=True --name "1ep"
done


for seed in 3 4 5 6 7 8; do 
    python -m ulmfit poleval19_full data/reddit/pl-100/models/sp25k/lstm_nl4_lmseed-${seed}.m --early_stopping=True --skip_train_seed=True --name "1ep"
done

### DROPOUT
```
{'Key': '5', 'Test Accuracy': 0.899, 'Test F1': tensor(0.5511), 'on': PosixPath('data/hate/pl-10-reddit/pl.test.csv'), 'files_count': 40}
{'File saved to': PosixPath('5.ensemble.csv')}
{'Key': '4', 'Test Accuracy': 0.898, 'Test F1': tensor(0.5446), 'on': PosixPath('data/hate/pl-10-reddit/pl.test.csv'), 'files_count': 40}
{'File saved to': PosixPath('4.ensemble.csv')}
{'Key': '6', 'Test Accuracy': 0.885, 'Test F1': tensor(0.5106), 'on': PosixPath('data/hate/pl-10-reddit/pl.test.csv'), 'files_count': 40}
{'File saved to': PosixPath('6.ensemble.csv')}
{'Key': '7', 'Test Accuracy': 0.906, 'Test F1': tensor(0.5948), 'on': PosixPath('data/hate/pl-10-reddit/pl.test.csv'), 'files_count': 40}
{'File saved to': PosixPath('7.ensemble.csv')}
```