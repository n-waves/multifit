## Laser Perforamnce

Accuracy matrix:
### 1k

| Train |   en  |   de  |   es  |   fr  |   it  |   ru  |   zh |
|-------|-------|-------|-------|-------|-------|-------|-------|
| en:   | 91.48 | 87.65 | 75.48 | 84.00 | 71.18 | 66.58 | 76.65 |
| de:   | 78.23 | 93.50 | 81.40 | 81.50 | 74.53 | 64.58 | 73.20 |
| es:   | 71.62 | 84.00 | 93.73 | 78.90 | 73.38 | 53.33 | 55.83 |
| fr:   | 81.30 | 88.75 | 80.12 | 90.85 | 72.58 | 67.35 | 79.40 |
| it:   | 74.33 | 83.53 | 80.58 | 79.78 | 84.48 | 66.45 | 63.35 |
| ru:   | 72.38 | 81.65 | 65.73 | 71.30 | 63.33 | 85.45 | 59.58 |
| zh:   | 74.98 | 81.35 | 72.20 | 73.28 | 70.08 | 66.23 | 88.30 |

### 10k
 
| Train |   en  |   de  |   es  |   fr  |   it  |   ru  |   zh |
|-------|-------|-------|-------|-------|-------|-------|-------|
|  en:  | 92.70 | 87.43 | 77.38 | 78.70 | 72.53 | 67.70 | 75.18 |
|  de:  | 81.60 | 95.40 | 83.50 | 82.85 | 76.60 | 68.80 | 73.12 |
|  es:  | 73.48 | 87.13 | 94.40 | 81.63 | 76.70 | 58.65 | 72.98 |
|  fr:  | 85.08 | 91.65 | 81.05 | 93.65 | 75.08 | 70.73 | 76.33 |
|  it:  | 76.75 | 86.68 | 82.55 | 82.65 | 87.80 | 65.90 | 73.35 |
|  ru:  | 75.23 | 81.88 | 66.83 | 68.60 | 67.38 | 87.00 | 62.68 |
|  zh:  | 76.05 | 82.05 | 68.40 | 77.38 | 68.45 | 66.88 | 90.38 |



### laser 1k 
#### Building dataset
```bash
for SRC_LANG in en de fr; do                                                                                                                                         ✘ 130
    for LANG in en de es fr it ru zh; do
        echo $LANG from $SRC_LANG
        python ../../source/classify.py embed10000/mldoc.${SRC_LANG}-${SRC_LANG}.h5 ~/workspace/ulmfit-multilingual/data/mldoc/${LANG}-10 --suffix=10 | grep Test:
    done
done
```

```
for SRC_LANG in en de fr; do                                                                                                                                         ✘ 130
    for LANG in en de es fr it ru zh; do
        echo $LANG from $SRC_LANG
        python ../../source/classify.py embed1000/mldoc.${SRC_LANG}-${SRC_LANG}.h5 ~/workspace/ulmfit-multilingual/data/mldoc/${LANG}-1 | grep Test:
    done
done

en from en
 | Test: 91.48% | classes: 23.77 24.90 26.25 25.07
de from en
 | Test: 87.65% | classes: 21.98 24.45 27.65 25.93
es from en
 | Test: 75.48% | classes: 21.60 15.82 22.10 40.48
fr from en
 | Test: 84.00% | classes: 23.18 29.12 27.90 19.80
it from en
 | Test: 71.18% | classes: 23.65 22.88 25.68 27.80
ru from en
 | Test: 66.58% | classes: 29.48 13.78 34.52 22.23
zh from en
 | Test: 76.65% | classes: 30.25 31.30 13.93 24.52
en from de
 | Test: 78.23% | classes: 31.80 17.73 30.15 20.32
de from de
 | Test: 93.50% | classes: 24.45 25.45 26.00 24.10
es from de
 | Test: 81.40% | classes: 24.15 25.77 20.12 29.95
fr from de
 | Test: 81.50% | classes: 25.52 29.45 27.45 17.57
it from de
 | Test: 74.53% | classes: 24.70 27.25 22.43 25.62
ru from de
 | Test: 64.58% | classes: 45.62  9.12 26.73 18.52
zh from de
 | Test: 73.20% | classes: 31.20 43.38  7.60 17.82
en from fr
 | Test: 81.30% | classes: 28.95 18.02 24.98 28.05
de from fr
 | Test: 88.75% | classes: 24.00 23.75 24.85 27.40
es from fr
 | Test: 80.12% | classes: 24.50 14.82 18.40 42.27
fr from fr
 | Test: 90.85% | classes: 24.50 24.75 24.68 26.07
it from fr
 | Test: 72.58% | classes: 25.45 24.10 17.50 32.95
ru from fr
 | Test: 67.35% | classes: 47.15 13.62 16.68 22.55
zh from fr
 | Test: 79.40% | classes: 33.60 31.12  9.07 26.20
```

#### Evaluation of Laser 1k Performance 
```
python -m ulmfit eval --glob="mldoc/*-1/models/sp60k/lstm_nl4.m" --dataset_template="{}-laser-*" --name nl4 --cuda-id=0                                   ✘ 130
Max vocab: 60000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1-laser-fr/models/sp60k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1-laser-fr/models/sp60k/lstm_nl4.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1-laser-fr/zh.dev.csv
Running tokenization...
Saving tokenized: cls.trn 13500, cls.val 1500
Running tokenization...
Saving tokenized: cls.trn 1000, cls.val 1000
Size of vocabulary: 60000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁的', '▁。', '▁年', '▁、', '▁在', '▁一', '▁中', '▁是', '▁人']
Training args:  {'tie_weights': True, 'clip': 0.12, 'bptt': 70, 'pretrained_fnames': [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp60k/lstm_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp60k/lstm_nl4.m/../itos')], 'pretrained_model': None, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  [0.25 0.1  0.2  0.02 0.15]
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1-laser-fr/models/sp60k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1-laser-fr/models/sp60k/lstm_nl4.m/info.json
Starting classifier training
epoch     train_loss  valid_loss  accuracy
1         0.789124    0.620514    0.781000
epoch     train_loss  valid_loss  accuracy
1         0.621348    0.524669    0.828000
epoch     train_loss  valid_loss  accuracy
1         0.497774    0.467979    0.842000
epoch     train_loss  valid_loss  accuracy
1         0.445851    0.479755    0.833000
2         0.424097    0.468968    0.826000
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1-laser-fr/models/sp60k/lstm_nl4.m
Loss and accuracy using (cls_best): [0.53502685, tensor(0.8235)]
[('data/mldoc/zh-1-laser-fr/models/sp60k/lstm_nl4.m', 0.8234999775886536)]
python -m ulmfit eval --glob="mldoc/*-1/models/sp60k/lstm_nl4.m" --dataset_template="{}-laser-*" --name nl4 --cuda-id=0                                   ✘ 130
Max vocab: 60000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1-laser-fr/models/sp60k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1-laser-fr/models/sp60k/lstm_nl4.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1-laser-fr/zh.dev.csv
Running tokenization...
Saving tokenized: cls.trn 13500, cls.val 1500
Running tokenization...
Saving tokenized: cls.trn 1000, cls.val 1000
Size of vocabulary: 60000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁的', '▁。', '▁年', '▁、', '▁在', '▁一', '▁中', '▁是', '▁人']
Training args:  {'tie_weights': True, 'clip': 0.12, 'bptt': 70, 'pretrained_fnames': [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp60k/lstm_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp60k/lstm_nl4.m/../itos')], 'pretrained_model': None, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  [0.25 0.1  0.2  0.02 0.15]
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1-laser-fr/models/sp60k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1-laser-fr/models/sp60k/lstm_nl4.m/info.json
Starting classifier training
epoch     train_loss  valid_loss  accuracy
1         0.789124    0.620514    0.781000
epoch     train_loss  valid_loss  accuracy
1         0.621348    0.524669    0.828000
epoch     train_loss  valid_loss  accuracy
1         0.497774    0.467979    0.842000
epoch     train_loss  valid_loss  accuracy
1         0.445851    0.479755    0.833000
2         0.424097    0.468968    0.826000
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1-laser-fr/models/sp60k/lstm_nl4.m
Loss and accuracy using (cls_best): [0.53502685, tensor(0.8235)]
[('data/mldoc/zh-1-laser-fr/models/sp60k/lstm_nl4.m', 0.8234999775886536)]
(fastaiv1) pczapla@galatea ~/w/ulmfit-multilingual ❯❯❯ python -m ulmfit eval --glob="mldoc/*-1/models/sp30k/lstm_nl4.m" --dataset_template="{}-laser-*" --name nl4 --cuda-id=0
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1-laser-de/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1-laser-de/models/sp30k/lstm_nl4.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1-laser-de/it.dev.csv
Running tokenization...
Saving tokenized: cls.trn 13500, cls.val 1500
Running tokenization...
Saving tokenized: cls.trn 1000, cls.val 1000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁.', '▁di', "▁&'", "'", '▁e', '▁il', '▁la', 'e', '▁in']
Training args:  {'tie_weights': True, 'clip': 0.12, 'bptt': 70, 'pretrained_fnames': [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1/models/sp30k/lstm_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1/models/sp30k/lstm_nl4.m/../itos')], 'pretrained_model': None, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  [0.25 0.1  0.2  0.02 0.15]
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1-laser-de/models/sp30k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1-laser-de/models/sp30k/lstm_nl4.m/info.json
Starting classifier training
epoch     train_loss  valid_loss  accuracy
1         0.823176    0.588192    0.802000
epoch     train_loss  valid_loss  accuracy
1         0.654395    0.465622    0.846000
epoch     train_loss  valid_loss  accuracy
1         0.536948    0.453061    0.847000
epoch     train_loss  valid_loss  accuracy
1         0.488410    0.454361    0.845000
2         0.450684    0.448873    0.849000
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1-laser-de/models/sp30k/lstm_nl4.m
Loss and accuracy using (cls_best): [0.6332891, tensor(0.7875)]
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1-laser-fr/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1-laser-fr/models/sp30k/lstm_nl4.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1-laser-fr/de.dev.csv
Running tokenization...
Saving tokenized: cls.trn 13500, cls.val 1500
Running tokenization...
Saving tokenized: cls.trn 1000, cls.val 1000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁.', '▁,', '▁der', '▁die', '▁und', '▁in', 'en', "▁&'", 's', '-']
Training args:  {'tie_weights': True, 'clip': 0.12, 'bptt': 70, 'pretrained_fnames': [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1/models/sp30k/lstm_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1/models/sp30k/lstm_nl4.m/../itos')], 'pretrained_model': None, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  [0.25 0.1  0.2  0.02 0.15]
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1-laser-fr/models/sp30k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1-laser-fr/models/sp30k/lstm_nl4.m/info.json
Starting classifier training
epoch     train_loss  valid_loss  accuracy
1         0.566941    0.389549    0.882000
epoch     train_loss  valid_loss  accuracy
1         0.399470    0.302616    0.898000
epoch     train_loss  valid_loss  accuracy
1         0.349054    0.336955    0.900000
epoch     train_loss  valid_loss  accuracy
1         0.278230    0.333488    0.896000
2         0.275510    0.343370    0.899000
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1-laser-fr/models/sp30k/lstm_nl4.m
Loss and accuracy using (cls_best): [0.26227093, tensor(0.9222)]
Traceback (most recent call last):
  File "/home/pczapla/anaconda3/envs/fastaiv1/lib/python3.7/runpy.py", line 193, in _run_module_as_main
    "__main__", mod_spec)
  File "/home/pczapla/anaconda3/envs/fastaiv1/lib/python3.7/runpy.py", line 85, in _run_code
    exec(code, run_globals)
  File "/home/pczapla/workspace/ulmfit-multilingual/ulmfit/__main__.py", line 58, in <module>
    fire.Fire(ULMFiT())
  File "/home/pczapla/anaconda3/envs/fastaiv1/lib/python3.7/site-packages/fire/core.py", line 127, in Fire
    component_trace = _Fire(component, args, context, name)
  File "/home/pczapla/anaconda3/envs/fastaiv1/lib/python3.7/site-packages/fire/core.py", line 366, in _Fire
    component, remaining_args)
  File "/home/pczapla/anaconda3/envs/fastaiv1/lib/python3.7/site-packages/fire/core.py", line 542, in _CallCallable
    result = fn(*varargs, **kwargs)
  File "/home/pczapla/workspace/ulmfit-multilingual/ulmfit/__main__.py", line 41, in eval
    dataset_path = get_dataset_path(base_model, dataset_template)
  File "/home/pczapla/workspace/ulmfit-multilingual/ulmfit/__main__.py", line 17, in get_dataset_path
    return list(ds.parent.glob(dataset_template.format(ds.name)))[0]
IndexError: list index out of range
(fastaiv1) pczapla@galatea ~/w/ulmfit-multilingual ❯❯❯ python -m ulmfit eval --glob="mldoc/*-1/models/sp30k/lstm_nl4.m" --dataset_template="{}-laser-*" --name nl4 --cuda-id=0                                     ✘ 1
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1-laser-de/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1-laser-de/models/sp30k/lstm_nl4.m
Evaluating previously trained model
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1-laser-de/it.dev.csv
Tokenized data loaded, lm.trn 13500, lm.val 1500
Tokenized data loaded, cls.trn 1000, cls.val 1000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁.', '▁di', "▁&'", "'", '▁e', '▁il', '▁la', 'e', '▁in']
Loss and accuracy using (cls_last): [0.6332891, tensor(0.7875)]
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1-laser-fr/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1-laser-fr/models/sp30k/lstm_nl4.m
Evaluating previously trained model
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1-laser-fr/de.dev.csv
Tokenized data loaded, lm.trn 13500, lm.val 1500
Tokenized data loaded, cls.trn 1000, cls.val 1000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁.', '▁,', '▁der', '▁die', '▁und', '▁in', 'en', "▁&'", 's', '-']
Loss and accuracy using (cls_last): [0.26227093, tensor(0.9222)]
Skipping data/mldoc/ja-1/models/sp30k/lstm_nl4.m as template {}-laser-* was not found
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1-laser-fr/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1-laser-fr/models/sp30k/lstm_nl4.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1-laser-fr/zh.dev.csv
Running tokenization...
Saving tokenized: cls.trn 13500, cls.val 1500
Running tokenization...
Saving tokenized: cls.trn 1000, cls.val 1000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁的', '▁。', '▁年', '▁、', '▁在', '▁一', '▁中', '▁人', '▁是']
Training args:  {'tie_weights': True, 'clip': 0.12, 'bptt': 70, 'pretrained_fnames': [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp30k/lstm_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp30k/lstm_nl4.m/../itos')], 'pretrained_model': None, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  [0.25 0.1  0.2  0.02 0.15]
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1-laser-fr/models/sp30k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1-laser-fr/models/sp30k/lstm_nl4.m/info.json
Starting classifier training
epoch     train_loss  valid_loss  accuracy
1         0.745636    0.601781    0.812000
epoch     train_loss  valid_loss  accuracy
1         0.564749    0.435314    0.851000
epoch     train_loss  valid_loss  accuracy
1         0.485875    0.428803    0.850000
epoch     train_loss  valid_loss  accuracy
1         0.405431    0.439304    0.847000
2         0.418333    0.442639    0.845000
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1-laser-fr/models/sp30k/lstm_nl4.m
Loss and accuracy using (cls_best): [0.5289812, tensor(0.8465)]
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1-laser-en/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1-laser-en/models/sp30k/lstm_nl4.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1-laser-en/fr.dev.csv
Running tokenization...
Saving tokenized: cls.trn 13500, cls.val 1500
Running tokenization...
Saving tokenized: cls.trn 1000, cls.val 1000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁.', "'", 's', '▁la', '▁le', '▁et', '▁l', '▁à']
Training args:  {'tie_weights': True, 'clip': 0.12, 'bptt': 70, 'pretrained_fnames': [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/models/sp30k/lstm_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/models/sp30k/lstm_nl4.m/../itos')], 'pretrained_model': None, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  [0.25 0.1  0.2  0.02 0.15]
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1-laser-en/models/sp30k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1-laser-en/models/sp30k/lstm_nl4.m/info.json
Starting classifier training
epoch     train_loss  valid_loss  accuracy
1         0.669493    0.510190    0.852000
epoch     train_loss  valid_loss  accuracy
1         0.464863    0.349456    0.888000
epoch     train_loss  valid_loss  accuracy
1         0.396977    0.335358    0.879000
epoch     train_loss  valid_loss  accuracy
1         0.316100    0.326822    0.882000
2         0.292052    0.326660    0.874000
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1-laser-en/models/sp30k/lstm_nl4.m
Loss and accuracy using (cls_best): [0.3416499, tensor(0.8878)]
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1-laser-fr/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1-laser-fr/models/sp30k/lstm_nl4.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1-laser-fr/ru.dev.csv
Running tokenization...
Saving tokenized: cls.trn 9195, cls.val 1021
Running tokenization...
Saving tokenized: cls.trn 1000, cls.val 1000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁.', '▁в', 'а', '▁и', 'е', 'и', 'й', '▁на', 'х']
Training args:  {'tie_weights': True, 'clip': 0.12, 'bptt': 70, 'pretrained_fnames': [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp30k/lstm_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp30k/lstm_nl4.m/../itos')], 'pretrained_model': None, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  [0.25 0.1  0.2  0.02 0.15]
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1-laser-fr/models/sp30k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1-laser-fr/models/sp30k/lstm_nl4.m/info.json
Starting classifier training
epoch     train_loss  valid_loss  accuracy
1         0.906124    0.592495    0.797000
epoch     train_loss  valid_loss  accuracy
1         0.751562    0.440800    0.842000
epoch     train_loss  valid_loss  accuracy
1         0.631221    0.393381    0.860000
epoch     train_loss  valid_loss  accuracy
1         0.582251    0.376320    0.867000
2         0.543821    0.374095    0.860000
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1-laser-fr/models/sp30k/lstm_nl4.m
Loss and accuracy using (cls_best): [1.0429544, tensor(0.6833)]
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1-laser-de/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1-laser-de/models/sp30k/lstm_nl4.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1-laser-de/es.dev.csv
Running tokenization...
Saving tokenized: cls.trn 13013, cls.val 1445
Running tokenization...
Saving tokenized: cls.trn 1000, cls.val 1000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁la', '▁.', '▁en', '▁el', '▁y', 's', '▁a', '▁que']
Training args:  {'tie_weights': True, 'clip': 0.12, 'bptt': 70, 'pretrained_fnames': [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp30k/lstm_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp30k/lstm_nl4.m/../itos')], 'pretrained_model': None, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  [0.25 0.1  0.2  0.02 0.15]
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1-laser-de/models/sp30k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1-laser-de/models/sp30k/lstm_nl4.m/info.json
Starting classifier training
epoch     train_loss  valid_loss  accuracy
1         0.667080    0.471187    0.884000
epoch     train_loss  valid_loss  accuracy
1         0.553853    0.329840    0.904000
epoch     train_loss  valid_loss  accuracy
1         0.463647    0.309136    0.907000
epoch     train_loss  valid_loss  accuracy
1         0.396284    0.282263    0.911000
2         0.368159    0.287222    0.916000
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1-laser-de/models/sp30k/lstm_nl4.m
Loss and accuracy using (cls_best): [0.5038375, tensor(0.8550)]
[('data/mldoc/de-1-laser-fr/models/sp30k/lstm_nl4.m', 0.922249972820282), ('data/mldoc/es-1-laser-de/models/sp30k/lstm_nl4.m', 0.8550000190734863), ('data/mldoc/fr-1-laser-en/models/sp30k/lstm_nl4.m', 0.8877500295639038), ('data/mldoc/it-1-laser-de/models/sp30k/lstm_nl4.m', 0.7875000238418579), ('data/mldoc/ru-1-laser-fr/models/sp30k/lstm_nl4.m', 0.6832500100135803), ('data/mldoc/zh-1-laser-fr/models/sp30k/lstm_nl4.m', 0.8464999794960022)]
```
second run
```
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1-laser-de/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1-laser-de/models/sp30k/lstm_nl4.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1-laser-de/de.dev.csv
Running tokenization...
Saving tokenized: cls.trn 13500, cls.val 1500
Running tokenization...
Saving tokenized: cls.trn 1000, cls.val 1000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁.', '▁,', '▁der', '▁die', '▁und', '▁in', 'en', "▁&'", 's', '-']
Training args:  {'tie_weights': True, 'clip': 0.12, 'bptt': 70, 'pretrained_fnames': [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1/models/sp30k/lstm_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1/models/sp30k/lstm_nl4.m/../itos')], 'pretrained_model': None, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  [0.25 0.1  0.2  0.02 0.15]
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1-laser-de/models/sp30k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1-laser-de/models/sp30k/lstm_nl4.m/info.json
Starting classifier training
epoch     train_loss  valid_loss  accuracy
1         0.467292    0.243158    0.919000
epoch     train_loss  valid_loss  accuracy
1         0.270090    0.207252    0.941000
epoch     train_loss  valid_loss  accuracy
1         0.201597    0.219442    0.934000
epoch     train_loss  valid_loss  accuracy
1         0.193163    0.199092    0.943000
2         0.169631    0.199501    0.940000
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1-laser-de/models/sp30k/lstm_nl4.m
Loss and accuracy using (cls_best): [0.16265252, tensor(0.9545)]
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1-laser-en/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1-laser-en/models/sp30k/lstm_nl4.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1-laser-en/de.dev.csv
Running tokenization...
Saving tokenized: cls.trn 13500, cls.val 1500
Running tokenization...
Saving tokenized: cls.trn 1000, cls.val 1000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁.', '▁,', '▁der', '▁die', '▁und', '▁in', 'en', "▁&'", 's', '-']
Training args:  {'tie_weights': True, 'clip': 0.12, 'bptt': 70, 'pretrained_fnames': [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1/models/sp30k/lstm_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1/models/sp30k/lstm_nl4.m/../itos')], 'pretrained_model': None, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  [0.25 0.1  0.2  0.02 0.15]
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1-laser-en/models/sp30k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1-laser-en/models/sp30k/lstm_nl4.m/info.json
Starting classifier training
epoch     train_loss  valid_loss  accuracy
1         0.575276    0.419917    0.879000
epoch     train_loss  valid_loss  accuracy
1         0.475003    0.263138    0.909000
epoch     train_loss  valid_loss  accuracy
1         0.345987    0.260215    0.911000
epoch     train_loss  valid_loss  accuracy
1         0.305776    0.268171    0.906000
2         0.289134    0.267642    0.911000
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1-laser-en/models/sp30k/lstm_nl4.m
Loss and accuracy using (cls_best): [0.23464507, tensor(0.9295)]
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1-laser-fr/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1-laser-fr/models/sp30k/lstm_nl4.m
Evaluating previously trained model
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1-laser-fr/de.dev.csv
Tokenized data loaded, lm.trn 13500, lm.val 1500
Tokenized data loaded, cls.trn 1000, cls.val 1000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁.', '▁,', '▁der', '▁die', '▁und', '▁in', 'en', "▁&'", 's', '-']
Loss and accuracy using (cls_last): [0.26227093, tensor(0.9222)]
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1-laser-de/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1-laser-de/models/sp30k/lstm_nl4.m
Evaluating previously trained model
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1-laser-de/es.dev.csv
Tokenized data loaded, lm.trn 13013, lm.val 1445
Tokenized data loaded, cls.trn 1000, cls.val 1000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁la', '▁.', '▁en', '▁el', '▁y', 's', '▁a', '▁que']
Loss and accuracy using (cls_last): [0.5038375, tensor(0.8550)]
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1-laser-en/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1-laser-en/models/sp30k/lstm_nl4.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1-laser-en/es.dev.csv
Running tokenization...
Saving tokenized: cls.trn 13013, cls.val 1445
Running tokenization...
Saving tokenized: cls.trn 1000, cls.val 1000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁la', '▁.', '▁en', '▁el', '▁y', 's', '▁a', '▁que']
Training args:  {'tie_weights': True, 'clip': 0.12, 'bptt': 70, 'pretrained_fnames': [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp30k/lstm_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp30k/lstm_nl4.m/../itos')], 'pretrained_model': None, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  [0.25 0.1  0.2  0.02 0.15]
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1-laser-en/models/sp30k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1-laser-en/models/sp30k/lstm_nl4.m/info.json
Starting classifier training
epoch     train_loss  valid_loss  accuracy
1         0.784271    0.667321    0.741000
epoch     train_loss  valid_loss  accuracy
1         0.601108    0.471457    0.854000
epoch     train_loss  valid_loss  accuracy
1         0.489287    0.428631    0.854000
epoch     train_loss  valid_loss  accuracy
1         0.434144    0.413409    0.864000
2         0.443724    0.385349    0.869000
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1-laser-en/models/sp30k/lstm_nl4.m
Loss and accuracy using (cls_best): [0.82167965, tensor(0.8050)]
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1-laser-fr/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1-laser-fr/models/sp30k/lstm_nl4.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1-laser-fr/es.dev.csv
Running tokenization...
Saving tokenized: cls.trn 13013, cls.val 1445
Running tokenization...
Saving tokenized: cls.trn 1000, cls.val 1000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁la', '▁.', '▁en', '▁el', '▁y', 's', '▁a', '▁que']
Training args:  {'tie_weights': True, 'clip': 0.12, 'bptt': 70, 'pretrained_fnames': [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp30k/lstm_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp30k/lstm_nl4.m/../itos')], 'pretrained_model': None, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  [0.25 0.1  0.2  0.02 0.15]
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1-laser-fr/models/sp30k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1-laser-fr/models/sp30k/lstm_nl4.m/info.json
Starting classifier training
epoch     train_loss  valid_loss  accuracy
1         0.752788    0.615142    0.786000
epoch     train_loss  valid_loss  accuracy
1         0.566108    0.403893    0.870000
epoch     train_loss  valid_loss  accuracy
1         0.503008    0.468810    0.865000
epoch     train_loss  valid_loss  accuracy
1         0.413641    0.448900    0.873000
2         0.381155    0.413034    0.879000
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1-laser-fr/models/sp30k/lstm_nl4.m
Loss and accuracy using (cls_best): [0.7937071, tensor(0.8100)]
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1-laser-de/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1-laser-de/models/sp30k/lstm_nl4.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1-laser-de/fr.dev.csv
Running tokenization...
Saving tokenized: cls.trn 13500, cls.val 1500
Running tokenization...
Saving tokenized: cls.trn 1000, cls.val 1000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁.', "'", 's', '▁la', '▁le', '▁et', '▁l', '▁à']
Training args:  {'tie_weights': True, 'clip': 0.12, 'bptt': 70, 'pretrained_fnames': [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/models/sp30k/lstm_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/models/sp30k/lstm_nl4.m/../itos')], 'pretrained_model': None, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  [0.25 0.1  0.2  0.02 0.15]
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1-laser-de/models/sp30k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1-laser-de/models/sp30k/lstm_nl4.m/info.json
Starting classifier training
epoch     train_loss  valid_loss  accuracy
1         0.674638    0.524605    0.796000
epoch     train_loss  valid_loss  accuracy
1         0.493693    0.401442    0.851000
epoch     train_loss  valid_loss  accuracy
1         0.418525    0.394886    0.859000
epoch     train_loss  valid_loss  accuracy
1         0.343561    0.402565    0.862000
2         0.335855    0.418237    0.851000
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1-laser-de/models/sp30k/lstm_nl4.m
Loss and accuracy using (cls_best): [0.44778627, tensor(0.8737)]
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1-laser-en/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1-laser-en/models/sp30k/lstm_nl4.m
Evaluating previously trained model
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1-laser-en/fr.dev.csv
Tokenized data loaded, lm.trn 13500, lm.val 1500
Tokenized data loaded, cls.trn 1000, cls.val 1000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁.', "'", 's', '▁la', '▁le', '▁et', '▁l', '▁à']
Loss and accuracy using (cls_last): [0.3416499, tensor(0.8878)]
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1-laser-fr/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1-laser-fr/models/sp30k/lstm_nl4.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1-laser-fr/fr.dev.csv
Running tokenization...
Saving tokenized: cls.trn 13500, cls.val 1500
Running tokenization...
Saving tokenized: cls.trn 1000, cls.val 1000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁.', "'", 's', '▁la', '▁le', '▁et', '▁l', '▁à']
Training args:  {'tie_weights': True, 'clip': 0.12, 'bptt': 70, 'pretrained_fnames': [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/models/sp30k/lstm_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/models/sp30k/lstm_nl4.m/../itos')], 'pretrained_model': None, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  [0.25 0.1  0.2  0.02 0.15]
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1-laser-fr/models/sp30k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1-laser-fr/models/sp30k/lstm_nl4.m/info.json
Starting classifier training
epoch     train_loss  valid_loss  accuracy
1         0.477812    0.332947    0.894000
epoch     train_loss  valid_loss  accuracy
1         0.305868    0.201659    0.937000
epoch     train_loss  valid_loss  accuracy
1         0.208116    0.224481    0.931000
epoch     train_loss  valid_loss  accuracy
1         0.146847    0.214640    0.941000
2         0.129603    0.227498    0.929000
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1-laser-fr/models/sp30k/lstm_nl4.m
Loss and accuracy using (cls_best): [0.19940722, tensor(0.9358)]
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1-laser-de/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1-laser-de/models/sp30k/lstm_nl4.m
Evaluating previously trained model
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1-laser-de/it.dev.csv
Tokenized data loaded, lm.trn 13500, lm.val 1500
Tokenized data loaded, cls.trn 1000, cls.val 1000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁.', '▁di', "▁&'", "'", '▁e', '▁il', '▁la', 'e', '▁in']
Loss and accuracy using (cls_last): [0.6332891, tensor(0.7875)]
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1-laser-en/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1-laser-en/models/sp30k/lstm_nl4.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1-laser-en/it.dev.csv
Running tokenization...
Saving tokenized: cls.trn 13500, cls.val 1500
Running tokenization...
Saving tokenized: cls.trn 1000, cls.val 1000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁.', '▁di', "▁&'", "'", '▁e', '▁il', '▁la', 'e', '▁in']
Training args:  {'tie_weights': True, 'clip': 0.12, 'bptt': 70, 'pretrained_fnames': [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1/models/sp30k/lstm_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1/models/sp30k/lstm_nl4.m/../itos')], 'pretrained_model': None, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  [0.25 0.1  0.2  0.02 0.15]
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1-laser-en/models/sp30k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1-laser-en/models/sp30k/lstm_nl4.m/info.json
Starting classifier training
epoch     train_loss  valid_loss  accuracy
1         0.845312    0.660645    0.769000
epoch     train_loss  valid_loss  accuracy
1         0.699314    0.584146    0.786000
epoch     train_loss  valid_loss  accuracy
1         0.556744    0.531658    0.801000
epoch     train_loss  valid_loss  accuracy
1         0.503091    0.529716    0.805000
2         0.474142    0.520058    0.806000
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1-laser-en/models/sp30k/lstm_nl4.m
Loss and accuracy using (cls_best): [0.7639212, tensor(0.7620)]
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1-laser-fr/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1-laser-fr/models/sp30k/lstm_nl4.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1-laser-fr/it.dev.csv
Running tokenization...
Saving tokenized: cls.trn 13500, cls.val 1500
Running tokenization...
Saving tokenized: cls.trn 1000, cls.val 1000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁.', '▁di', "▁&'", "'", '▁e', '▁il', '▁la', 'e', '▁in']
Training args:  {'tie_weights': True, 'clip': 0.12, 'bptt': 70, 'pretrained_fnames': [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1/models/sp30k/lstm_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1/models/sp30k/lstm_nl4.m/../itos')], 'pretrained_model': None, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  [0.25 0.1  0.2  0.02 0.15]
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1-laser-fr/models/sp30k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1-laser-fr/models/sp30k/lstm_nl4.m/info.json
Starting classifier training
epoch     train_loss  valid_loss  accuracy
1         0.773207    0.568426    0.803000
epoch     train_loss  valid_loss  accuracy
1         0.570457    0.516704    0.821000
epoch     train_loss  valid_loss  accuracy
1         0.527280    0.460192    0.840000
epoch     train_loss  valid_loss  accuracy
1         0.469201    0.461563    0.841000
2         0.458892    0.443310    0.836000
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1-laser-fr/models/sp30k/lstm_nl4.m
Loss and accuracy using (cls_best): [0.80693215, tensor(0.7688)]
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1-laser-de/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1-laser-de/models/sp30k/lstm_nl4.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1-laser-de/ru.dev.csv
Running tokenization...
Saving tokenized: cls.trn 9195, cls.val 1021
Running tokenization...
Saving tokenized: cls.trn 1000, cls.val 1000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁.', '▁в', 'а', '▁и', 'е', 'и', 'й', '▁на', 'х']
Training args:  {'tie_weights': True, 'clip': 0.12, 'bptt': 70, 'pretrained_fnames': [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp30k/lstm_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp30k/lstm_nl4.m/../itos')], 'pretrained_model': None, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  [0.25 0.1  0.2  0.02 0.15]
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1-laser-de/models/sp30k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1-laser-de/models/sp30k/lstm_nl4.m/info.json
Starting classifier training
epoch     train_loss  valid_loss  accuracy
1         0.878202    0.549530    0.815000
epoch     train_loss  valid_loss  accuracy
1         0.747663    0.439798    0.860000
epoch     train_loss  valid_loss  accuracy
1         0.610381    0.391122    0.878000
epoch     train_loss  valid_loss  accuracy
1         0.563902    0.393633    0.880000
2         0.515117    0.403987    0.878000
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1-laser-de/models/sp30k/lstm_nl4.m
Loss and accuracy using (cls_best): [1.3181443, tensor(0.6695)]
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1-laser-en/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1-laser-en/models/sp30k/lstm_nl4.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1-laser-en/ru.dev.csv
Running tokenization...
Saving tokenized: cls.trn 9195, cls.val 1021
Running tokenization...
Saving tokenized: cls.trn 1000, cls.val 1000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁.', '▁в', 'а', '▁и', 'е', 'и', 'й', '▁на', 'х']
Training args:  {'tie_weights': True, 'clip': 0.12, 'bptt': 70, 'pretrained_fnames': [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp30k/lstm_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp30k/lstm_nl4.m/../itos')], 'pretrained_model': None, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  [0.25 0.1  0.2  0.02 0.15]
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1-laser-en/models/sp30k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1-laser-en/models/sp30k/lstm_nl4.m/info.json
Starting classifier training
epoch     train_loss  valid_loss  accuracy
1         0.897468    0.570228    0.801000
epoch     train_loss  valid_loss  accuracy
1         0.704874    0.560132    0.812000
epoch     train_loss  valid_loss  accuracy
1         0.595008    0.507041    0.816000
epoch     train_loss  valid_loss  accuracy
1         0.484754    0.479213    0.825000
2         0.454896    0.501114    0.824000
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1-laser-en/models/sp30k/lstm_nl4.m
Loss and accuracy using (cls_best): [1.1765001, tensor(0.7005)]
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1-laser-fr/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1-laser-fr/models/sp30k/lstm_nl4.m
Evaluating previously trained model
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1-laser-fr/ru.dev.csv
Tokenized data loaded, lm.trn 9195, lm.val 1021
Tokenized data loaded, cls.trn 1000, cls.val 1000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁.', '▁в', 'а', '▁и', 'е', 'и', 'й', '▁на', 'х']
Loss and accuracy using (cls_last): [1.0429544, tensor(0.6833)]
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1-laser-de/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1-laser-de/models/sp30k/lstm_nl4.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1-laser-de/zh.dev.csv
Running tokenization...
Saving tokenized: cls.trn 13500, cls.val 1500
Running tokenization...
Saving tokenized: cls.trn 1000, cls.val 1000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁的', '▁。', '▁年', '▁、', '▁在', '▁一', '▁中', '▁人', '▁是']
Training args:  {'tie_weights': True, 'clip': 0.12, 'bptt': 70, 'pretrained_fnames': [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp30k/lstm_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp30k/lstm_nl4.m/../itos')], 'pretrained_model': None, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  [0.25 0.1  0.2  0.02 0.15]
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1-laser-de/models/sp30k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1-laser-de/models/sp30k/lstm_nl4.m/info.json
Starting classifier training
epoch     train_loss  valid_loss  accuracy
1         0.759435    0.762386    0.707000
epoch     train_loss  valid_loss  accuracy
1         0.631534    0.591862    0.786000
epoch     train_loss  valid_loss  accuracy
1         0.534237    0.589429    0.801000
epoch     train_loss  valid_loss  accuracy
1         0.454291    0.589220    0.799000
2         0.446990    0.586956    0.804000
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1-laser-de/models/sp30k/lstm_nl4.m
Loss and accuracy using (cls_best): [0.8401224, tensor(0.7232)]
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1-laser-en/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1-laser-en/models/sp30k/lstm_nl4.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1-laser-en/zh.dev.csv
Running tokenization...
Saving tokenized: cls.trn 13500, cls.val 1500
Running tokenization...
Saving tokenized: cls.trn 1000, cls.val 1000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁的', '▁。', '▁年', '▁、', '▁在', '▁一', '▁中', '▁人', '▁是']
Training args:  {'tie_weights': True, 'clip': 0.12, 'bptt': 70, 'pretrained_fnames': [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp30k/lstm_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp30k/lstm_nl4.m/../itos')], 'pretrained_model': None, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  [0.25 0.1  0.2  0.02 0.15]
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1-laser-en/models/sp30k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1-laser-en/models/sp30k/lstm_nl4.m/info.json
Starting classifier training
epoch     train_loss  valid_loss  accuracy
1         0.827517    0.821250    0.712000
epoch     train_loss  valid_loss  accuracy
1         0.636761    0.656195    0.772000
epoch     train_loss  valid_loss  accuracy
1         0.582199    0.675501    0.769000
epoch     train_loss  valid_loss  accuracy
1         0.511542    0.634232    0.764000
2         0.508244    0.647197    0.771000
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1-laser-en/models/sp30k/lstm_nl4.m
Loss and accuracy using (cls_best): [0.5421255, tensor(0.8045)]
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1-laser-fr/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1-laser-fr/models/sp30k/lstm_nl4.m
Evaluating previously trained model
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1-laser-fr/zh.dev.csv
Tokenized data loaded, lm.trn 13500, lm.val 1500
Tokenized data loaded, cls.trn 1000, cls.val 1000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁的', '▁。', '▁年', '▁、', '▁在', '▁一', '▁中', '▁人', '▁是']
Loss and accuracy using (cls_last): [0.5289812, tensor(0.8465)]
OrderedDict([('data/mldoc/de-1-laser-de/models/sp30k/lstm_nl4.m',
              0.9545000195503235),
             ('data/mldoc/de-1-laser-en/models/sp30k/lstm_nl4.m',
              0.9294999837875366),
             ('data/mldoc/de-1-laser-fr/models/sp30k/lstm_nl4.m',
              0.922249972820282),
             ('data/mldoc/es-1-laser-de/models/sp30k/lstm_nl4.m',
              0.8550000190734863),
             ('data/mldoc/es-1-laser-en/models/sp30k/lstm_nl4.m',
              0.8050000071525574),
             ('data/mldoc/es-1-laser-fr/models/sp30k/lstm_nl4.m',
              0.8100000023841858),
             ('data/mldoc/fr-1-laser-de/models/sp30k/lstm_nl4.m',
              0.8737499713897705),
             ('data/mldoc/fr-1-laser-en/models/sp30k/lstm_nl4.m',
              0.8877500295639038),
             ('data/mldoc/fr-1-laser-fr/models/sp30k/lstm_nl4.m',
              0.9357500076293945),
             ('data/mldoc/it-1-laser-de/models/sp30k/lstm_nl4.m',
              0.7875000238418579),
             ('data/mldoc/it-1-laser-en/models/sp30k/lstm_nl4.m',
              0.7620000243186951),
             ('data/mldoc/it-1-laser-fr/models/sp30k/lstm_nl4.m',
              0.768750011920929),
             ('data/mldoc/ru-1-laser-de/models/sp30k/lstm_nl4.m',
              0.6694999933242798),
             ('data/mldoc/ru-1-laser-en/models/sp30k/lstm_nl4.m',
              0.7005000114440918),
             ('data/mldoc/ru-1-laser-fr/models/sp30k/lstm_nl4.m',
              0.6832500100135803),
             ('data/mldoc/zh-1-laser-de/models/sp30k/lstm_nl4.m',
              0.7232499718666077),
             ('data/mldoc/zh-1-laser-en/models/sp30k/lstm_nl4.m',
              0.8044999837875366),
             ('data/mldoc/zh-1-laser-fr/models/sp30k/lstm_nl4.m',
              0.8464999794960022)])
```


### Laser 10k
### Building dataset 10k
```
 for SRC_LANG in en de fr; do                                                                                                                                         ✘ 130
    for LANG in en de es fr it ru zh; do
        echo $LANG from $SRC_LANG
        python ../../source/classify.py embed10000/mldoc.${SRC_LANG}-${SRC_LANG}.h5 ~/workspace/ulmfit-multilingual/data/mldoc/${LANG}-10 --suffix=10 | grep Test:
    done
done
zsh: command not found: ✘
en from en
 | Test: 92.70% | classes: 24.65 25.52 26.20 23.62
de from en
 | Test: 87.43% | classes: 20.05 25.55 28.88 25.52
es from en
 | Test: 77.38% | classes: 24.70 15.40 22.27 37.62
fr from en
 | Test: 78.70% | classes: 17.65 29.75 34.60 18.00
it from en
 | Test: 72.53% | classes: 20.75 24.60 28.52 26.12
ru from en
 | Test: 67.70% | classes: 33.40 17.12 29.35 20.12
zh from en
 | Test: 75.18% | classes: 27.70 39.00 12.07 21.23
zsh: command not found: ✘
en from de
 | Test: 81.60% | classes: 31.62 23.82 24.93 19.62
de from de
 | Test: 95.40% | classes: 24.57 26.00 25.62 23.80
es from de
 | Test: 83.50% | classes: 28.73 22.60 18.65 30.02
fr from de
 | Test: 82.85% | classes: 27.52 29.23 25.23 18.02
it from de
 | Test: 76.60% | classes: 26.52 25.62 21.43 26.43
ru from de
 | Test: 68.80% | classes: 48.48 13.22 19.43 18.88
zh from de
 | Test: 73.12% | classes: 33.12 40.10 10.53 16.25
zsh: command not found: ✘
en from fr
 | Test: 85.08% | classes: 25.43 23.43 25.43 25.73
de from fr
 | Test: 91.65% | classes: 23.15 27.10 25.00 24.75
es from fr
 | Test: 81.05% | classes: 24.18 17.68 18.93 39.23
fr from fr
 | Test: 93.65% | classes: 24.25 25.10 25.80 24.85
it from fr
 | Test: 75.08% | classes: 24.07 26.85 18.25 30.82
ru from fr
 | Test: 70.73% | classes: 43.17 19.15 16.57 21.10
zh from fr
 | Test: 76.33% | classes: 34.23 34.05 10.95 20.77
 ```
  1 10
### Building dataset 1k
```
 for SRC_LANG in en de fr; do                                                                                                                                        
    for LANG in en de es fr it ru zh; do
        echo $LANG from $SRC_LANG
        python ../../source/classify.py embed1000/mldoc.${SRC_LANG}-${SRC_LANG}.h5 ~/workspace/ulmfit-multilingual/data/mldoc/${LANG}-10 --suffix=1 | grep Test:
    done
 done
 for SRC_LANG in en de fr; do
    for LANG in en de es fr it ru zh; do
        echo $LANG from $SRC_LANG
        python ../../source/classify.py embed1000/mldoc.${SRC_LANG}-${SRC_LANG}.h5 ~/workspace/ulmfit-multilingual/data/mldoc/${LANG}-10 --suffix=1 | grep Test:
    done
 done
en from en
 | Test: 91.48% | classes: 23.77 24.90 26.25 25.07
de from en
 | Test: 87.65% | classes: 21.98 24.45 27.65 25.93
es from en
 | Test: 75.48% | classes: 21.60 15.82 22.10 40.48
fr from en
 | Test: 84.00% | classes: 23.18 29.12 27.90 19.80
it from en
 | Test: 71.18% | classes: 23.65 22.88 25.68 27.80
ru from en
 | Test: 66.58% | classes: 29.48 13.78 34.52 22.23
zh from en
 | Test: 76.65% | classes: 30.25 31.30 13.93 24.52
en from de
 | Test: 78.23% | classes: 31.80 17.73 30.15 20.32
de from de
 | Test: 93.50% | classes: 24.45 25.45 26.00 24.10
es from de
 | Test: 81.40% | classes: 24.15 25.77 20.12 29.95
fr from de
 | Test: 81.50% | classes: 25.52 29.45 27.45 17.57
it from de
 | Test: 74.53% | classes: 24.70 27.25 22.43 25.62
ru from de
 | Test: 64.58% | classes: 45.62  9.12 26.73 18.52
zh from de
 | Test: 73.20% | classes: 31.20 43.38  7.60 17.82
en from fr
 | Test: 81.30% | classes: 28.95 18.02 24.98 28.05
de from fr
 | Test: 88.75% | classes: 24.00 23.75 24.85 27.40
es from fr
 | Test: 80.12% | classes: 24.50 14.82 18.40 42.27
fr from fr
 | Test: 90.85% | classes: 24.50 24.75 24.68 26.07
it from fr
 | Test: 72.58% | classes: 25.45 24.10 17.50 32.95
ru from fr
 | Test: 67.35% | classes: 47.15 13.62 16.68 22.55
zh from fr
 | Test: 79.40% | classes: 33.60 31.12  9.07 26.20
```
### No Unfreeze
#### one epoch
```
python -m ulmfit cls --dataset-path data/mldoc/fr-1-laser  --base-lm-path data/mldoc/fr-1/models/sp30k/lstm_nl4.m  --lang=fr --name 'nl4-no_unfreeze' --cuda-id=1 - train 0 --bs 40 --num-cls-epochs=2 --unfreeze=False
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1-laser/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1-laser/models/sp30k/lstm_nl4-no_unfreeze.m
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1-laser/fr.dev.csv
Tokenized data loaded, lm.trn 13500, lm.val 1500
Tokenized data loaded, cls.trn 1000, cls.val 1000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁.', "'", 's', '▁la', '▁le', '▁et', '▁l', '▁à']
Training args:  {'tie_weights': True, 'clip': 0.12, 'bptt': 70, 'pretrained_fnames': [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/models/sp30k/lstm_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/models/sp30k/lstm_nl4.m/../itos')], 'pretrained_model': None, 'drop_mult': 0.3} dps:  [0.25 0.1  0.2  0.02 0.15]
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1-laser/models/sp30k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1-laser/models/sp30k/lstm_nl4-no_unfreeze.m/info.json
Starting classifier training
epoch     train_loss  valid_loss  accuracy
1         0.800256    0.783174    0.701000
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1-laser/models/sp30k/lstm_nl4-no_unfreeze.m
Loss and accuracy using (cls_best): [1.1735736, tensor(0.5077)]
1.173573613166809
0.5077499747276306
```
#### 4 epochs
ulmfit: 63.67%
```
python -m ulmfit cls --dataset-path data/mldoc/fr-1-laser  --base-lm-path data/mldoc/fr-1/models/sp30k/lstm_nl4.m  --lang=fr --name 'nl4-no_unfreeze2' --cuda-id=1 - train 0 --bs 40 --num-cls-epochs=2 --unfreeze=False --num-cls-frozen-epochs=4
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1-laser/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1-laser/models/sp30k/lstm_nl4-no_unfreeze2.m
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1-laser/fr.dev.csv
Tokenized data loaded, lm.trn 13500, lm.val 1500
Tokenized data loaded, cls.trn 1000, cls.val 1000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁.', "'", 's', '▁la', '▁le', '▁et', '▁l', '▁à']
Training args:  {'tie_weights': True, 'clip': 0.12, 'bptt': 70, 'pretrained_fnames': [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/models/sp30k/lstm_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/models/sp30k/lstm_nl4.m/../itos')], 'pretrained_model': None, 'drop_mult': 0.3} dps:  [0.25 0.1  0.2  0.02 0.15]
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1-laser/models/sp30k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1-laser/models/sp30k/lstm_nl4-no_unfreeze2.m/info.json
Starting classifier training
epoch     train_loss  valid_loss  accuracy
1         0.832118    0.750073    0.717000
2         0.729266    0.617375    0.749000
3         0.645946    0.623189    0.751000
4         0.566385    0.608672    0.760000
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1-laser/models/sp30k/lstm_nl4-no_unfreeze2.m
Loss and accuracy using (cls_best): [0.97152597, tensor(0.6367)]
```