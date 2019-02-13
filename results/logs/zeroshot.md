# Laser Performance
Accuracy matrix:

| Train |   en  |   de  |   es  |   fr  |   it  |   ru  |   zh  |
|-------|-------|-------|-------|-------|-------|-------|-------|
|  en:  | 90.88 | 86.48 | 67.62 | 61.98 | 69.95 | 22.95 | 11.65 |
|  de:  | 73.23 | 92.90 | 77.23 | 74.05 | 72.30 | 24.80 |  9.93 |
|  es:  | 65.62 | 80.58 | 92.03 | 73.28 | 69.03 | 34.10 | 12.58 |
|  fr:  | 78.35 | 85.45 | 78.20 | 89.68 | 69.85 | 33.88 |  9.68 |
|  it:  | 73.93 | 84.58 | 79.23 | 76.73 | 84.03 | 34.48 | 11.83 |
|  ru:  | 57.33 | 63.78 | 45.80 | 52.78 | 51.15 | 66.08 | 36.28 |
|  zh:  | 26.15 | 28.13 | 21.88 | 29.33 | 30.58 | 34.38 | 75.62 |
 
# DE
Laser 0shot: 86.48, ULMFiT 0shot: 91.97
```
python ../../source/classify.py embed-2019-02-12/mldoc.en-en.h5 ~/workspace/ulmfit-multilingual/data/mldoc/de-1
 | Test: 86.48% | classes: 24.30 22.77 28.90 24.02
 Making train set
 | Train: 85.70% | classes: 27.00 21.40 27.60 24.00
Accuracy 0.857
   0                                                  1
0  3  Tokio (Reuter) - Der Dollar ist am Donnerstag ...
1  3  Kairo (Reuter) - Die ägyptische Zentralbank se...
2  2  Bonn (Reuter) - Wegen einer Bombendrohung ist ...
3  0  Berlin (Reuter) - Die Bahn AG will mit Hilfe p...
4  3  08.15 Uhr MEZ - Deutsche Aktien nach den Rekor...

 Making dev set
 | Train: 85.60% | classes: 23.70 22.30 30.60 23.40
Accuracy 0.856
   0                                                  1
0  1  New York (Reuter) - Das Vertrauen der US-Verbr...
1  2  Tokio (Reuter) - Russische Patrouillenboote ha...
2  2  Paris (Reuter) - Bei der Volksabstimmung in Al...
3  2  Belgrad (Reuter) - Die serbische Polizei hat n...
4  0  München (Reuter) - Der Stuttgarter Bosch-Konze...
```
```
python -m ulmfit cls --dataset-path data/mldoc/de-1-laser  --base-lm-path data/mldoc/de-1/models/sp30k/lstm_nl4.m  --lang=de --name 'nl4' --cuda-id=1 - train 0 --bs 40 --num-cls-epochs=2
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1-laser/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1-laser/models/sp30k/lstm_nl4.m
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1-laser/de.dev.csv
Running tokenization...
Saving tokenized: cls.trn 13500, cls.val 1500
Running tokenization...
Saving tokenized: cls.trn 1000, cls.val 1000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁.', '▁,', '▁der', '▁die', '▁und', '▁in', 'en', "▁&'", 's', '-']
Training args:  {'tie_weights': True, 'clip': 0.12, 'bptt': 70, 'pretrained_fnames': [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1/models/sp30k/lstm_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1/models/sp30k/lstm_nl4.m/../itos')], 'pretrained_model': None, 'drop_mult': 0.3} dps:  [0.25 0.1  0.2  0.02 0.15]
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1-laser/models/sp30k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1-laser/models/sp30k/lstm_nl4.m/info.json
Starting classifier training
epoch     train_loss  valid_loss  accuracy
1         0.671869    0.466408    0.863000
epoch     train_loss  valid_loss  accuracy
1         0.518045    0.388151    0.887000
epoch     train_loss  valid_loss  accuracy
1         0.375156    0.370652    0.893000
epoch     train_loss  valid_loss  accuracy
1         0.339284    0.367223    0.891000
2         0.314325    0.369492    0.891000
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1-laser/models/sp30k/lstm_nl4.m
Loss and accuracy using (cls_best): [0.25416428, tensor(0.9197)]
0.25416427850723267
0.9197499752044678
```





# ES from IT
```
python ../../source/classify.py embed-2019-02-12/mldoc.it-it.h5 ~/workspace/ulmfit-multilingual/data/mldoc/es-1                                                      ✘ 130
 | Test: 79.23% | classes: 25.48 16.45 24.18 33.90
 Making train set
 | Train: 80.30% | classes: 27.10 19.20 22.60 31.10
Accuracy 0.803
   0                                                  1
0  3  LONDRES, 5 sep (Reuter) - El dólar se mantenía...
1  1  MADRID, 30 dic (Reuter) - La Generalitat de Va...
2  3  PARIS, 30 jun (Reuter) - La Bolsa de París neg...
3  0  MADRID, 23 dic (Reuter) - La agencia de valore...
4  0  MADRID, 4 Feb (Reuter) - El Banco Bilbao Vizca...

 Making dev set
 | Train: 79.70% | classes: 25.40 17.50 26.20 30.90
Accuracy 0.797
   0                                                  1
0  0  NUEVA YORK, 11 abr (Reuter) - MCI Communicatio...
1  3  FRANCFORT, 17 jun (Reuter) - La Bolsa de Franc...
2  1  BONN, 3 jun (Reuter) - Un destacado miembro de...
3  2  LONDRES, 3 sep (Reuter) - El secretario de Def...
4  2  MADRID, 3 oct (Reuter) - Las acciones de Pryca...
```

```
python -m ulmfit cls --dataset-path data/mldoc/es-1-laser-it  --base-lm-path data/mldoc/es-1/models/sp30k/lstm_nl4.m  --lang=es --name 'nl4' --cuda-id=1 - train 0 --bs 40 --num-cls-epochs=2
```

# FR from IT
```
python ../../source/classify.py embed-2019-02-12/mldoc.it-it.h5 ~/workspace/ulmfit-multilingual/data/mldoc/fr-1
 | Test: 76.73% | classes: 21.65 21.98 31.77 24.60
 Making train set
 | Train: 79.20% | classes: 22.20 22.40 31.40 24.00
Accuracy 0.792
   0                                                  1
0  2  WASHINGTON, 13 septembre, Reuter - Les Etats-U...
1  1  PARIS, 10 juillet, Reuter - L'audit des financ...
2  2  MOSCOU, 29 mai, Reuter - Après l'accord interv...
3  2  PARIS, 1er octobre, Reuter - Le groupe communi...
4  0  LONDRES, 3 juin, Reuter - National Grid Group ...

 Making dev set
 | Train: 76.60% | classes: 23.30 20.10 33.00 23.60
Accuracy 0.766
   0                                                  1
0  0  PARIS, 30 décembre, Reuter - Zodiac . Chiffre ...
1  0  AJACCIO, 11 décembre, Reuter - Une charge de 7...
2  0  BRUXELLES, 26 décembre, Reuter - 1997 s'annonc...
3  0  PARIS, 26 septembre, Reuter - Alcatel Alsthom ...
4  0  NEW YORK, 25 octobre, Reuter - La hausse plus ...
```

```
python -m ulmfit cls --dataset-path data/mldoc/fr-1-laser-it  --base-lm-path data/mldoc/fr-1/models/sp30k/lstm_nl4.m  --lang=fr --name 'nl4' --cuda-id=1 - train 0 --bs 40 --num-cls-epochs=2
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1-laser-it/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1-laser-it/models/sp30k/lstm_nl4.m
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1-laser-it/fr.dev.csv
Running tokenization...
Saving tokenized: cls.trn 13500, cls.val 1500
Running tokenization...
Saving tokenized: cls.trn 1000, cls.val 1000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁.', "'", 's', '▁la', '▁le', '▁et', '▁l', '▁à']
Training args:  {'tie_weights': True, 'clip': 0.12, 'bptt': 70, 'pretrained_fnames': [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/models/sp30k/lstm_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/models/sp30k/lstm_nl4.m/../itos')], 'pretrained_model': None, 'drop_mult': 0.3} dps:  [0.25 0.1  0.2  0.02 0.15]
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1-laser-it/models/sp30k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1-laser-it/models/sp30k/lstm_nl4.m/info.json
Starting classifier training
epoch     train_loss  valid_loss  accuracy
1         0.737947    0.627607    0.793000
epoch     train_loss  valid_loss  accuracy
1         0.603060    0.513449    0.831000
epoch     train_loss  valid_loss  accuracy
1         0.481312    0.499689    0.828000
epoch     train_loss  valid_loss  accuracy
1         0.422958    0.508330    0.825000
2         0.408061    0.493875    0.839000
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1-laser-it/models/sp30k/lstm_nl4.m
Loss and accuracy using (cls_best): [0.4295174, tensor(0.8555)]
0.42951738834381104
0.8554999828338623
```
# FR From EN
```
python ../../source/classify.py embed-2019-02-12/mldoc.en-en.h5 ~/workspace/ulmfit-multilingual/data/mldoc/fr-1
 | Test: 61.98% | classes: 11.85 41.10 40.05  7.00
 Making train set
 | Train: 63.70% | classes: 11.70 43.80 38.40  6.10
Accuracy 0.637
   0                                                  1
0  2  WASHINGTON, 13 septembre, Reuter - Les Etats-U...
1  1  PARIS, 10 juillet, Reuter - L'audit des financ...
2  2  MOSCOU, 29 mai, Reuter - Après l'accord interv...
3  2  PARIS, 1er octobre, Reuter - Le groupe communi...
4  0  LONDRES, 3 juin, Reuter - National Grid Group ...

 Making dev set
 | Train: 61.60% | classes: 11.90 40.90 39.70  7.50
Accuracy 0.616
   0                                                  1
0  1  PARIS, 30 décembre, Reuter - Zodiac . Chiffre ...
1  0  AJACCIO, 11 décembre, Reuter - Une charge de 7...
2  1  BRUXELLES, 26 décembre, Reuter - 1997 s'annonc...
3  1  PARIS, 26 septembre, Reuter - Alcatel Alsthom ...
4  1  NEW YORK, 25 octobre, Reuter - La hausse plus ...
```
```

python -m ulmfit cls --dataset-path data/mldoc/fr-1-laser  --base-lm-path data/mldoc/fr-1/models/sp30k/lstm_nl4.m  --lang=fr --name 'nl4-laser' --cuda-id=1 - train 0 --bs 40 --num-cls-epochs=2
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1-laser/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1-laser/models/sp30k/lstm_nl4-laser.m
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1-laser/fr.dev.csv
Running tokenization...
Saving tokenized: cls.trn 13500, cls.val 1500
Running tokenization...
Saving tokenized: cls.trn 1000, cls.val 1000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁.', "'", 's', '▁la', '▁le', '▁et', '▁l', '▁à']
Training args:  {'tie_weights': True, 'clip': 0.12, 'bptt': 70, 'pretrained_fnames': [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/models/sp30k/lstm_nl4.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/models/sp30k/lstm_nl4.m/../itos')], 'pretrained_model': None, 'drop_mult': 0.3} dps:  [0.25 0.1  0.2  0.02 0.15]
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1-laser/models/sp30k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1-laser/models/sp30k/lstm_nl4-laser.m/info.json
Starting classifier training
epoch     train_loss  valid_loss  accuracy
1         0.797327    0.697984    0.730000
epoch     train_loss  valid_loss  accuracy
1         0.639780    0.582377    0.763000
epoch     train_loss  valid_loss  accuracy
1         0.585295    0.582596    0.762000
epoch     train_loss  valid_loss  accuracy
1         0.482629    0.582803    0.765000
2         0.470849    0.582416    0.771000
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1-laser/models/sp30k/lstm_nl4-laser.m
Loss and accuracy using (cls_best): [0.80327946, tensor(0.6920)]
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