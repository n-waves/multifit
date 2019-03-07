## Debugging random init
````
  warnings.warn(ABI_INCOMPATIBILITY_WARNING.format(compiler))
Loading pretrained model
Unknown tokens 0, first 100: []
Bptt 70
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4_rnd2_0.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
````





## first attempt at random init
```
 python -m ulmfit eval_noise_resistance --lang=de --size=10 --prefix-name="_rnd_" --model="sp15k/qrnn_rnd-nl4.m" --label-smoothing-eps=0.1
Noise:  0
Processing data/mldoc/de-1/models/sp15k/qrnn_rnd-nl4.m
de-10
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4__rnd_0.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/de.dev.csv
Data lm, trn: 13500, val: 1500
Data cls, trn: 10000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁.', '▁,', '▁der', '▁die', 'en', '▁und', 's', '▁in', 'er', "▁&'"]
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
/home/pczapla/anaconda3/envs/fastaiv1/lib/python3.7/site-packages/torch/utils/cpp_extension.py:152: UserWarning:

                               !! WARNING !!

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
Your compiler (c++) may be ABI-incompatible with PyTorch!
Please use a compiler that is ABI-compatible with GCC 4.9 and above.
See https://gcc.gnu.org/onlinedocs/libstdc++/manual/abi.html.

See https://gist.github.com/goldsborough/d466f43e8ffc948ff92de7486c5216d6
for instructions on how to install GCC 4.9 or higher.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                              !! WARNING !!

  warnings.warn(ABI_INCOMPATIBILITY_WARNING.format(compiler))
Loading pretrained model
Unknown tokens 0, first 100: []
Bptt 70
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4__rnd_0.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.787174    0.985413    0.701000
2         0.679534    0.697764    0.875000
3         0.630987    7.125103    0.873000
4         0.588259    0.653497    0.915000
5         0.568135    0.641379    0.942000
6         0.529713    0.557198    0.948000
7         0.500168    0.538946    0.958000
8         0.505462    0.550917    0.954000
Total time: 19:37
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4__rnd_0.m
Loss and accuracy using (cls_best): [0.21539633, tensor(0.9613)]
OrderedDict([('data/mldoc/de-10/models/sp15k/qrnn_nl4__rnd_0.m',
              0.9612500071525574)])
Noise:  5
Processing data/mldoc/de-1/models/sp15k/qrnn_rnd-nl4.m
de-10
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4__rnd_5.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/de.dev.csv
Added noise to 500 examples, only 0.95 have correct labels
Added noise to 50 examples, only 0.95 have correct labels
Data lm, trn: 13500, val: 1500
Data clsnoise0.05tv, trn: 10000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁.', '▁,', '▁der', '▁die', 'en', '▁und', 's', '▁in', 'er', "▁&'"]
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
Bptt 70
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4__rnd_5.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.917813    0.874308    0.790000
2         0.821460    1.239760    0.671000
3         0.747909    2.624352    0.667000
4         0.713649    0.735327    0.893000
5         0.689947    1.175884    0.844000
6         0.639073    1.066042    0.862000
7         0.617660    0.845634    0.875000
8         0.620402    0.670972    0.901000
Total time: 19:28
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4__rnd_5.m
Loss and accuracy using (cls_best): [0.25263783, tensor(0.9560)]
OrderedDict([('data/mldoc/de-10/models/sp15k/qrnn_nl4__rnd_5.m',
              0.9559999704360962)])
Noise:  10
Processing data/mldoc/de-1/models/sp15k/qrnn_rnd-nl4.m
de-10
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4__rnd_10.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/de.dev.csv
Added noise to 1000 examples, only 0.9 have correct labels
Added noise to 100 examples, only 0.9 have correct labels
Data lm, trn: 13500, val: 1500
Data clsnoise0.1tv, trn: 10000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁.', '▁,', '▁der', '▁die', 'en', '▁und', 's', '▁in', 'er', "▁&'"]
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
Bptt 70
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4__rnd_10.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.976570    1.054016    0.705000
2         0.885775    0.813666    0.835000
3         0.861244    0.968860    0.762000
4         0.790501    0.815453    0.839000
5         0.754292    0.805088    0.849000
6         0.742595    0.770547    0.864000
7         0.712961    0.771171    0.863000
8         0.695449    0.787870    0.858000
Total time: 19:48
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4__rnd_10.m
Loss and accuracy using (cls_best): [0.2744636, tensor(0.9510)]
OrderedDict([('data/mldoc/de-10/models/sp15k/qrnn_nl4__rnd_10.m',
              0.9509999752044678)])
Noise:  15
Processing data/mldoc/de-1/models/sp15k/qrnn_rnd-nl4.m
de-10
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4__rnd_15.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/de.dev.csv
Added noise to 1500 examples, only 0.85 have correct labels
Added noise to 150 examples, only 0.85 have correct labels
Data lm, trn: 13500, val: 1500
Data clsnoise0.15tv, trn: 10000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁.', '▁,', '▁der', '▁die', 'en', '▁und', 's', '▁in', 'er', "▁&'"]
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
Bptt 70
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4__rnd_15.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.081478    1.075213    0.674000
2         0.989475    0.939438    0.779000
3         0.963180    0.984908    0.723000
4         0.915556    1.209332    0.662000
5         0.884642    1.000015    0.786000
6         0.844702    0.884871    0.794000
7         0.793699    0.882503    0.802000
8         0.797470    0.871922    0.802000
Total time: 19:50
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4__rnd_15.m
Loss and accuracy using (cls_best): [0.32492134, tensor(0.9445)]
OrderedDict([('data/mldoc/de-10/models/sp15k/qrnn_nl4__rnd_15.m',
              0.9445000290870667)])
Noise:  20
Processing data/mldoc/de-1/models/sp15k/qrnn_rnd-nl4.m
de-10
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4__rnd_20.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/de.dev.csv
Added noise to 2000 examples, only 0.8 have correct labels
Added noise to 200 examples, only 0.8 have correct labels
Data lm, trn: 13500, val: 1500
Data clsnoise0.2tv, trn: 10000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁.', '▁,', '▁der', '▁die', 'en', '▁und', 's', '▁in', 'er', "▁&'"]
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
Bptt 70
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4__rnd_20.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.130177    1.651658    0.424000
2         1.049187    1.508039    0.286000
3         1.045260    1.976680    0.578000
4         0.971859    1.121615    0.735000
5         0.965327    2.376971    0.684000
6         0.901961    1.089674    0.744000
7         0.868971    1.082978    0.750000
8         0.845376    1.019824    0.740000
Total time: 19:51
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4__rnd_20.m
Loss and accuracy using (cls_best): [0.46514454, tensor(0.9438)]
OrderedDict([('data/mldoc/de-10/models/sp15k/qrnn_nl4__rnd_20.m',
              0.9437500238418579)])
Noise:  25
Processing data/mldoc/de-1/models/sp15k/qrnn_rnd-nl4.m
de-10
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4__rnd_25.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/de.dev.csv
Added noise to 2500 examples, only 0.75 have correct labels
Added noise to 250 examples, only 0.75 have correct labels
Data lm, trn: 13500, val: 1500
Data clsnoise0.25tv, trn: 10000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁.', '▁,', '▁der', '▁die', 'en', '▁und', 's', '▁in', 'er', "▁&'"]
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
Bptt 70
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4__rnd_25.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.204033    1.368233    0.500000
2         1.121006    1.219437    0.586000
3         1.057657    1.139297    0.659000
4         1.054685    1.043641    0.700000
5         1.023957    1.069890    0.706000
6         0.992645    1.073037    0.708000
7         0.948602    1.054931    0.699000
8         0.945395    1.078187    0.703000
Total time: 20:09
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4__rnd_25.m
Loss and accuracy using (cls_best): [0.4676742, tensor(0.9137)]
OrderedDict([('data/mldoc/de-10/models/sp15k/qrnn_nl4__rnd_25.m',
              0.9137499928474426)])
Noise:  30
Processing data/mldoc/de-1/models/sp15k/qrnn_rnd-nl4.m
de-10
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4__rnd_30.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/de.dev.csv
Added noise to 3000 examples, only 0.7 have correct labels
Added noise to 300 examples, only 0.7 have correct labels
Data lm, trn: 13500, val: 1500
Data clsnoise0.3tv, trn: 10000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁.', '▁,', '▁der', '▁die', 'en', '▁und', 's', '▁in', 'er', "▁&'"]
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
Bptt 70
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4__rnd_30.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.245468    1.243019    0.492000
2         1.192702    1.644169    0.435000
3         1.187665    4.143492    0.490000
4         1.113116    20.139246   0.540000
5         1.092624    1.189916    0.609000
6         1.052626    1.264737    0.617000
7         1.032403    1.317357    0.649000
8         1.003000    1.187038    0.653000
Total time: 20:02
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4__rnd_30.m
Loss and accuracy using (cls_best): [0.5831716, tensor(0.9215)]
OrderedDict([('data/mldoc/de-10/models/sp15k/qrnn_nl4__rnd_30.m',
              0.921500027179718)])
Noise:  35
Processing data/mldoc/de-1/models/sp15k/qrnn_rnd-nl4.m
de-10
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4__rnd_35.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/de.dev.csv
Added noise to 3500 examples, only 0.65 have correct labels
Added noise to 350 examples, only 0.65 have correct labels
Data lm, trn: 13500, val: 1500
Data clsnoise0.35tv, trn: 10000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁.', '▁,', '▁der', '▁die', 'en', '▁und', 's', '▁in', 'er', "▁&'"]
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
Bptt 70
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4__rnd_35.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.285933    1.719733    0.309000
2         1.258459    1.465174    0.422000
3         1.240111    1.205106    0.512000
4         1.195793    2.153573    0.571000
5         1.150691    3.427428    0.588000
6         1.115649    1.933489    0.601000
7         1.078265    1.214095    0.599000
8         1.045297    1.140148    0.604000
Total time: 19:55
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4__rnd_35.m
Loss and accuracy using (cls_best): [0.5903087, tensor(0.9105)]
OrderedDict([('data/mldoc/de-10/models/sp15k/qrnn_nl4__rnd_35.m',
              0.9104999899864197)])
Noise:  40
Processing data/mldoc/de-1/models/sp15k/qrnn_rnd-nl4.m
de-10
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4__rnd_40.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/de.dev.csv
Added noise to 4000 examples, only 0.6 have correct labels
Added noise to 400 examples, only 0.6 have correct labels
Data lm, trn: 13500, val: 1500
Data clsnoise0.4tv, trn: 10000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁.', '▁,', '▁der', '▁die', 'en', '▁und', 's', '▁in', 'er', "▁&'"]
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
Bptt 70
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4__rnd_40.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.313393    1.523955    0.316000
2         1.303059    1.964390    0.418000
3         1.291624    1.550615    0.458000
4         1.263588    2.995128    0.390000
5         1.206715    1.265662    0.524000
6         1.191890    1.221754    0.536000
7         1.162122    1.223106    0.527000
8         1.150922    1.240103    0.531000
Total time: 19:53
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4__rnd_40.m
Loss and accuracy using (cls_best): [0.7210464, tensor(0.8583)]
OrderedDict([('data/mldoc/de-10/models/sp15k/qrnn_nl4__rnd_40.m',
              0.8582500219345093)])
Noise:  45
Processing data/mldoc/de-1/models/sp15k/qrnn_rnd-nl4.m
de-10
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4__rnd_45.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/de.dev.csv
Added noise to 4500 examples, only 0.55 have correct labels
Added noise to 450 examples, only 0.55 have correct labels
Data lm, trn: 13500, val: 1500
Data clsnoise0.45tv, trn: 10000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁.', '▁,', '▁der', '▁die', 'en', '▁und', 's', '▁in', 'er', "▁&'"]
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
Bptt 70
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4__rnd_45.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.345056    1.343081    0.378000
2         1.281120    11.283777   0.232000
3         1.284114    14.679921   0.390000
4         1.267963    2.869378    0.485000
5         1.227434    1.466781    0.490000
6         1.209261    1.634938    0.495000
7         1.170042    1.372811    0.494000
8         1.162168    2.157310    0.492000
Total time: 20:05
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4__rnd_45.m
Loss and accuracy using (cls_best): [1.0457553, tensor(0.8635)]
OrderedDict([('data/mldoc/de-10/models/sp15k/qrnn_nl4__rnd_45.m',
              0.8634999990463257)])
Noise:  50
Processing data/mldoc/de-1/models/sp15k/qrnn_rnd-nl4.m
de-10
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4__rnd_50.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/de.dev.csv
Added noise to 5000 examples, only 0.5 have correct labels
Added noise to 500 examples, only 0.5 have correct labels
Data lm, trn: 13500, val: 1500
Data clsnoise0.5tv, trn: 10000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁.', '▁,', '▁der', '▁die', 'en', '▁und', 's', '▁in', 'er', "▁&'"]
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
Bptt 70
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4__rnd_50.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.362338    1.361343    0.339000
2         1.343794    1.358407    0.328000
3         1.326264    3.336083    0.325000
4         1.321352    4.200035    0.254000
5         1.289333    1.363007    0.408000
6         1.275341    1.449265    0.405000
7         1.245595    1.358157    0.423000
8         1.234815    1.346797    0.411000
Total time: 19:35
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4__rnd_50.m
Loss and accuracy using (cls_best): [1.3260584, tensor(0.7103)]
OrderedDict([('data/mldoc/de-10/models/sp15k/qrnn_nl4__rnd_50.m',
              0.7102500200271606)])
Noise:  55
Processing data/mldoc/de-1/models/sp15k/qrnn_rnd-nl4.m
de-10
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4__rnd_55.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/de.dev.csv
Added noise to 5500 examples, only 0.45 have correct labels
Added noise to 550 examples, only 0.45 have correct labels
Data lm, trn: 13500, val: 1500
Data clsnoise0.55tv, trn: 10000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁.', '▁,', '▁der', '▁die', 'en', '▁und', 's', '▁in', 'er', "▁&'"]
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
Bptt 70
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4__rnd_55.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.373838    1.385533    0.265000
2         1.355375    2.033619    0.316000
3         1.358652    2.010394    0.260000
4         1.337999    7.118755    0.351000
5         1.309082    3.053319    0.361000
6         1.286589    19.251106   0.359000
7         1.276432    1.328096    0.379000
8         1.266364    1.324883    0.378000
Total time: 19:34
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4__rnd_55.m
Loss and accuracy using (cls_best): [1.3107486, tensor(0.6503)]
OrderedDict([('data/mldoc/de-10/models/sp15k/qrnn_nl4__rnd_55.m',
              0.6502500176429749)])
Noise:  60
Processing data/mldoc/de-1/models/sp15k/qrnn_rnd-nl4.m
de-10
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4__rnd_60.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/de.dev.csv
Added noise to 6000 examples, only 0.4 have correct labels
Added noise to 600 examples, only 0.4 have correct labels
Data lm, trn: 13500, val: 1500
Data clsnoise0.6tv, trn: 10000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁.', '▁,', '▁der', '▁die', 'en', '▁und', 's', '▁in', 'er', "▁&'"]
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
Bptt 70
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4__rnd_60.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.378700    1.377784    0.309000
2         1.359247    9.472390    0.250000
3         1.343403    1.714557    0.321000
4         1.336044    1.331355    0.357000
5         1.322668    1.450317    0.332000
6         1.283835    2.692688    0.349000
7         1.261502    1.541230    0.335000
8         1.230086    1.839382    0.340000
Total time: 19:58
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4__rnd_60.m
Loss and accuracy using (cls_best): [1.1523782, tensor(0.5580)]
OrderedDict([('data/mldoc/de-10/models/sp15k/qrnn_nl4__rnd_60.m',
              0.5580000281333923)])
Noise:  65
Processing data/mldoc/de-1/models/sp15k/qrnn_rnd-nl4.m
de-10
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4__rnd_65.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/de.dev.csv
Added noise to 6500 examples, only 0.35 have correct labels
Added noise to 650 examples, only 0.35 have correct labels
Data lm, trn: 13500, val: 1500
Data clsnoise0.65tv, trn: 10000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁.', '▁,', '▁der', '▁die', 'en', '▁und', 's', '▁in', 'er', "▁&'"]
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
Bptt 70
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4__rnd_65.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.374414    1.361766    0.327000
2         1.367130    1.353700    0.341000
3         1.365781    1.421649    0.269000
4         1.358339    1.385666    0.280000
5         1.357855    3.068685    0.334000
6         1.343958    1.586822    0.316000
7         1.330202    2.436025    0.324000
8         1.322320    1.743209    0.330000
Total time: 19:52
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4__rnd_65.m
Loss and accuracy using (cls_best): [1.7415464, tensor(0.4467)]
OrderedDict([('data/mldoc/de-10/models/sp15k/qrnn_nl4__rnd_65.m',
              0.4467499852180481)])
Noise:  70
Processing data/mldoc/de-1/models/sp15k/qrnn_rnd-nl4.m
de-10
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4__rnd_70.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/de.dev.csv
Added noise to 7000 examples, only 0.3 have correct labels
Added noise to 700 examples, only 0.3 have correct labels
Data lm, trn: 13500, val: 1500
Data clsnoise0.7tv, trn: 10000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁.', '▁,', '▁der', '▁die', 'en', '▁und', 's', '▁in', 'er', "▁&'"]
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
Bptt 70
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4__rnd_70.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.386991    1.377855    0.302000
2         1.370086    1.741303    0.298000
3         1.371910    1.402328    0.316000
4         1.349717    1.378567    0.277000
5         1.360438    1.471136    0.298000
6         1.345680    1.395034    0.312000
7         1.327264    1.611867    0.312000
8         1.327243    1.657344    0.312000
Total time: 20:09
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4__rnd_70.m
Loss and accuracy using (cls_best): [2.7352421, tensor(0.2693)]
OrderedDict([('data/mldoc/de-10/models/sp15k/qrnn_nl4__rnd_70.m',
              0.2692500054836273)])
Noise:  75
Processing data/mldoc/de-1/models/sp15k/qrnn_rnd-nl4.m
de-10
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4__rnd_75.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/de.dev.csv
Added noise to 7500 examples, only 0.25 have correct labels
Added noise to 750 examples, only 0.25 have correct labels
Data lm, trn: 13500, val: 1500
Data clsnoise0.75tv, trn: 10000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁.', '▁,', '▁der', '▁die', 'en', '▁und', 's', '▁in', 'er', "▁&'"]
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
Bptt 70
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4__rnd_75.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.376097    1.392457    0.263000
2         1.372446    1.367712    0.320000
3         1.356304    1.354679    0.297000
4         1.342981    1.350475    0.335000
5         1.340915    1.337473    0.343000
6         1.320882    1.904698    0.357000
7         1.291946    1.368179    0.339000
8         1.283206    1.466608    0.353000
Total time: 19:50
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4__rnd_75.m
Loss and accuracy using (cls_best): [1.4704828, tensor(0.1248)]
OrderedDict([('data/mldoc/de-10/models/sp15k/qrnn_nl4__rnd_75.m',
              0.12475000321865082)])
    noise  accuracy
0    0.00   0.96125
1    0.05   0.95600
2    0.10   0.95100
3    0.15   0.94450
4    0.20   0.94375
5    0.25   0.91375
6    0.30   0.92150
7    0.35   0.91050
8    0.40   0.85825
9    0.45   0.86350
10   0.50   0.71025
11   0.55   0.65025
12   0.60   0.55800
13   0.65   0.44675
14   0.70   0.26925
15   0.75   0.12475
```