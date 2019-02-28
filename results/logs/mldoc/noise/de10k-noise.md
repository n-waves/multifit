# Correct Val data
```
python -m ulmfit eval_noise_resistance --lang=de --size=10 --prefix-name="val_"
Noise:  0
Processing data/mldoc/de-1/models/sp15k/qrnn_nl4.m
de-10
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4_val_0.m
Evaluating previously trained model
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/de.dev.csv
Data lm, trn: 13500, val: 1500
Data cls, trn: 10000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁.', '▁,', '▁der', '▁die', 'en', '▁und', 's', '▁in', 'er', "▁&'"]
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
Loss and accuracy using (cls_last): [0.3033199, tensor(0.9712)]
OrderedDict([('data/mldoc/de-10/models/sp15k/qrnn_nl4_val_0.m',
              0.9712499976158142)])
Noise:  5
Processing data/mldoc/de-1/models/sp15k/qrnn_nl4.m
de-10
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4_val_5.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/de.dev.csv
Added noise to 500 examples, only 0.95 have correct labels
Added noise to 50 examples, only 0.95 have correct labels
Data lm, trn: 13500, val: 1500
Running tokenization clsnoise0.05tv...
Data clsnoise0.05tv, trn: 10000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁.', '▁,', '▁der', '▁die', 'en', '▁und', 's', '▁in', 'er', "▁&'"]
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4_val_5.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.436501    0.416959    0.909000
2         0.460737    0.426361    0.904000
3         0.360029    0.502090    0.907000
4         0.362643    0.404165    0.916000
5         0.283832    0.478411    0.911000
6         0.224228    0.538951    0.915000
7         0.159160    0.662829    0.909000
8         0.096864    0.699768    0.910000
Total time: 18:20
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4_val_5.m
Loss and accuracy using (cls_best): [0.22873034, tensor(0.9565)]
OrderedDict([('data/mldoc/de-10/models/sp15k/qrnn_nl4_val_5.m',
              0.9564999938011169)])
Noise:  10
Processing data/mldoc/de-1/models/sp15k/qrnn_nl4.m
de-10
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4_val_10.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/de.dev.csv
Added noise to 1000 examples, only 0.9 have correct labels
Added noise to 100 examples, only 0.9 have correct labels
Data lm, trn: 13500, val: 1500
Running tokenization clsnoise0.1tv...
Data clsnoise0.1tv, trn: 10000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁.', '▁,', '▁der', '▁die', 'en', '▁und', 's', '▁in', 'er', "▁&'"]
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4_val_10.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.608604    0.614120    0.851000
2         0.563473    0.521960    0.858000
3         0.534108    0.564487    0.867000
4         0.473966    0.600789    0.870000
5         0.450963    0.579453    0.869000
6         0.365191    0.641545    0.864000
7         0.282259    0.785328    0.850000
8         0.227324    0.844709    0.848000
Total time: 18:56
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4_val_10.m
Loss and accuracy using (cls_best): [0.24939896, tensor(0.9435)]
OrderedDict([('data/mldoc/de-10/models/sp15k/qrnn_nl4_val_10.m',
              0.9434999823570251)])
Noise:  15
Processing data/mldoc/de-1/models/sp15k/qrnn_nl4.m
de-10
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4_val_15.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/de.dev.csv
Added noise to 1500 examples, only 0.85 have correct labels
Added noise to 150 examples, only 0.85 have correct labels
Data lm, trn: 13500, val: 1500
Running tokenization clsnoise0.15tv...
Data clsnoise0.15tv, trn: 10000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁.', '▁,', '▁der', '▁die', 'en', '▁und', 's', '▁in', 'er', "▁&'"]
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4_val_15.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.679152    0.757799    0.816000
2         0.715950    0.683697    0.810000
3         0.694975    0.646247    0.817000
4         0.629321    0.658788    0.818000
5         0.588069    0.728769    0.805000
6         0.517607    0.815471    0.796000
7         0.402233    0.961845    0.781000
8         0.318577    0.961705    0.779000
Total time: 18:38
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4_val_15.m
Loss and accuracy using (cls_best): [0.6598115, tensor(0.9300)]
OrderedDict([('data/mldoc/de-10/models/sp15k/qrnn_nl4_val_15.m',
              0.9300000071525574)])
Noise:  20
Processing data/mldoc/de-1/models/sp15k/qrnn_nl4.m
de-10
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4_val_20.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/de.dev.csv
Added noise to 2000 examples, only 0.8 have correct labels
Added noise to 200 examples, only 0.8 have correct labels
Data lm, trn: 13500, val: 1500
Running tokenization clsnoise0.2tv...
Data clsnoise0.2tv, trn: 10000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁.', '▁,', '▁der', '▁die', 'en', '▁und', 's', '▁in', 'er', "▁&'"]
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4_val_20.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.814279    0.829818    0.755000
2         0.824145    0.879503    0.763000
3         0.778472    0.853143    0.766000
4         0.782468    0.779830    0.762000
5         0.700055    0.840979    0.766000
6         0.600067    0.887506    0.748000
7         0.477300    1.100056    0.734000
8         0.372687    1.117019    0.729000
Total time: 18:50
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4_val_20.m
Loss and accuracy using (cls_best): [0.32372993, tensor(0.9053)]
OrderedDict([('data/mldoc/de-10/models/sp15k/qrnn_nl4_val_20.m',
              0.9052500128746033)])
Noise:  25
Processing data/mldoc/de-1/models/sp15k/qrnn_nl4.m
de-10
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4_val_25.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/de.dev.csv
Added noise to 2500 examples, only 0.75 have correct labels
Added noise to 250 examples, only 0.75 have correct labels
Data lm, trn: 13500, val: 1500
Running tokenization clsnoise0.25tv...
Data clsnoise0.25tv, trn: 10000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁.', '▁,', '▁der', '▁die', 'en', '▁und', 's', '▁in', 'er', "▁&'"]
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4_val_25.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.930055    0.914462    0.709000
2         0.875372    0.947374    0.723000
3         0.903772    0.995556    0.716000
4         0.830826    0.843742    0.727000
5         0.761732    0.948633    0.719000
6         0.652835    1.037753    0.698000
7         0.522116    1.141807    0.687000
8         0.456265    1.270369    0.675000
Total time: 18:53
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4_val_25.m
Loss and accuracy using (cls_best): [0.42412135, tensor(0.8655)]
OrderedDict([('data/mldoc/de-10/models/sp15k/qrnn_nl4_val_25.m',
              0.8654999732971191)])
Noise:  30
Processing data/mldoc/de-1/models/sp15k/qrnn_nl4.m
de-10
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4_val_30.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/de.dev.csv
Added noise to 3000 examples, only 0.7 have correct labels
Added noise to 300 examples, only 0.7 have correct labels
Data lm, trn: 13500, val: 1500
Running tokenization clsnoise0.3tv...
Data clsnoise0.3tv, trn: 10000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁.', '▁,', '▁der', '▁die', 'en', '▁und', 's', '▁in', 'er', "▁&'"]
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4_val_30.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.979780    0.992838    0.647000
2         0.964802    0.975392    0.650000
3         0.984818    0.938703    0.664000
4         0.947048    0.971088    0.667000
5         0.835190    0.978763    0.651000
6         0.779068    1.057726    0.654000
7         0.640580    1.204921    0.617000
8         0.581258    1.248827    0.613000
Total time: 18:57
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4_val_30.m
Loss and accuracy using (cls_best): [0.48816764, tensor(0.8470)]
OrderedDict([('data/mldoc/de-10/models/sp15k/qrnn_nl4_val_30.m',
              0.847000002861023)])
Noise:  35
Processing data/mldoc/de-1/models/sp15k/qrnn_nl4.m
de-10
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4_val_35.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/de.dev.csv
Added noise to 3500 examples, only 0.65 have correct labels
Added noise to 350 examples, only 0.65 have correct labels
Data lm, trn: 13500, val: 1500
Running tokenization clsnoise0.35tv...
Data clsnoise0.35tv, trn: 10000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁.', '▁,', '▁der', '▁die', 'en', '▁und', 's', '▁in', 'er', "▁&'"]
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4_val_35.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.055006    0.998555    0.620000
2         1.044137    0.994613    0.631000
3         1.002715    1.070116    0.624000
4         0.986447    1.054147    0.621000
5         0.967057    1.016487    0.622000
6         0.833950    1.251236    0.580000
7         0.717729    1.236029    0.571000
8         0.628723    1.342408    0.559000
Total time: 19:03
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4_val_35.m
Loss and accuracy using (cls_best): [0.6226422, tensor(0.7990)]
OrderedDict([('data/mldoc/de-10/models/sp15k/qrnn_nl4_val_35.m',
              0.7990000247955322)])
Noise:  40
Processing data/mldoc/de-1/models/sp15k/qrnn_nl4.m
de-10
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4_val_40.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/de.dev.csv
Added noise to 4000 examples, only 0.6 have correct labels
Added noise to 400 examples, only 0.6 have correct labels
Data lm, trn: 13500, val: 1500
Running tokenization clsnoise0.4tv...
Data clsnoise0.4tv, trn: 10000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁.', '▁,', '▁der', '▁die', 'en', '▁und', 's', '▁in', 'er', "▁&'"]
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4_val_40.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.103033    1.086538    0.570000
2         1.107361    1.056345    0.569000
3         1.081129    1.063769    0.548000
4         1.095896    1.061330    0.556000
5         0.991429    1.102777    0.569000
6         0.915104    1.178768    0.525000
7         0.779928    1.327309    0.513000
8         0.686467    1.426985    0.502000
Total time: 18:50
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4_val_40.m
Loss and accuracy using (cls_best): [0.8506142, tensor(0.7530)]
OrderedDict([('data/mldoc/de-10/models/sp15k/qrnn_nl4_val_40.m',
              0.753000020980835)])
Noise:  45
Processing data/mldoc/de-1/models/sp15k/qrnn_nl4.m
de-10
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4_val_45.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/de.dev.csv
Added noise to 4500 examples, only 0.55 have correct labels
Added noise to 450 examples, only 0.55 have correct labels
Data lm, trn: 13500, val: 1500
Running tokenization clsnoise0.45tv...
Data clsnoise0.45tv, trn: 10000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁.', '▁,', '▁der', '▁die', 'en', '▁und', 's', '▁in', 'er', "▁&'"]
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4_val_45.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.140645    1.110783    0.512000
2         1.136563    1.121161    0.535000
3         1.129539    1.099606    0.509000
4         1.108058    1.073747    0.524000
5         1.048675    1.126527    0.504000
6         0.956287    1.194883    0.479000
7         0.818828    1.354711    0.461000
8         0.758420    1.486676    0.448000
Total time: 19:12
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4_val_45.m
Loss and accuracy using (cls_best): [0.82442254, tensor(0.7000)]
OrderedDict([('data/mldoc/de-10/models/sp15k/qrnn_nl4_val_45.m',
              0.699999988079071)])
Noise:  50
Processing data/mldoc/de-1/models/sp15k/qrnn_nl4.m
de-10
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4_val_50.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/de.dev.csv
Added noise to 5000 examples, only 0.5 have correct labels
Added noise to 500 examples, only 0.5 have correct labels
Data lm, trn: 13500, val: 1500
Running tokenization clsnoise0.5tv...
Data clsnoise0.5tv, trn: 10000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁.', '▁,', '▁der', '▁die', 'en', '▁und', 's', '▁in', 'er', "▁&'"]
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4_val_50.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.179953    1.145395    0.438000
2         1.165212    1.175898    0.483000
3         1.181145    1.127925    0.477000
4         1.139963    1.151791    0.471000
5         1.124215    1.130031    0.466000
6         1.029675    1.185899    0.450000
7         0.911210    1.320619    0.427000
8         0.838111    1.383757    0.419000
Total time: 18:22
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4_val_50.m
Loss and accuracy using (cls_best): [0.9008743, tensor(0.6545)]
OrderedDict([('data/mldoc/de-10/models/sp15k/qrnn_nl4_val_50.m',
              0.6545000076293945)])
Noise:  55
Processing data/mldoc/de-1/models/sp15k/qrnn_nl4.m
de-10
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4_val_55.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/de.dev.csv
Added noise to 5500 examples, only 0.45 have correct labels
Added noise to 550 examples, only 0.45 have correct labels
Data lm, trn: 13500, val: 1500
Running tokenization clsnoise0.55tv...
Data clsnoise0.55tv, trn: 10000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁.', '▁,', '▁der', '▁die', 'en', '▁und', 's', '▁in', 'er', "▁&'"]
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4_val_55.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.183246    1.166291    0.448000
2         1.196840    1.163279    0.404000
3         1.193226    1.180232    0.407000
4         1.183507    1.153738    0.425000
5         1.164359    1.166123    0.396000
6         1.095628    1.207745    0.387000
7         0.982455    1.293190    0.383000
8         0.911601    1.383209    0.369000
Total time: 19:04
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4_val_55.m
Loss and accuracy using (cls_best): [1.0093645, tensor(0.6365)]
OrderedDict([('data/mldoc/de-10/models/sp15k/qrnn_nl4_val_55.m',
              0.6365000009536743)])
Noise:  60
Processing data/mldoc/de-1/models/sp15k/qrnn_nl4.m
de-10
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4_val_60.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/de.dev.csv
Added noise to 6000 examples, only 0.4 have correct labels
Added noise to 600 examples, only 0.4 have correct labels
Data lm, trn: 13500, val: 1500
Running tokenization clsnoise0.6tv...
Data clsnoise0.6tv, trn: 10000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁.', '▁,', '▁der', '▁die', 'en', '▁und', 's', '▁in', 'er', "▁&'"]
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4_val_60.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.234308    1.216432    0.370000
2         1.218607    1.166292    0.360000
3         1.206626    1.179762    0.335000
4         1.192636    1.164426    0.332000
5         1.167782    1.226321    0.362000
6         1.118646    1.225461    0.360000
7         1.057706    1.246721    0.358000
8         1.011610    1.281406    0.346000
Total time: 18:43
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4_val_60.m
Loss and accuracy using (cls_best): [1.0831424, tensor(0.4775)]
OrderedDict([('data/mldoc/de-10/models/sp15k/qrnn_nl4_val_60.m',
              0.47749999165534973)])
Noise:  65
Processing data/mldoc/de-1/models/sp15k/qrnn_nl4.m
de-10
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4_val_65.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/de.dev.csv
Added noise to 6500 examples, only 0.35 have correct labels
Added noise to 650 examples, only 0.35 have correct labels
Data lm, trn: 13500, val: 1500
Running tokenization clsnoise0.65tv...
Data clsnoise0.65tv, trn: 10000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁.', '▁,', '▁der', '▁die', 'en', '▁und', 's', '▁in', 'er', "▁&'"]
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4_val_65.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.215925    1.200209    0.326000
2         1.235970    1.173920    0.313000
3         1.218506    1.251267    0.310000
4         1.204741    1.182633    0.347000
5         1.173637    1.171351    0.333000
6         1.109750    1.226862    0.329000
7         1.054864    1.330735    0.309000
8         1.002966    1.325511    0.323000
Total time: 18:20
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4_val_65.m
Loss and accuracy using (cls_best): [1.2715616, tensor(0.3695)]
OrderedDict([('data/mldoc/de-10/models/sp15k/qrnn_nl4_val_65.m',
              0.3695000112056732)])
Noise:  70
Processing data/mldoc/de-1/models/sp15k/qrnn_nl4.m
de-10
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4_val_70.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/de.dev.csv
Added noise to 7000 examples, only 0.3 have correct labels
Added noise to 700 examples, only 0.3 have correct labels
Data lm, trn: 13500, val: 1500
Running tokenization clsnoise0.7tv...
Data clsnoise0.7tv, trn: 10000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁.', '▁,', '▁der', '▁die', 'en', '▁und', 's', '▁in', 'er', "▁&'"]
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4_val_70.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.217860    1.219061    0.351000
2         1.222316    1.191306    0.332000
3         1.222368    1.254068    0.343000
4         1.208153    1.216313    0.339000
5         1.180702    1.166293    0.336000
6         1.128917    1.211504    0.333000
7         1.080312    1.271096    0.337000
8         1.017895    1.335092    0.339000
Total time: 18:47
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4_val_70.m
Loss and accuracy using (cls_best): [1.4655445, tensor(0.2688)]
OrderedDict([('data/mldoc/de-10/models/sp15k/qrnn_nl4_val_70.m',
              0.26875001192092896)])
Noise:  75
Processing data/mldoc/de-1/models/sp15k/qrnn_nl4.m
de-10
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4_val_75.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/de.dev.csv
Added noise to 7500 examples, only 0.25 have correct labels
Added noise to 750 examples, only 0.25 have correct labels
Data lm, trn: 13500, val: 1500
Running tokenization clsnoise0.75tv...
Data clsnoise0.75tv, trn: 10000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁.', '▁,', '▁der', '▁die', 'en', '▁und', 's', '▁in', 'er', "▁&'"]
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4_val_75.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.203315    1.147710    0.380000
2         1.212273    1.190271    0.333000
3         1.200970    1.208472    0.367000
4         1.169665    1.165628    0.358000
5         1.166296    1.213566    0.374000
6         1.112285    1.462085    0.380000
7         1.054748    1.266731    0.389000
8         0.981918    1.692906    0.364000
Total time: 18:34
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4_val_75.m
Loss and accuracy using (cls_best): [1.957079, tensor(0.1252)]
OrderedDict([('data/mldoc/de-10/models/sp15k/qrnn_nl4_val_75.m',
              0.12524999678134918)])
{'data/mldoc/de-10/models/sp15k/qrnn_nl4_val_0.m': 0.9712499976158142, 'data/mldoc/de-10/models/sp15k/qrnn_nl4_val_5.m': 0.9564999938011169, 'data/mldoc/de-10/models/sp15k/qrnn_nl4_val_10.m': 0.9434999823570251, 'data/mldoc/de-10/models/sp15k/qrnn_nl4_val_15.m': 0.9300000071525574, 'data/mldoc/de-10/models/sp15k/qrnn_nl4_val_20.m': 0.9052500128746033, 'data/mldoc/de-10/models/sp15k/qrnn_nl4_val_25.m': 0.8654999732971191, 'data/mldoc/de-10/models/sp15k/qrnn_nl4_val_30.m': 0.847000002861023, 'data/mldoc/de-10/models/sp15k/qrnn_nl4_val_35.m': 0.7990000247955322, 'data/mldoc/de-10/models/sp15k/qrnn_nl4_val_40.m': 0.753000020980835, 'data/mldoc/de-10/models/sp15k/qrnn_nl4_val_45.m': 0.699999988079071, 'data/mldoc/de-10/models/sp15k/qrnn_nl4_val_50.m': 0.6545000076293945, 'data/mldoc/de-10/models/sp15k/qrnn_nl4_val_55.m': 0.6365000009536743, 'data/mldoc/de-10/models/sp15k/qrnn_nl4_val_60.m': 0.47749999165534973, 'data/mldoc/de-10/models/sp15k/qrnn_nl4_val_65.m': 0.3695000112056732, 'data/mldoc/de-10/models/sp15k/qrnn_nl4_val_70.m': 0.26875001192092896, 'data/mldoc/de-10/models/sp15k/qrnn_nl4_val_75.m': 0.12524999678134918}
```
# Incorrect Val data

``` python -m ulmfit eval_noise_resistance --size=10
Noise:  0
Processing data/mldoc/de-1/models/sp15k/qrnn_nl4.m
de-10
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4_0.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/de.dev.csv
Running tokenization lm...
Data lm, trn: 13500, val: 1500
Running tokenization cls...
Data cls, trn: 10000, val: 1000
Running tokenization tst...
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
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4_0.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.248545    0.179782    0.959000
2         0.189914    0.262213    0.960000
3         0.166565    0.489769    0.942000
4         0.126547    0.252591    0.956000
5         0.125812    0.265295    0.959000
6         0.052031    0.339353    0.966000
7         0.055669    0.452688    0.965000
8         0.019922    0.439593    0.963000
Total time: 18:52
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4_0.m
Loss and accuracy using (cls_best): [0.3236656, tensor(0.9720)]
OrderedDict([('data/mldoc/de-10/models/sp15k/qrnn_nl4_0.m', 0.972000002861023)])
Noise:  5
Processing data/mldoc/de-1/models/sp15k/qrnn_nl4.m
de-10
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4_5.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/de.dev.csv
Added noise to 500 examples, only 0.95 have correct labels
Data lm, trn: 13500, val: 1500
Running tokenization clsnoise0.05...
Data clsnoise0.05, trn: 10000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁.', '▁,', '▁der', '▁die', 'en', '▁und', 's', '▁in', 'er', "▁&'"]
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4_5.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.467763    0.204974    0.949000
2         0.403194    0.166174    0.957000
3         0.412454    0.176702    0.958000
4         0.337252    0.957854    0.964000
5         0.314917    0.179809    0.959000
6         0.244516    0.202331    0.959000
7         0.145377    0.225054    0.960000
8         0.129216    0.239015    0.960000
Total time: 19:15
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4_5.m
Loss and accuracy using (cls_best): [0.18567306, tensor(0.9620)]
OrderedDict([('data/mldoc/de-10/models/sp15k/qrnn_nl4_5.m',
              0.9620000123977661)])
Noise:  10
Processing data/mldoc/de-1/models/sp15k/qrnn_nl4.m
de-10
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4_10.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/de.dev.csv
Added noise to 1000 examples, only 0.9 have correct labels
Data lm, trn: 13500, val: 1500
Running tokenization clsnoise0.1...
Data clsnoise0.1, trn: 10000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁.', '▁,', '▁der', '▁die', 'en', '▁und', 's', '▁in', 'er', "▁&'"]
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4_10.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.611526    0.220041    0.950000
2         0.523582    0.209251    0.959000
3         0.566672    0.190530    0.960000
4         0.530805    0.214880    0.949000
5         0.470179    0.238882    0.954000
6         0.368767    0.224209    0.950000
7         0.276495    0.254708    0.942000
8         0.242059    0.260761    0.940000
Total time: 19:23
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4_10.m
Loss and accuracy using (cls_best): [0.19178627, tensor(0.9557)]
OrderedDict([('data/mldoc/de-10/models/sp15k/qrnn_nl4_10.m',
              0.9557499885559082)])
Noise:  15
Processing data/mldoc/de-1/models/sp15k/qrnn_nl4.m
de-10
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4_15.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/de.dev.csv
Added noise to 1500 examples, only 0.85 have correct labels
Data lm, trn: 13500, val: 1500
Running tokenization clsnoise0.15...
Data clsnoise0.15, trn: 10000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁.', '▁,', '▁der', '▁die', 'en', '▁und', 's', '▁in', 'er', "▁&'"]
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4_15.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.712348    0.224500    0.948000
2         0.702935    0.188924    0.956000
3         0.645956    0.224620    0.957000
4         0.644018    0.248982    0.952000
5         0.580399    0.255238    0.958000
6         0.497618    0.297617    0.930000
7         0.339819    0.272245    0.936000
8         0.323421    0.297615    0.926000
Total time: 18:50
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4_15.m
Loss and accuracy using (cls_best): [0.25138617, tensor(0.9350)]
OrderedDict([('data/mldoc/de-10/models/sp15k/qrnn_nl4_15.m',
              0.9350000023841858)])
Noise:  20
Processing data/mldoc/de-1/models/sp15k/qrnn_nl4.m
de-10
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4_20.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/de.dev.csv
Added noise to 2000 examples, only 0.8 have correct labels
Data lm, trn: 13500, val: 1500
Running tokenization clsnoise0.2...
Data clsnoise0.2, trn: 10000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁.', '▁,', '▁der', '▁die', 'en', '▁und', 's', '▁in', 'er', "▁&'"]
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4_20.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.821758    0.270898    0.954000
2         0.831244    0.440837    0.923000
3         0.787815    0.313425    0.962000
4         0.730238    0.255505    0.958000
5         0.748861    0.302285    0.959000
6         0.603880    0.324315    0.933000
7         0.539476    0.306479    0.931000
8         0.431666    0.315723    0.916000
Total time: 19:28
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4_20.m
Loss and accuracy using (cls_best): [0.28679955, tensor(0.9202)]
OrderedDict([('data/mldoc/de-10/models/sp15k/qrnn_nl4_20.m',
              0.9202499985694885)])
Noise:  25
Processing data/mldoc/de-1/models/sp15k/qrnn_nl4.m
de-10
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4_25.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/de.dev.csv
Added noise to 2500 examples, only 0.75 have correct labels
Data lm, trn: 13500, val: 1500
Running tokenization clsnoise0.25...
Data clsnoise0.25, trn: 10000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁.', '▁,', '▁der', '▁die', 'en', '▁und', 's', '▁in', 'er', "▁&'"]
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4_25.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.913877    0.325624    0.956000
2         0.914296    0.380646    0.956000
3         0.862887    0.400977    0.909000
4         0.859236    0.291990    0.957000
5         0.832637    0.447769    0.944000
6         0.643751    0.319626    0.935000
7         0.562953    0.444973    0.875000
8         0.471118    0.437798    0.876000
Total time: 19:29
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4_25.m
Loss and accuracy using (cls_best): [0.53222567, tensor(0.8808)]
OrderedDict([('data/mldoc/de-10/models/sp15k/qrnn_nl4_25.m',
              0.8807500004768372)])
Noise:  30
Processing data/mldoc/de-1/models/sp15k/qrnn_nl4.m
de-10
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4_30.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/de.dev.csv
Added noise to 3000 examples, only 0.7 have correct labels
Data lm, trn: 13500, val: 1500
Running tokenization clsnoise0.3...
Data clsnoise0.3, trn: 10000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁.', '▁,', '▁der', '▁die', 'en', '▁und', 's', '▁in', 'er', "▁&'"]
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4_30.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.967625    0.434347    0.921000
2         1.015967    0.491977    0.927000
3         0.943155    0.499572    0.935000
4         0.953488    0.362500    0.951000
5         0.859923    0.440134    0.944000
6         0.761165    0.879942    0.894000
7         0.662025    0.566052    0.857000
8         0.572539    0.546405    0.826000
Total time: 19:17
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4_30.m
Loss and accuracy using (cls_best): [0.47183233, tensor(0.8540)]
OrderedDict([('data/mldoc/de-10/models/sp15k/qrnn_nl4_30.m',
              0.8539999723434448)])
Noise:  35
Processing data/mldoc/de-1/models/sp15k/qrnn_nl4.m
de-10
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4_35.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/de.dev.csv
Added noise to 3500 examples, only 0.65 have correct labels
Data lm, trn: 13500, val: 1500
Running tokenization clsnoise0.35...
Data clsnoise0.35, trn: 10000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁.', '▁,', '▁der', '▁die', 'en', '▁und', 's', '▁in', 'er', "▁&'"]
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4_35.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.079700    0.486082    0.944000
2         1.032614    0.470132    0.943000
3         1.007776    0.460361    0.948000
4         1.018879    0.476889    0.909000
5         0.943363    1.618245    0.871000
6         0.897375    0.604067    0.844000
7         0.731791    10.002838   0.829000
8         0.631117    6.527773    0.809000
Total time: 19:26
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4_35.m
Loss and accuracy using (cls_best): [0.79261243, tensor(0.8185)]
OrderedDict([('data/mldoc/de-10/models/sp15k/qrnn_nl4_35.m',
              0.8184999823570251)])
Noise:  40
Processing data/mldoc/de-1/models/sp15k/qrnn_nl4.m
de-10
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4_40.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/de.dev.csv
Added noise to 4000 examples, only 0.6 have correct labels
Data lm, trn: 13500, val: 1500
Running tokenization clsnoise0.4...
Data clsnoise0.4, trn: 10000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁.', '▁,', '▁der', '▁die', 'en', '▁und', 's', '▁in', 'er', "▁&'"]
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4_40.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.105347    0.646775    0.915000
2         1.121720    0.547566    0.943000
3         1.084002    0.581386    0.949000
4         1.058243    0.486947    0.953000
5         1.020843    1.061957    0.945000
6         0.944650    0.561371    0.872000
7         0.852374    2.072736    0.815000
8         0.745296    0.637481    0.807000
Total time: 18:42
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4_40.m
Loss and accuracy using (cls_best): [0.6664878, tensor(0.7915)]
OrderedDict([('data/mldoc/de-10/models/sp15k/qrnn_nl4_40.m',
              0.7914999723434448)])
Noise:  45
Processing data/mldoc/de-1/models/sp15k/qrnn_nl4.m
de-10
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4_45.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/de.dev.csv
Added noise to 4500 examples, only 0.55 have correct labels
Data lm, trn: 13500, val: 1500
Running tokenization clsnoise0.45...
Data clsnoise0.45, trn: 10000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁.', '▁,', '▁der', '▁die', 'en', '▁und', 's', '▁in', 'er', "▁&'"]
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4_45.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.149571    0.715101    0.835000
2         1.130510    0.812303    0.671000
3         1.137401    0.566132    0.938000
4         1.124969    0.654854    0.944000
5         1.066340    0.624603    0.928000
6         1.000705    0.700255    0.836000
7         0.895356    0.726855    0.784000
8         0.813315    0.740214    0.747000
Total time: 19:12
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4_45.m
Loss and accuracy using (cls_best): [0.74985105, tensor(0.7402)]
OrderedDict([('data/mldoc/de-10/models/sp15k/qrnn_nl4_45.m',
              0.7402499914169312)])
Noise:  50
Processing data/mldoc/de-1/models/sp15k/qrnn_nl4.m
de-10
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4_50.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/de.dev.csv
Added noise to 5000 examples, only 0.5 have correct labels
Data lm, trn: 13500, val: 1500
Running tokenization clsnoise0.5...
Data clsnoise0.5, trn: 10000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁.', '▁,', '▁der', '▁die', 'en', '▁und', 's', '▁in', 'er', "▁&'"]
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4_50.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.176207    0.770225    0.894000
2         1.146716    0.698875    0.916000
3         1.159391    0.808971    0.908000
4         1.148178    0.817309    0.715000
5         1.142510    0.755769    0.872000
6         1.068410    0.837932    0.839000
7         0.985223    1.619010    0.714000
8         0.900792    0.850288    0.672000
Total time: 18:43
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4_50.m
Loss and accuracy using (cls_best): [0.87151223, tensor(0.6888)]
OrderedDict([('data/mldoc/de-10/models/sp15k/qrnn_nl4_50.m',
              0.6887500286102295)])
Noise:  55
Processing data/mldoc/de-1/models/sp15k/qrnn_nl4.m
de-10
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4_55.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/de.dev.csv
Added noise to 5500 examples, only 0.45 have correct labels
Data lm, trn: 13500, val: 1500
Running tokenization clsnoise0.55...
Data clsnoise0.55, trn: 10000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁.', '▁,', '▁der', '▁die', 'en', '▁und', 's', '▁in', 'er', "▁&'"]
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4_55.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.194526    0.829596    0.896000
2         1.206947    0.800475    0.856000
3         1.205839    0.832071    0.895000
4         1.193005    8.321078    0.884000
5         1.135684    3.162405    0.759000
6         1.087508    0.866630    0.751000
7         0.969481    0.909597    0.654000
8         0.903410    1.040992    0.612000
Total time: 18:41
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4_55.m
Loss and accuracy using (cls_best): [0.944759, tensor(0.6118)]
OrderedDict([('data/mldoc/de-10/models/sp15k/qrnn_nl4_55.m',
              0.6117500066757202)])
Noise:  60
Processing data/mldoc/de-1/models/sp15k/qrnn_nl4.m
de-10
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4_60.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/de.dev.csv
Added noise to 6000 examples, only 0.4 have correct labels
Data lm, trn: 13500, val: 1500
Running tokenization clsnoise0.6...
Data clsnoise0.6, trn: 10000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁.', '▁,', '▁der', '▁die', 'en', '▁und', 's', '▁in', 'er', "▁&'"]
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4_60.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.228728    1.235617    0.451000
2         1.231521    1.103163    0.465000
3         1.205319    1.067342    0.535000
4         1.208439    1.171245    0.738000
5         1.166922    1.071724    0.481000
6         1.105134    0.913977    0.680000
7         0.996566    1.203313    0.546000
8         0.933625    1.092093    0.506000
Total time: 18:30
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4_60.m
Loss and accuracy using (cls_best): [1.0928471, tensor(0.5045)]
OrderedDict([('data/mldoc/de-10/models/sp15k/qrnn_nl4_60.m',
              0.5044999718666077)])
Noise:  65
Processing data/mldoc/de-1/models/sp15k/qrnn_nl4.m
de-10
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4_65.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/de.dev.csv
Added noise to 6500 examples, only 0.35 have correct labels
Data lm, trn: 13500, val: 1500
Running tokenization clsnoise0.65...
Data clsnoise0.65, trn: 10000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁.', '▁,', '▁der', '▁die', 'en', '▁und', 's', '▁in', 'er', "▁&'"]
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4_65.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.233954    1.126451    0.312000
2         1.228535    1.022103    0.514000
3         1.223979    1.275958    0.040000
4         1.201343    1.281353    0.233000
5         1.174795    1.088655    0.714000
6         1.137927    1.304352    0.430000
7         1.061154    2.778949    0.450000
8         0.992379    1.507618    0.409000
Total time: 18:44
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4_65.m
Loss and accuracy using (cls_best): [1.7004116, tensor(0.4027)]
OrderedDict([('data/mldoc/de-10/models/sp15k/qrnn_nl4_65.m',
              0.4027499854564667)])
Noise:  70
Processing data/mldoc/de-1/models/sp15k/qrnn_nl4.m
de-10
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4_70.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/de.dev.csv
Added noise to 7000 examples, only 0.3 have correct labels
Data lm, trn: 13500, val: 1500
Running tokenization clsnoise0.7...
Data clsnoise0.7, trn: 10000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁.', '▁,', '▁der', '▁die', 'en', '▁und', 's', '▁in', 'er', "▁&'"]
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4_70.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.212556    1.284562    0.230000
2         1.221429    1.494182    0.025000
3         1.208761    1.601475    0.044000
4         1.196318    1.222757    0.493000
5         1.170390    1.235789    0.240000
6         1.114086    1.487270    0.244000
7         1.054812    1.470409    0.244000
8         0.998996    1.476172    0.270000
Total time: 19:12
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4_70.m
Loss and accuracy using (cls_best): [1.4951355, tensor(0.2860)]
OrderedDict([('data/mldoc/de-10/models/sp15k/qrnn_nl4_70.m',
              0.28600001335144043)])
Noise:  75
Processing data/mldoc/de-1/models/sp15k/qrnn_nl4.m
de-10
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4_75.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/de.dev.csv
Added noise to 7500 examples, only 0.25 have correct labels
Data lm, trn: 13500, val: 1500
Running tokenization clsnoise0.75...
Data clsnoise0.75, trn: 10000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁.', '▁,', '▁der', '▁die', 'en', '▁und', 's', '▁in', 'er', "▁&'"]
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4_75.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.206901    1.635707    0.035000
2         1.226015    1.508531    0.222000
3         1.203663    1.570317    0.020000
4         1.201733    1.404657    0.054000
5         1.161620    1.456142    0.040000
6         1.142974    1.518606    0.037000
7         1.067800    1.744556    0.098000
8         0.983552    1.720158    0.117000
Total time: 18:41
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-10/models/sp15k/qrnn_nl4_75.m
Loss and accuracy using (cls_best): [1.7742158, tensor(0.1138)]
OrderedDict([('data/mldoc/de-10/models/sp15k/qrnn_nl4_75.m',
              0.11375000327825546)])
{'data/mldoc/de-10/models/sp15k/qrnn_nl4_0.m': 0.972000002861023, 'data/mldoc/de-10/models/sp15k/qrnn_nl4_5.m': 0.9620000123977661, 'data/mldoc/de-10/models/sp15k/qrnn_nl4_10.m': 0.9557499885559082, 'data/mldoc/de-10/models/sp15k/qrnn_nl4_15.m': 0.9350000023841858, 'data/mldoc/de-10/models/sp15k/qrnn_nl4_20.m': 0.9202499985694885, 'data/mldoc/de-10/models/sp15k/qrnn_nl4_25.m': 0.8807500004768372, 'data/mldoc/de-10/models/sp15k/qrnn_nl4_30.m': 0.8539999723434448, 'data/mldoc/de-10/models/sp15k/qrnn_nl4_35.m': 0.8184999823570251, 'data/mldoc/de-10/models/sp15k/qrnn_nl4_40.m': 0.7914999723434448, 'data/mldoc/de-10/models/sp15k/qrnn_nl4_45.m': 0.7402499914169312, 'data/mldoc/de-10/models/sp15k/qrnn_nl4_50.m': 0.6887500286102295, 'data/mldoc/de-10/models/sp15k/qrnn_nl4_55.m': 0.6117500066757202, 'data/mldoc/de-10/models/sp15k/qrnn_nl4_60.m': 0.5044999718666077, 'data/mldoc/de-10/models/sp15k/qrnn_nl4_65.m': 0.4027499854564667, 'data/mldoc/de-10/models/sp15k/qrnn_nl4_70.m': 0.28600001335144043, 'data/mldoc/de-10/models/sp15k/qrnn_nl4_75.m': 0.11375000327825546}
```