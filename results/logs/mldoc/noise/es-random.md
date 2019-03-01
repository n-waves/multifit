# Noise resistance on ES Random 
## 1k



## 10k
```
 python -m ulmfit eval_noise_resistance --lang=es --size=10 --prefix-name="rnd2_" --model="sp15k/qrnn_rnd-nl4.m" --label-smoothing-eps=0.1 --random-init=True
Noise:  0
Processing data/mldoc/es-1/models/sp15k/qrnn_rnd-nl4.m
es-10
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp15k/qrnn_nl4_rnd2_0.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/es.dev.csv
Data lm, trn: 13013, val: 1445
Data cls, trn: 9458, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁la', '▁.', 's', '▁en', '▁el', '▁y', '▁a', '▁que']
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
Starting classifier from random weights
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.674091    1.283792    0.574000
2         0.624245    11.202841   0.519000
3         0.634349    3.687749    0.516000
4         0.596913    4.074763    0.608000
5         0.594516    1.093833    0.715000
6         0.557703    1.010482    0.694000
7         0.543755    0.995253    0.702000
8         0.537600    0.952028    0.754000
Total time: 11:03
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp15k/qrnn_nl4_rnd2_0.m
Loss and accuracy using (cls_best): [0.9988524, tensor(0.7690)]
OrderedDict([('data/mldoc/es-10/models/sp15k/qrnn_nl4_rnd2_0.m',
              0.7689999938011169)])
              python -m ulmfit eval_noise_resistance --lang=es --size=10 --prefix-name="rnd2_" --model="sp15k/qrnn_rnd-nl4.m" --label-smoothing-eps=0.1 --random-init=True
Noise:  0
Processing data/mldoc/es-1/models/sp15k/qrnn_rnd-nl4.m
es-10
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp15k/qrnn_nl4_rnd2_0.m
Evaluating previously trained model
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/es.dev.csv
Data lm, trn: 13013, val: 1445
Data cls, trn: 9458, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁la', '▁.', 's', '▁en', '▁el', '▁y', '▁a', '▁que']
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
Loss and accuracy using (cls_best): [0.99926454, tensor(0.7695)]
OrderedDict([('data/mldoc/es-10/models/sp15k/qrnn_nl4_rnd2_0.m',
              0.7695000171661377)])
Noise:  5
Processing data/mldoc/es-1/models/sp15k/qrnn_rnd-nl4.m
es-10
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp15k/qrnn_nl4_rnd2_5.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/es.dev.csv
Added noise to 472 examples, only 0.9500951575385916 have correct labels
Added noise to 50 examples, only 0.95 have correct labels
Data lm, trn: 13013, val: 1445
Data clsnoise0.05tv, trn: 9458, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁la', '▁.', 's', '▁en', '▁el', '▁y', '▁a', '▁que']
Starting classifier from random weights
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.743423    1.354918    0.428000
2         0.729408    1.153078    0.543000
3         0.736531    1.465756    0.450000
4         0.749902    4.479099    0.541000
5         0.691128    1.322383    0.466000
6         0.694896    1.274096    0.597000
7         0.664847    1.020145    0.673000
8         0.645594    1.027490    0.668000
Total time: 10:56
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp15k/qrnn_nl4_rnd2_5.m
Loss and accuracy using (cls_best): [0.68077123, tensor(0.7272)]
OrderedDict([('data/mldoc/es-10/models/sp15k/qrnn_nl4_rnd2_5.m',
              0.7272499799728394)])
Noise:  10
Processing data/mldoc/es-1/models/sp15k/qrnn_rnd-nl4.m
es-10
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp15k/qrnn_nl4_rnd2_10.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/es.dev.csv
Added noise to 945 examples, only 0.9000845844787482 have correct labels
Added noise to 100 examples, only 0.9 have correct labels
Data lm, trn: 13013, val: 1445
Data clsnoise0.1tv, trn: 9458, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁la', '▁.', 's', '▁en', '▁el', '▁y', '▁a', '▁que']
Starting classifier from random weights
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.834923    1.409279    0.485000
2         0.854848    1.614973    0.320000
3         0.828790    1.701499    0.396000
4         0.803852    2.424411    0.547000
5         0.801694    2.024830    0.562000
6         0.786992    2.021667    0.609000
7         0.765137    2.020704    0.652000
8         0.746047    5.385053    0.652000
Total time: 10:56
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp15k/qrnn_nl4_rnd2_10.m
Loss and accuracy using (cls_best): [0.80370593, tensor(0.7172)]
OrderedDict([('data/mldoc/es-10/models/sp15k/qrnn_nl4_rnd2_10.m',
              0.7172499895095825)])
Noise:  15
Processing data/mldoc/es-1/models/sp15k/qrnn_rnd-nl4.m
es-10
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp15k/qrnn_nl4_rnd2_15.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/es.dev.csv
Added noise to 1418 examples, only 0.8500740114189046 have correct labels
Added noise to 150 examples, only 0.85 have correct labels
Data lm, trn: 13013, val: 1445
Data clsnoise0.15tv, trn: 9458, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁la', '▁.', 's', '▁en', '▁el', '▁y', '▁a', '▁que']
Starting classifier from random weights
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.885186    1.486315    0.448000
2         0.922848    1.944151    0.485000
3         0.919082    1.599238    0.400000
4         0.895376    1.527287    0.451000
5         0.881112    3.831651    0.478000
6         0.919497    1.247746    0.508000
7         0.829800    1.175510    0.508000
8         0.861669    1.174659    0.504000
Total time: 10:35
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp15k/qrnn_nl4_rnd2_15.m
Loss and accuracy using (cls_best): [0.9063169, tensor(0.5955)]
OrderedDict([('data/mldoc/es-10/models/sp15k/qrnn_nl4_rnd2_15.m',
              0.5954999923706055)])
Noise:  20
Processing data/mldoc/es-1/models/sp15k/qrnn_rnd-nl4.m
es-10
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp15k/qrnn_nl4_rnd2_20.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/es.dev.csv
Added noise to 1891 examples, only 0.8000634383590611 have correct labels
Added noise to 200 examples, only 0.8 have correct labels
Data lm, trn: 13013, val: 1445
Data clsnoise0.2tv, trn: 9458, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁la', '▁.', 's', '▁en', '▁el', '▁y', '▁a', '▁que']
Starting classifier from random weights
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.000344    1.388730    0.335000
2         1.007181    12.045410   0.472000
3         0.983002    1.479323    0.371000
4         0.958265    1.298010    0.400000
5         0.939612    4.368020    0.414000
6         0.928165    3.608808    0.492000
7         0.926556    2.253188    0.471000
8         0.905125    2.181204    0.499000
Total time: 11:06
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp15k/qrnn_nl4_rnd2_20.m
Loss and accuracy using (cls_best): [0.9292287, tensor(0.5985)]
OrderedDict([('data/mldoc/es-10/models/sp15k/qrnn_nl4_rnd2_20.m',
              0.5985000133514404)])
Noise:  25
Processing data/mldoc/es-1/models/sp15k/qrnn_rnd-nl4.m
es-10
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp15k/qrnn_nl4_rnd2_25.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/es.dev.csv
Added noise to 2364 examples, only 0.7500528652992176 have correct labels
Added noise to 250 examples, only 0.75 have correct labels
Data lm, trn: 13013, val: 1445
Data clsnoise0.25tv, trn: 9458, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁la', '▁.', 's', '▁en', '▁el', '▁y', '▁a', '▁que']
Starting classifier from random weights
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.057209    1.246537    0.466000
2         1.038750    34.810867   0.114000
3         1.030894    87.796410   0.376000
4         1.041276    1.264513    0.462000
5         1.029240    2.345500    0.422000
6         1.000014    1.293007    0.464000
7         0.960223    1.266377    0.466000
8         0.942258    1.348627    0.447000
Total time: 10:44
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp15k/qrnn_nl4_rnd2_25.m
Loss and accuracy using (cls_best): [1.0587388, tensor(0.5707)]
OrderedDict([('data/mldoc/es-10/models/sp15k/qrnn_nl4_rnd2_25.m',
              0.5707499980926514)])
Noise:  30
Processing data/mldoc/es-1/models/sp15k/qrnn_rnd-nl4.m
es-10
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp15k/qrnn_nl4_rnd2_30.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/es.dev.csv
Added noise to 2837 examples, only 0.7000422922393741 have correct labels
Added noise to 300 examples, only 0.7 have correct labels
Data lm, trn: 13013, val: 1445
Data clsnoise0.3tv, trn: 9458, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁la', '▁.', 's', '▁en', '▁el', '▁y', '▁a', '▁que']
Starting classifier from random weights
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.087622    1.363222    0.435000
2         1.095437    4.608276    0.221000
3         1.094145    1.402207    0.371000
4         1.078363    2.236299    0.388000
5         1.056162    1.310936    0.431000
python -m ulmfit eval_noise_resistance --lang=es --size=10 --prefix-name="rnd2_" --model="sp15k/qrnn_rnd-nl4.m" --label-smoo6         1.050225    1.267474    0.425000
7         1.038286    1.299926    0.421000
8         1.044985    1.300490    0.422000
Total time: 10:41
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp15k/qrnn_nl4_rnd2_30.m
Loss and accuracy using (cls_best): [0.9783086, tensor(0.5817)]
OrderedDict([('data/mldoc/es-10/models/sp15k/qrnn_nl4_rnd2_30.m',
              0.5817499756813049)])
Noise:  35
Processing data/mldoc/es-1/models/sp15k/qrnn_rnd-nl4.m
es-10
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp15k/qrnn_nl4_rnd2_35.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/es.dev.csv
Added noise to 3310 examples, only 0.6500317191795305 have correct labels
Added noise to 350 examples, only 0.65 have correct labels
Data lm, trn: 13013, val: 1445
Data clsnoise0.35tv, trn: 9458, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁la', '▁.', 's', '▁en', '▁el', '▁y', '▁a', '▁que']
Starting classifier from random weights
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.153989    1.421947    0.240000
2         1.138209    1.703624    0.246000
3         1.124318    112.371529  0.280000
4         1.133281    1.417321    0.317000
5         1.138515    1.317285    0.387000
6         1.101543    1.366418    0.390000
7         1.085555    1.617780    0.399000
8         1.092133    1.394255    0.394000
Total time: 11:06
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp15k/qrnn_nl4_rnd2_35.m
Loss and accuracy using (cls_best): [1.1545019, tensor(0.5490)]
OrderedDict([('data/mldoc/es-10/models/sp15k/qrnn_nl4_rnd2_35.m',
              0.5490000247955322)])
Noise:  40
Processing data/mldoc/es-1/models/sp15k/qrnn_rnd-nl4.m
es-10
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp15k/qrnn_nl4_rnd2_40.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/es.dev.csv
Added noise to 3783 examples, only 0.600021146119687 have correct labels
Added noise to 400 examples, only 0.6 have correct labels
Data lm, trn: 13013, val: 1445
Data clsnoise0.4tv, trn: 9458, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁la', '▁.', 's', '▁en', '▁el', '▁y', '▁a', '▁que']
Starting classifier from random weights
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.199116    1.436976    0.385000
2         1.182144    3.020936    0.350000
3         1.169826    1.429357    0.359000
4         1.184806    2.984490    0.362000
5         1.142364    2.399835    0.322000
6         1.136863    25.139421   0.359000
7         1.139416    12.117358   0.365000
8         1.132085    2.642181    0.368000
Total time: 10:59
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp15k/qrnn_nl4_rnd2_40.m
Loss and accuracy using (cls_best): [2.2691653, tensor(0.5375)]
OrderedDict([('data/mldoc/es-10/models/sp15k/qrnn_nl4_rnd2_40.m',
              0.5375000238418579)])
Noise:  45
Processing data/mldoc/es-1/models/sp15k/qrnn_rnd-nl4.m
es-10
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp15k/qrnn_nl4_rnd2_45.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/es.dev.csv
Added noise to 4256 examples, only 0.5500105730598435 have correct labels
Added noise to 450 examples, only 0.55 have correct labels
Data lm, trn: 13013, val: 1445
Data clsnoise0.45tv, trn: 9458, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁la', '▁.', 's', '▁en', '▁el', '▁y', '▁a', '▁que']
Starting classifier from random weights
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.211562    1.438882    0.337000
2         1.218273    222.648880  0.291000
3         1.203696    1.414955    0.343000
4         1.204386    1.364311    0.369000
5         1.203539    3.488942    0.356000
6         1.175719    6.136192    0.349000
7         1.145031    1.604449    0.365000
8         1.141291    2.528898    0.361000
Total time: 10:55
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp15k/qrnn_nl4_rnd2_45.m
Loss and accuracy using (cls_best): [1.6499722, tensor(0.5272)]
OrderedDict([('data/mldoc/es-10/models/sp15k/qrnn_nl4_rnd2_45.m',
              0.5272499918937683)])
Noise:  50
Processing data/mldoc/es-1/models/sp15k/qrnn_rnd-nl4.m
es-10
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp15k/qrnn_nl4_rnd2_50.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/es.dev.csv
Added noise to 4729 examples, only 0.5 have correct labels
Added noise to 500 examples, only 0.5 have correct labels
Data lm, trn: 13013, val: 1445
Data clsnoise0.5tv, trn: 9458, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁la', '▁.', 's', '▁en', '▁el', '▁y', '▁a', '▁que']
Starting classifier from random weights
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.232532    1.421933    0.272000
2         1.237531    1.822951    0.252000
3         1.254736    1.534031    0.266000
4         1.252628    1.536729    0.266000
5         1.234980    1.544544    0.266000
6         1.234228    1.558047    0.266000
7         1.240368    1.538028    0.266000
8         1.253012    1.540725    0.266000
Total time: 11:01
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp15k/qrnn_nl4_rnd2_50.m
Loss and accuracy using (cls_best): [1.4770876, tensor(0.3072)]
OrderedDict([('data/mldoc/es-10/models/sp15k/qrnn_nl4_rnd2_50.m',
              0.3072499930858612)])
Noise:  55
Processing data/mldoc/es-1/models/sp15k/qrnn_rnd-nl4.m
es-10
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp15k/qrnn_nl4_rnd2_55.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/es.dev.csv
Added noise to 5201 examples, only 0.4500951575385917 have correct labels
Added noise to 550 examples, only 0.45 have correct labels
Data lm, trn: 13013, val: 1445
Data clsnoise0.55tv, trn: 9458, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁la', '▁.', 's', '▁en', '▁el', '▁y', '▁a', '▁que']
Starting classifier from random weights
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.267338    1.647622    0.253000
2         1.267763    1.562070    0.288000
3         1.249864    2.255809    0.258000
4         1.246395    1.470975    0.307000
5         1.229589    1.466512    0.305000
6         1.221072    1.472953    0.307000
7         1.202461    1.430212    0.323000
8         1.202863    1.416951    0.320000
Total time: 10:59
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp15k/qrnn_nl4_rnd2_55.m
Loss and accuracy using (cls_best): [1.1913521, tensor(0.5217)]
OrderedDict([('data/mldoc/es-10/models/sp15k/qrnn_nl4_rnd2_55.m',
              0.5217499732971191)])
Noise:  60
Processing data/mldoc/es-1/models/sp15k/qrnn_rnd-nl4.m
es-10
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp15k/qrnn_nl4_rnd2_60.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/es.dev.csv
Added noise to 5674 examples, only 0.4000845844787482 have correct labels
Added noise to 600 examples, only 0.4 have correct labels
Data lm, trn: 13013, val: 1445
Data clsnoise0.6tv, trn: 9458, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁la', '▁.', 's', '▁en', '▁el', '▁y', '▁a', '▁que']
Starting classifier from random weights
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.277718    1.426537    0.253000
2         1.261508    1.677472    0.229000
3         1.263114    1.585918    0.276000
4         1.255874    2.783197    0.293000
5         1.250460    3.303977    0.353000
6         1.239010    10.652842   0.326000
7         1.220644    4.388613    0.317000
8         1.226328    6.756069    0.327000
Total time: 11:07
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp15k/qrnn_nl4_rnd2_60.m
Loss and accuracy using (cls_best): [2.6658566, tensor(0.5073)]
OrderedDict([('data/mldoc/es-10/models/sp15k/qrnn_nl4_rnd2_60.m',
              0.5072500109672546)])
Noise:  65
Processing data/mldoc/es-1/models/sp15k/qrnn_rnd-nl4.m
es-10
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp15k/qrnn_nl4_rnd2_65.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/es.dev.csv
Added noise to 6147 examples, only 0.35007401141890465 have correct labels
Added noise to 650 examples, only 0.35 have correct labels
Data lm, trn: 13013, val: 1445
Data clsnoise0.65tv, trn: 9458, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁la', '▁.', 's', '▁en', '▁el', '▁y', '▁a', '▁que']
Starting classifier from random weights
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.263687    1.667461    0.279000
2         1.287779    8.032739    0.227000
3         1.289650    1.522185    0.229000
4         1.287830    1.587737    0.272000
5         1.267796    1.598588    0.272000
6         1.276992    1.563360    0.272000
7         1.276122    1.564773    0.272000
8         1.268917    1.559838    0.229000
Total time: 11:10
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp15k/qrnn_nl4_rnd2_65.m
Loss and accuracy using (cls_best): [1.4682757, tensor(0.3072)]
OrderedDict([('data/mldoc/es-10/models/sp15k/qrnn_nl4_rnd2_65.m',
              0.3072499930858612)])
Noise:  70
Processing data/mldoc/es-1/models/sp15k/qrnn_rnd-nl4.m
es-10
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp15k/qrnn_nl4_rnd2_70.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/es.dev.csv
Added noise to 6620 examples, only 0.30006343835906113 have correct labels
Added noise to 700 examples, only 0.3 have correct labels
Data lm, trn: 13013, val: 1445
Data clsnoise0.7tv, trn: 9458, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁la', '▁.', 's', '▁en', '▁el', '▁y', '▁a', '▁que']
Starting classifier from random weights
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.278968    1.395868    0.308000
2         1.287285    2.381748    0.271000
3         1.281547    1.510829    0.273000
4         1.285452    1.517207    0.273000
5         1.280697    1.535823    0.273000
6         1.289347    1.505417    0.273000
7         1.278146    1.527413    0.273000
8         1.271342    1.522274    0.273000
Total time: 10:48
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp15k/qrnn_nl4_rnd2_70.m
Loss and accuracy using (cls_best): [1.4725544, tensor(0.3115)]
OrderedDict([('data/mldoc/es-10/models/sp15k/qrnn_nl4_rnd2_70.m',
              0.31150001287460327)])
Noise:  75
Processing data/mldoc/es-1/models/sp15k/qrnn_rnd-nl4.m
es-10
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp15k/qrnn_nl4_rnd2_75.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/es.dev.csv
Added noise to 7093 examples, only 0.2500528652992176 have correct labels
Added noise to 750 examples, only 0.25 have correct labels
Data lm, trn: 13013, val: 1445
Data clsnoise0.75tv, trn: 9458, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁la', '▁.', 's', '▁en', '▁el', '▁y', '▁a', '▁que']
Starting classifier from random weights
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.278254    1.406109    0.294000
2         1.284943    1.538882    0.268000
3         1.277497    3.605533    0.256000
4         1.260796    1.501276    0.259000
5         1.254600    2.722533    0.273000
6         1.250827    4.186543    0.269000
7         1.236754    3.883370    0.273000
8         1.239405    1.523115    0.291000
Total time: 11:17
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp15k/qrnn_nl4_rnd2_75.m
Loss and accuracy using (cls_best): [1.7807395, tensor(0.1478)]
OrderedDict([('data/mldoc/es-10/models/sp15k/qrnn_nl4_rnd2_75.m',
              0.14775000512599945)])
    noise  accuracy
0    0.00   0.76950
1    0.05   0.72725
2    0.10   0.71725
3    0.15   0.59550
4    0.20   0.59850
5    0.25   0.57075
6    0.30   0.58175
7    0.35   0.54900
8    0.40   0.53750
9    0.45   0.52725
10   0.50   0.30725
11   0.55   0.52175
12   0.60   0.50725
13   0.65   0.30725
14   0.70   0.31150
15   0.75   0.14775
```

# LSTM
```
 python -m ulmfit eval_noise_resistance --lang=es --size=10 --prefix-name="rnd2_" --model="sp30k/lstm_nl4.m" --label-smoothing-eps=0.1 --random-init=True
Noise:  0
Processing data/mldoc/es-1/models/sp30k/lstm_nl4.m
es-10
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp30k/lstm_nl4_rnd2_0.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/es.dev.csv
Data lm, trn: 13013, val: 1445
Data cls, trn: 9458, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁la', '▁.', '▁en', '▁el', '▁y', 's', '▁a', '▁que']
Starting classifier from random weights
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.659161    4.290442    0.318000
2         0.750901    2.120531    0.277000
3         0.780925    2.740564    0.277000
4         0.761562    4.889140    0.293000
5         0.743035    14.893404   0.279000
6         0.726612    3.620207    0.277000
7         0.760487    2.801041    0.293000
8         0.751857    2.819516    0.293000
Total time: 30:06
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp30k/lstm_nl4_rnd2_0.m
Loss and accuracy using (cls_best): [2.932457, tensor(0.3072)]
OrderedDict([('data/mldoc/es-10/models/sp30k/lstm_nl4_rnd2_0.m',
              0.3072499930858612)])
Noise:  5
Processing data/mldoc/es-1/models/sp30k/lstm_nl4.m
es-10
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp30k/lstm_nl4_rnd2_5.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/es.dev.csv
Added noise to 472 examples, only 0.9500951575385916 have correct labels
Added noise to 50 examples, only 0.95 have correct labels
Data lm, trn: 13013, val: 1445
Data clsnoise0.05tv, trn: 9458, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁la', '▁.', '▁en', '▁el', '▁y', 's', '▁a', '▁que']
Starting classifier from random weights
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.744206    2.318675    0.274000
2         0.845912    1.891602    0.276000
3         0.831846    3.916036    0.274000
4         0.831695    1.779413    0.274000
5         0.830590    1.956843    0.274000
6         0.831666    1.893295    0.274000
7         0.850189    1.924907    0.274000
8         0.829423    1.925661    0.274000
Total time: 30:02
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp30k/lstm_nl4_rnd2_5.m
Loss and accuracy using (cls_best): [1.8306484, tensor(0.3072)]
OrderedDict([('data/mldoc/es-10/models/sp30k/lstm_nl4_rnd2_5.m',
              0.3072499930858612)])
Noise:  10
Processing data/mldoc/es-1/models/sp30k/lstm_nl4.m
es-10
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp30k/lstm_nl4_rnd2_10.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/es.dev.csv
Added noise to 945 examples, only 0.9000845844787482 have correct labels
Added noise to 100 examples, only 0.9 have correct labels
Data lm, trn: 13013, val: 1445
Data clsnoise0.1tv, trn: 9458, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁la', '▁.', '▁en', '▁el', '▁y', 's', '▁a', '▁que']
Starting classifier from random weights
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.898602    1.545597    0.293000
2         0.903453    176.651031  0.209000
3         0.916878    13.823226   0.276000
4         0.898918    24.522129   0.273000
5         0.912165    1.825259    0.274000
6         0.931385    1.808111    0.274000
7         0.889932    2.430218    0.273000
8         0.927120    2.782772    0.276000
Total time: 31:02
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp30k/lstm_nl4_rnd2_10.m
Loss and accuracy using (cls_best): [1.7723552, tensor(0.3105)]
OrderedDict([('data/mldoc/es-10/models/sp30k/lstm_nl4_rnd2_10.m',
              0.31049999594688416)])
Noise:  15
Processing data/mldoc/es-1/models/sp30k/lstm_nl4.m
es-10
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp30k/lstm_nl4_rnd2_15.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/es.dev.csv
Added noise to 1418 examples, only 0.8500740114189046 have correct labels
Added noise to 150 examples, only 0.85 have correct labels
Data lm, trn: 13013, val: 1445
Data clsnoise0.15tv, trn: 9458, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁la', '▁.', '▁en', '▁el', '▁y', 's', '▁a', '▁que']
Starting classifier from random weights
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.973683    6.861582    0.322000
2         0.966286    2.543203    0.273000
3         1.016659    3.716355    0.205000
4         0.955702    1.692103    0.273000
5         0.967680    3.590133    0.280000
6         0.982580    8.608456    0.269000
7         0.940222    2.429212    0.273000
8         0.996999    2.896548    0.269000
Total time: 30:04
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp30k/lstm_nl4_rnd2_15.m
Loss and accuracy using (cls_best): [1.9653525, tensor(0.3075)]
OrderedDict([('data/mldoc/es-10/models/sp30k/lstm_nl4_rnd2_15.m',
              0.3075000047683716)])
Noise:  20
Processing data/mldoc/es-1/models/sp30k/lstm_nl4.m
es-10
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp30k/lstm_nl4_rnd2_20.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/es.dev.csv
Added noise to 1891 examples, only 0.8000634383590611 have correct labels
Added noise to 200 examples, only 0.8 have correct labels
Data lm, trn: 13013, val: 1445
Data clsnoise0.2tv, trn: 9458, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁la', '▁.', '▁en', '▁el', '▁y', 's', '▁a', '▁que']
Starting classifier from random weights
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.031621    2.023383    0.255000
2         1.055999    1.585043    0.255000
3         1.004598    1.747062    0.255000
4         1.036506    12.106596   0.257000
5         1.037610    7.782064    0.257000
6         1.010497    4.877003    0.255000
7         1.035525    9.258689    0.257000
8         1.032628    9.379140    0.257000
Total time: 30:00
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp30k/lstm_nl4_rnd2_20.m
Loss and accuracy using (cls_best): [2.198144, tensor(0.3105)]
OrderedDict([('data/mldoc/es-10/models/sp30k/lstm_nl4_rnd2_20.m',
              0.31049999594688416)])
Noise:  25
Processing data/mldoc/es-1/models/sp30k/lstm_nl4.m
es-10
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp30k/lstm_nl4_rnd2_25.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/es.dev.csv
Added noise to 2364 examples, only 0.7500528652992176 have correct labels
Added noise to 250 examples, only 0.75 have correct labels
Data lm, trn: 13013, val: 1445
Data clsnoise0.25tv, trn: 9458, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁la', '▁.', '▁en', '▁el', '▁y', 's', '▁a', '▁que']
Starting classifier from random weights
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.062350    1.959029    0.214000
2         1.061749    1.823441    0.262000
3         1.083258    12.511975   0.275000
4         1.089900    2.291190    0.275000
5         1.086922    8.510952    0.262000
6         1.090885    2.293679    0.261000
7         1.075075    2.249190    0.275000
8         1.068257    2.230014    0.275000
Total time: 30:26
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp30k/lstm_nl4_rnd2_25.m
Loss and accuracy using (cls_best): [2.080095, tensor(0.3072)]
OrderedDict([('data/mldoc/es-10/models/sp30k/lstm_nl4_rnd2_25.m',
              0.3072499930858612)])
Noise:  30
Processing data/mldoc/es-1/models/sp30k/lstm_nl4.m
es-10
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp30k/lstm_nl4_rnd2_30.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/es.dev.csv
Added noise to 2837 examples, only 0.7000422922393741 have correct labels
Added noise to 300 examples, only 0.7 have correct labels
Data lm, trn: 13013, val: 1445
Data clsnoise0.3tv, trn: 9458, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁la', '▁.', '▁en', '▁el', '▁y', 's', '▁a', '▁que']
Starting classifier from random weights
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.133636    1.806463    0.274000
2         1.110103    10.866484   0.265000
3         1.122795    6.634718    0.274000
4         1.112287    5.359313    0.262000
5         1.121410    174.423630  0.261000
6         1.110173    18.667475   0.262000
7         1.102498    12.188397   0.262000
8         1.117406    15.158224   0.249000
Total time: 29:59
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp30k/lstm_nl4_rnd2_30.m
Loss and accuracy using (cls_best): [8.974576, tensor(0.2555)]
OrderedDict([('data/mldoc/es-10/models/sp30k/lstm_nl4_rnd2_30.m',
              0.2554999887943268)])
Noise:  35
Processing data/mldoc/es-1/models/sp30k/lstm_nl4.m
es-10
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp30k/lstm_nl4_rnd2_35.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/es.dev.csv
Added noise to 3310 examples, only 0.6500317191795305 have correct labels
Added noise to 350 examples, only 0.65 have correct labels
Data lm, trn: 13013, val: 1445
Data clsnoise0.35tv, trn: 9458, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁la', '▁.', '▁en', '▁el', '▁y', 's', '▁a', '▁que']
Starting classifier from random weights
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.177323    2.044810    0.260000
2         1.145307    1.615969    0.254000
3         1.178776    2.619027    0.260000
4         1.170236    7.795065    0.253000
5         1.160120    1.604842    0.254000
6         1.160545    1.587269    0.254000
7         1.137365    1.629242    0.254000
8         1.150483    1.610021    0.254000
Total time: 29:59
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp30k/lstm_nl4_rnd2_35.m
Loss and accuracy using (cls_best): [1.5346162, tensor(0.3072)]
OrderedDict([('data/mldoc/es-10/models/sp30k/lstm_nl4_rnd2_35.m',
              0.3072499930858612)])
Noise:  40
Processing data/mldoc/es-1/models/sp30k/lstm_nl4.m
es-10
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp30k/lstm_nl4_rnd2_40.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/es.dev.csv
Added noise to 3783 examples, only 0.600021146119687 have correct labels
Added noise to 400 examples, only 0.6 have correct labels
Data lm, trn: 13013, val: 1445
Data clsnoise0.4tv, trn: 9458, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁la', '▁.', '▁en', '▁el', '▁y', 's', '▁a', '▁que']
Starting classifier from random weights
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.200657    1.747926    0.242000
2         1.213164    33.535637   0.250000
3         1.188038    1.609861    0.246000
4         1.185921    2.510113    0.246000
5         1.218471    4.206014    0.242000
6         1.187989    1.855121    0.251000
7         1.210737    4.329615    0.242000
8         1.193102    5.145269    0.242000
Total time: 30:09
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp30k/lstm_nl4_rnd2_40.m
Loss and accuracy using (cls_best): [2.3451035, tensor(0.3075)]
OrderedDict([('data/mldoc/es-10/models/sp30k/lstm_nl4_rnd2_40.m',
              0.3075000047683716)])
Noise:  45
Processing data/mldoc/es-1/models/sp30k/lstm_nl4.m
es-10
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp30k/lstm_nl4_rnd2_45.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/es.dev.csv
Added noise to 4256 examples, only 0.5500105730598435 have correct labels
Added noise to 450 examples, only 0.55 have correct labels
Data lm, trn: 13013, val: 1445
Data clsnoise0.45tv, trn: 9458, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁la', '▁.', '▁en', '▁el', '▁y', 's', '▁a', '▁que']
Starting classifier from random weights
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.241620    1.534527    0.278000
2         1.234608    4.180245    0.251000
3         1.225962    18.634438   0.251000
4         1.231608    1.596539    0.245000
5         1.242316    1.529011    0.245000
6         1.224234    1.559436    0.245000
7         1.219737    1.570357    0.245000
8         1.217654    1.558945    0.245000
Total time: 29:55
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp30k/lstm_nl4_rnd2_45.m
Loss and accuracy using (cls_best): [1.4820943, tensor(0.3072)]
OrderedDict([('data/mldoc/es-10/models/sp30k/lstm_nl4_rnd2_45.m',
              0.3072499930858612)])
Noise:  50
Processing data/mldoc/es-1/models/sp30k/lstm_nl4.m
es-10
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp30k/lstm_nl4_rnd2_50.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/es.dev.csv
Added noise to 4729 examples, only 0.5 have correct labels
Added noise to 500 examples, only 0.5 have correct labels
Data lm, trn: 13013, val: 1445
Data clsnoise0.5tv, trn: 9458, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁la', '▁.', '▁en', '▁el', '▁y', 's', '▁a', '▁que']
Starting classifier from random weights
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.248102    1.559488    0.241000
2         1.239413    6.912020    0.242000
3         1.258250    19.964916   0.240000
4         1.244972    14.503420   0.240000
5         1.231926    7.417413    0.237000
6         1.240820    3.726920    0.240000
7         1.250362    3.813594    0.240000
8         1.246718    4.295069    0.240000
Total time: 29:56
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp30k/lstm_nl4_rnd2_50.m
Loss and accuracy using (cls_best): [1.6464796, tensor(0.3105)]
OrderedDict([('data/mldoc/es-10/models/sp30k/lstm_nl4_rnd2_50.m',
              0.31049999594688416)])
Noise:  55
Processing data/mldoc/es-1/models/sp30k/lstm_nl4.m
es-10
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp30k/lstm_nl4_rnd2_55.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/es.dev.csv
Added noise to 5201 examples, only 0.4500951575385917 have correct labels
Added noise to 550 examples, only 0.45 have correct labels
Data lm, trn: 13013, val: 1445
Data clsnoise0.55tv, trn: 9458, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁la', '▁.', '▁en', '▁el', '▁y', 's', '▁a', '▁que']
Starting classifier from random weights
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.262681    1.390237    0.243000
2         1.261966    1.613178    0.221000
3         1.281932    80.386887   0.237000
4         1.266172    20.980446   0.220000
5         1.267419    1.564067    0.221000
6         1.256289    2.409412    0.215000
7         1.259451    2.007908    0.221000
8         1.248848    1.682695    0.221000
Total time: 30:44
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp30k/lstm_nl4_rnd2_55.m
Loss and accuracy using (cls_best): [1.5071006, tensor(0.3072)]
OrderedDict([('data/mldoc/es-10/models/sp30k/lstm_nl4_rnd2_55.m',
              0.3072499930858612)])
Noise:  60
Processing data/mldoc/es-1/models/sp30k/lstm_nl4.m
es-10
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp30k/lstm_nl4_rnd2_60.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/es.dev.csv
Added noise to 5674 examples, only 0.4000845844787482 have correct labels
Added noise to 600 examples, only 0.4 have correct labels
Data lm, trn: 13013, val: 1445
Data clsnoise0.6tv, trn: 9458, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁la', '▁.', '▁en', '▁el', '▁y', 's', '▁a', '▁que']
Starting classifier from random weights
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.299198    1.514330    0.257000
2         1.276767    1.648853    0.246000
3         1.273444    1.702971    0.249000
4         1.275886    5.679698    0.254000
5         1.282117    15.944865   0.249000
6         1.273512    4.761624    0.254000
7         1.283532    5.982591    0.254000
8         1.266130    4.487469    0.254000
Total time: 29:48
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp30k/lstm_nl4_rnd2_60.m
Loss and accuracy using (cls_best): [2.5968323, tensor(0.3072)]
OrderedDict([('data/mldoc/es-10/models/sp30k/lstm_nl4_rnd2_60.m',
              0.3072499930858612)])
Noise:  65
Processing data/mldoc/es-1/models/sp30k/lstm_nl4.m
es-10
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp30k/lstm_nl4_rnd2_65.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/es.dev.csv
Added noise to 6147 examples, only 0.35007401141890465 have correct labels
Added noise to 650 examples, only 0.35 have correct labels
Data lm, trn: 13013, val: 1445
Data clsnoise0.65tv, trn: 9458, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁la', '▁.', '▁en', '▁el', '▁y', 's', '▁a', '▁que']
Starting classifier from random weights
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.288758    1.800278    0.266000
2         1.286780    1.503401    0.272000
3         1.280267    1.921974    0.278000
4         1.285067    1.523227    0.259000
5         1.285825    1.526260    0.272000
6         1.276409    1.531018    0.267000
7         1.270072    1.525349    0.259000
8         1.269815    1.514847    0.242000
Total time: 29:50
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp30k/lstm_nl4_rnd2_65.m
Loss and accuracy using (cls_best): [1.4596524, tensor(0.3072)]
OrderedDict([('data/mldoc/es-10/models/sp30k/lstm_nl4_rnd2_65.m',
              0.3072499930858612)])
Noise:  70
Processing data/mldoc/es-1/models/sp30k/lstm_nl4.m
es-10
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp30k/lstm_nl4_rnd2_70.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/es.dev.csv
Added noise to 6620 examples, only 0.30006343835906113 have correct labels
Added noise to 700 examples, only 0.3 have correct labels
Data lm, trn: 13013, val: 1445
Data clsnoise0.7tv, trn: 9458, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁la', '▁.', '▁en', '▁el', '▁y', 's', '▁a', '▁que']
Starting classifier from random weights
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.271206    1.656565    0.258000
2         1.285489    2.193548    0.271000
3         1.283617    1.947972    0.264000
4         1.282205    8.228713    0.270000
5         1.275360    21.401678   0.270000
6         1.266841    32.251156   0.270000
7         1.263846    20.252125   0.258000
8         1.270404    21.024738   0.258000
Total time: 29:24
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp30k/lstm_nl4_rnd2_70.m
Loss and accuracy using (cls_best): [7.6955824, tensor(0.1828)]
OrderedDict([('data/mldoc/es-10/models/sp30k/lstm_nl4_rnd2_70.m',
              0.18275000154972076)])
Noise:  75
Processing data/mldoc/es-1/models/sp30k/lstm_nl4.m
es-10
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp30k/lstm_nl4_rnd2_75.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/es.dev.csv
Added noise to 7093 examples, only 0.2500528652992176 have correct labels
Added noise to 750 examples, only 0.25 have correct labels
Data lm, trn: 13013, val: 1445
Data clsnoise0.75tv, trn: 9458, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁la', '▁.', '▁en', '▁el', '▁y', 's', '▁a', '▁que']
Starting classifier from random weights
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.282777    1.870099    0.271000
2         1.271396    3.192471    0.264000
3         1.282228    11.233303   0.260000
4         1.276331    35.493816   0.256000
5         1.269741    3.304629    0.264000
6         1.266907    6.163653    0.261000
7         1.279470    3.861671    0.271000
8         1.269125    2.668693    0.260000
Total time: 30:42
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-10/models/sp30k/lstm_nl4_rnd2_75.m
Loss and accuracy using (cls_best): [1.6692431, tensor(0.1828)]
OrderedDict([('data/mldoc/es-10/models/sp30k/lstm_nl4_rnd2_75.m',
              0.18275000154972076)])
    noise  accuracy
0    0.00   0.30725
1    0.05   0.30725
2    0.10   0.31050
3    0.15   0.30750
4    0.20   0.31050
5    0.25   0.30725
6    0.30   0.25550
7    0.35   0.30725
8    0.40   0.30750
9    0.45   0.30725
10   0.50   0.31050
11   0.55   0.30725
12   0.60   0.30725
13   0.65   0.30725
14   0.70   0.18275
15   0.75   0.18275
```