## bptt140
### CLS
```
LANG=ru
python -m ulmfit lm --dataset-path data/wiki/${LANG}-100 --tokenizer='sp' --nl 4 --name 'nl4-bptt140' --max-vocab 15000 --lang ${LANG} --qrnn=True - train 14 --bs=50 --drop_mult=0  --label-smoothing-eps=0.1

Max vocab: 15000
Cache dir: data/wiki/ru-100/models/sp15k
Model dir: data/wiki/ru-100/models/sp15k/qrnn_nl4-bptt140.m
Wiki text was split to 193047 articles
Wiki text was split to 460 articles
Data lm, trn: 193047, val: 460
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁.', '▁в', 'а', 'и', 'е', '▁и', 'й', '▁на', 'х']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
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
Bptt 70
Training lm from random weights
epoch     train_loss  valid_loss  accuracy
1         4.022324    4.046203    0.451453
2         3.840025    3.935647    0.462081
3         3.873172    3.940451    0.459741
4         3.850415    3.918466    0.462763
5         3.814188    3.898976    0.465359
6         3.771836    3.857443    0.472302
7         3.761032    3.801748    0.479811
8         3.712323    3.755207    0.486181
9         3.706044    3.707724    0.493604
10        3.693287    3.650429    0.502407
11        3.563701    3.588871    0.513251
12        3.477192    3.538018    0.522175
13        3.486541    3.504327    0.528571
14        3.484132    3.495028    0.530480
Total time: 19:53:42
data/wiki/ru-100/models/sp15k
Saving info data/wiki/ru-100/models/sp15k/qrnn_nl4-bptt140.m/info.json
```

### MLDoc
```
python -m ulmfit cls --dataset-path data/mldoc/${LANG}-1 --base-lm-path  data/wiki/${LANG}-100/models/sp15k/qrnn_nl4-bptt140.m  --lang=${LANG} --name nl4-bptt140 --bptt=140 - train 20 --bs 18 --num-cls-epochs=8 --lr_sched=1cycle --label-smoothing-eps=0.1
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp15k/qrnn_nl4-bptt140.m
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1/ru.dev.csv
Running tokenization lm140...
Data lm140, trn: 9195, val: 1021
Data cls, trn: 1000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁.', '▁в', 'а', 'и', 'е', '▁и', 'й', '▁на', 'х']
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
Bptt 140
Training lm from:  [PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/wiki/ru-100/models/sp15k/qrnn_nl4-bptt140.m/lm_best'), PosixPath('/home/pczapla/workspace/ulmfit-multilingual/data/wiki/ru-100/models/sp15k/qrnn_nl4-bptt140.m/../itos')]
epoch     train_loss  valid_loss  accuracy
1         3.855653    2.852676    0.452966
Total time: 01:56
epoch     train_loss  valid_loss  accuracy
1         3.052491    2.572216    0.504417
2         2.565436    2.252638    0.557341
3         2.238792    1.980807    0.599827
4         1.990266    1.784574    0.629615
5         1.851867    1.647570    0.651466
6         1.800950    1.539561    0.668753
7         1.692110    1.447140    0.684268
8         1.546868    1.380541    0.696082
9         1.618451    1.312476    0.708090
10        1.478336    1.255234    0.718722
11        1.477739    1.197032    0.729453
12        1.418238    1.151929    0.738932
13        1.384237    1.103246    0.748681
14        1.245625    1.061356    0.757009
15        1.289399    1.028937    0.763857
16        1.280893    1.006447    0.768844
17        1.268177    0.985106    0.773329
18        1.251713    0.975138    0.775565
19        1.288352    0.968812    0.776884
20        1.164147    0.967133    0.777174
Total time: 52:46
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp15k/qrnn_nl4-bptt140.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.905054    0.572781    0.811000
2         0.747270    0.606469    0.806000
3         0.644590    0.682804    0.810000
4         0.457427    0.605931    0.863000
5         0.351969    0.652187    0.842000
6         0.286099    0.589351    0.860000
7         0.218377    0.622760    0.857000
8         0.185043    0.597372    0.860000
Total time: 03:05
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp15k/qrnn_nl4-bptt140.m
Loss and accuracy using (cls_best): [0.47860995, tensor(0.8737)] [0.48851612, tensor(0.8600)]
val_loss:     0.48851612
val_accuracy: 0.8600000143051147
tst_loss:     0.47860995
tst_accuracy: 0.8737499713897705
```