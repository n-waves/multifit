                                            3         2.812496    2.877055    0.468569

4         2.705551    2.792535    0.479420

5         2.649598    2.726415    0.487439

6         2.599835    2.635610    0.499679

7         2.574639    2.554657    0.512358

8         2.489573    2.475936    0.523280

9         2.396540    2.415555    0.534089

10        2.374290    2.401968    0.536601

Total time: 15:49:20
data/wiki/ru-100/models/sp15k
Saving info data/wiki/ru-100/models/sp15k/qrnn_nl4-wide2.m/info.json
Fire trace:
1. Initial component
2. Accessed property "lm" (/home/test/workspace/ulmfit-multilingual/ulmfit/__main__.py:32)
3. Called routine "LMHyperParams" (/home/test/workspace/ulmfit-multilingual/ulmfit/__main__.py:32)
4. Accessed property "train" (/home/test/workspace/ulmfit-multilingual/ulmfit/pretrain_lm.py:174)
5. Called routine "train_lm" (/home/test/workspace/ulmfit-multilingual/ulmfit/pretrain_lm.py:174)
6. ('Could not consume arg:', '--nh')

Type:        NoneType
String form: None

Usage:       __main__.py lm --dataset-path data/wiki/ru-100 --tokenizer=sp --nl 4 --name nl4-wide2 --max-vocab 15000 --lang ru --qrnn=True - train 10 --bs=100 --drop_mult=0 -
(multifit) test@test:~/workspace/ulmfit-multilingual$ less data/wiki/ru-100/models/sp15k/qrnn_nl4-wide2.m/info.json
(multifit) test@test:~/workspace/ulmfit-multilingual$ CUDA_VISIBLE_DEVICES=1 python -m ulmfit lm --dataset-path data/wiki/${LANG}-100 --tokenizer='sp' --nl 4 --name 'nl4-wide2' --max-vocab 15000 --lang ${LANG} --qrnn=True - train 10 --bs=100 --drop_mult=0 ^C100 --
(multifit) test@test:~/workspace/ulmfit-multilingual$ mv data/wiki/ru-100/models/sp15k/qrnn_nl4-wide2.m/ data/wiki/ru-100/models/sp15k/qrnn_nl4-2.m/
(multifit) test@test:~/workspace/ulmfit-multilingual$ less data/wiki/ru-100/models/sp15k/qrnn_nl4-wide2.m/info.json^C
(multifit) test@test:~/workspace/ulmfit-multilingual$ CUDA_VISIBLE_DEVICES=1 python -m ulmfit lm --dataset-path data/wiki/${LANG}-100 --tokenizer='sp' --nl 4 --name 'nl4-wid
e2' --max-vocab 15000 --lang ${LANG} --qrnn=True --nh 3100 - train 10 --bs=100 --drop_mult=0
Max vocab: 15000
Cache dir: data/wiki/ru-100/models/sp15k
Model dir: data/wiki/ru-100/models/sp15k/qrnn_nl4-wide2.m
^CTraceback (most recent call last):
  File "/home/test/anaconda3/envs/multifit/lib/python3.7/runpy.py", line 193, in _run_module_as_main
    "__main__", mod_spec)
  File "/home/test/anaconda3/envs/multifit/lib/python3.7/runpy.py", line 85, in _run_code
    exec(code, run_globals)
  File "/home/test/workspace/ulmfit-multilingual/ulmfit/__main__.py", line 119, in <module>
    fire.Fire(ULMFiT())
  File "/home/test/anaconda3/envs/multifit/lib/python3.7/site-packages/fire/core.py", line 127, in Fire
    component_trace = _Fire(component, args, context, name)
  File "/home/test/anaconda3/envs/multifit/lib/python3.7/site-packages/fire/core.py", line 366, in _Fire
    component, remaining_args)
  File "/home/test/anaconda3/envs/multifit/lib/python3.7/site-packages/fire/core.py", line 542, in _CallCallable
    result = fn(*varargs, **kwargs)
  File "/home/test/workspace/ulmfit-multilingual/ulmfit/pretrain_lm.py", line 176, in train_lm
    data_lm = self.load_wiki_data(bs=bs) if data_lm is None else data_lm
  File "/home/test/workspace/ulmfit-multilingual/ulmfit/pretrain_lm.py", line 253, in load_wiki_data
    train_df=read_wiki_articles(trn_path),
  File "/home/test/workspace/ulmfit-multilingual/ulmfit/pretrain_lm.py", line 48, in read_wiki_articles
    if i < len(lines)-2 and lines[i+1].strip() == "" and istitle(lines[i+2]):
  File "/home/test/workspace/ulmfit-multilingual/ulmfit/pretrain_lm.py", line 39, in istitle
    return len(re.findall(r'^ ?= [^=]* = ?$', line)) != 0
  File "/home/test/anaconda3/envs/multifit/lib/python3.7/re.py", line 223, in findall
    return _compile(pattern, flags).findall(string)
KeyboardInterrupt
^C
(multifit) test@test:~/workspace/ulmfit-multilingual$ CUDA_VISIBLE_DEVICES=1 python -m ulmfit lm --dataset-path data/wiki/${LANG}-100 --tokenizer='sp' --nl 4 --name 'nl4-wide2' --max-vocab 15000 --lang ${LANG} --qrnn=True --nh 3100 - train 10 --bs=100 --drop_mult=0  --label-smoothing-eps=0.1
Max vocab: 15000
Cache dir: data/wiki/ru-100/models/sp15k
Model dir: data/wiki/ru-100/models/sp15k/qrnn_nl4-wide2.m
Wiki text was split to 193047 articles
Wiki text was split to 460 articles
Data lm, trn: 193047, val: 460
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁.', '▁в', 'а', 'и', 'е', '▁и', 'й', '▁на', 'х']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Training lm from random weights
epoch     train_loss  valid_loss  accuracy
1         3.908233    3.950865    0.463469
2         3.738863    3.815026    0.477703
3         3.696502    3.779513    0.483625
4         3.692592    3.720908    0.490143
5         3.600519    3.652444    0.501671
6         3.564568    3.582584    0.511550
7         3.472859    3.493226    0.525943
8         3.390483    3.407970    0.541749
9         3.351620    3.344207    0.552758
10        3.329683    3.330087    0.556380
Total time: 51:05:43
data/wiki/ru-100/models/sp15k
Saving info data/wiki/ru-100/models/sp15k/qrnn_nl4-wide2.m/info.json
(multifit) test@test:~/workspace/ulmfit-multilingual$ export CUDA_VISIBLE_DEVICES=1
(multifit) test@test:~/workspace/ulmfit-multilingual$ LANG=ru
(multifit) test@test:~/workspace/ulmfit-multilingual$ NAME=nl4-wide2
(multifit) test@test:~/workspace/ulmfit-multilingual$ python -m ulmfit cls --dataset-path data/mldoc/${LANG}-1 --base-lm-path data/wiki/${LANG}-100/models/sp15k/qrnn_${NAME}.m  --lang=${LANG} --name ${NAME} - train 20 --bs 18 --num-cls-epochs=4 --lr_sched=1cycle --label-smoothing-eps=0.1
Max vocab: 15000
Cache dir: /home/test/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp15k
Model dir: /home/test/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp15k/qrnn_nl4-wide2.m
Loading validation /home/test/workspace/ulmfit-multilingual/data/mldoc/ru-1/ru.dev.csv
Data lm, trn: 9195, val: 1021
Data cls, trn: 1000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁.', '▁в', 'а', 'и', 'е', '▁и', 'й', '▁на', 'х']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
Training lm from:  [PosixPath('/home/test/workspace/ulmfit-multilingual/data/wiki/ru-100/models/sp15k/qrnn_nl4-wide2.m/lm_best'), PosixPath('/home/test/workspace/ulmfit-multilingual/data/wiki/ru-100/models/sp15k/qrnn_nl4-wide2.m/../itos')]
epoch     train_loss  valid_loss  accuracy
1         3.777423    3.222261    0.564503
Total time: 04:35
epoch     train_loss  valid_loss  accuracy
1         3.292465    3.029143    0.602257
2         3.034045    2.858176    0.634576
3         2.943366    2.710314    0.665116
4         2.722069    2.596702    0.687515
5         2.819853    2.508158    0.705020
6         2.734984    2.417240    0.724748
7         2.674353    2.332395    0.743694
8         2.527344    2.251373    0.762892
9         2.473972    2.168185    0.784043
10        2.359504    2.093983    0.803255
11        2.287590    2.019540    0.823566
12        2.254421    1.943832    0.845138
13        2.203321    1.884380    0.863381
14        2.142532    1.824186    0.881509
15        2.121573    1.777664    0.894901
16        2.013238    1.740772    0.905824
17        2.026189    1.715271    0.913569
18        1.904322    1.700163    0.917917
19        1.889113    1.692539    0.919811
20        1.903118    1.691033    0.920319
Total time: 3:10:09
/home/test/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp15k
Saving info /home/test/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp15k/qrnn_nl4-wide2.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.006922    0.880313    0.788000
2         0.823572    0.782953    0.860000
3         0.679078    0.749164    0.872000
4         0.579215    0.707200    0.872000
Total time: 06:30
Saving models at /home/test/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp15k/qrnn_nl4-wide2.m
Loss and accuracy using (cls_best): [0.3935929, tensor(0.8708)]
0.393592894077301
0.8707500100135803
(multifit) test@test:~/workspace/ulmfit-multilingual$ python -m ulmfit cls --dataset-path data/mldoc/${LANG}-1 --base-lm-path data/mldoc/${LANG}-1/models/sp15k/qrnn_${NAME}.m  --lang=${LANG} --name ${NAME}-16 - train 0 --bs 18 --num-cls-epochs=16 --lr_sched=1cycle --label-smoothing-eps=0.1
Max vocab: 15000
Cache dir: /home/test/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp15k
Model dir: /home/test/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp15k/qrnn_nl4-wide2-16.m
Loading validation /home/test/workspace/ulmfit-multilingual/data/mldoc/ru-1/ru.dev.csv
Data lm, trn: 9195, val: 1021
Data cls, trn: 1000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁.', '▁в', 'а', 'и', 'е', '▁и', 'й', '▁на', 'х']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
/home/test/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp15k
Saving info /home/test/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp15k/qrnn_nl4-wide2-16.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.067446    0.822965    0.824000
2         0.897088    0.845636    0.826000
3         0.778055    0.828693    0.847000
4         0.685080    0.893327    0.823000
5         0.620457    0.929057    0.800000
6         0.587644    0.802154    0.859000
7         0.570255    0.713434    0.872000
8         0.543071    0.705259    0.871000
9         0.517465    0.715090    0.867000
10        0.498291    0.695459    0.876000
11        0.497857    0.698052    0.862000
12        0.486924    0.681911    0.878000
13        0.479041    0.676714    0.874000
14        0.475131    0.677843    0.878000
15        0.467238    0.672065    0.876000
16        0.476889    0.680850    0.875000
Total time: 23:47
Saving models at /home/test/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp15k/qrnn_nl4-wide2-16.m
Loss and accuracy using (cls_best): [0.41155785, tensor(0.8700)]
0.4115578532218933
0.8700000047683716