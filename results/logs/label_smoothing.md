# MLDoc

## QRNN 15k

## LM+CLS training
```
python -m ulmfit eval --glob="wiki/*-100/models/sp15k/qrnn_nl4.m" --name nl4-sl --dataset-template='../mldoc/${lang}-1' --num-lm-epochs=20  --num-cls-epochs=8  --bs=18 --lr_sched=1cycle --label-smoothing-eps=0.1
Processing data/wiki/de-100/models/sp15k/qrnn_nl4.m
../mldoc/de-1
Max vocab: 15000
Cache dir: /home/test/workspace/ulmfit-multilingual/data/mldoc/de-1/models/sp15k
Model dir: /home/test/workspace/ulmfit-multilingual/data/mldoc/de-1/models/sp15k/qrnn_nl4-sl.m
Training
Loading validation /home/test/workspace/ulmfit-multilingual/data/mldoc/de-1/de.dev.csv
Data lm, trn: 13500, val: 1500
Data cls, trn: 1000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁.', '▁,', '▁der', '▁die', 'en', '▁und', 's', '▁in', 'er', "▁&'"]
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
Training lm from:  [PosixPath('/home/test/workspace/ulmfit-multilingual/data/wiki/de-100/models/sp15k/qrnn_nl4.m/lm_best'), PosixPath('/home/test/workspace/ulmfit-multilingual/data/wiki/de-100/models/sp15k/qrnn_nl4.m/../itos')]
epoch     train_loss  valid_loss  accuracy
1         4.336932    3.600409    0.524499

Total time: 01:55
epoch     train_loss  valid_loss  accuracy
1         3.867303    3.446689    0.547847

2         3.475966    3.257031    0.578702

3         3.233705    3.085521    0.607024

4         3.170250    2.964533    0.627028

5         3.059212    2.876878    0.641094

6         2.965926    2.797152    0.654905

7         2.974514    2.743403    0.663470

8         2.858759    2.690824    0.672891

9         2.866673    2.646101    0.680956

10        2.814579    2.610239    0.687777

11        2.806775    2.577145    0.694683

12        2.741160    2.540292    0.702336

13        2.753407    2.506782    0.709361

14        2.720171    2.480167    0.715709

15        2.631490    2.452236    0.721706

16        2.587928    2.431256    0.726922

17        2.608380    2.417473    0.729975

18        2.564811    2.407112    0.732487

19        2.599782    2.403481    0.733463

20        2.603479    2.402438    0.733552

Total time: 1:18:42
/home/test/workspace/ulmfit-multilingual/data/mldoc/de-1/models/sp15k
Saving info /home/test/workspace/ulmfit-multilingual/data/mldoc/de-1/models/sp15k/qrnn_nl4-sl.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.790572    0.610132    0.925000

2         0.655777    0.619519    0.945000

3         0.591051    0.654355    0.930000

4         0.541992    0.582572    0.944000

5         0.514052    0.574799    0.939000

6         0.490411    0.550166    0.949000

7         0.476344    0.553866    0.947000

8         0.469109    0.548871    0.947000

Total time: 02:43
Saving models at /home/test/workspace/ulmfit-multilingual/data/mldoc/de-1/models/sp15k/qrnn_nl4-sl.m
Loss and accuracy using (cls_best): [0.2010318, tensor(0.9610)]

Processing data/wiki/en-100/models/sp15k/qrnn_nl4.m
../mldoc/en-1
Max vocab: 15000
Cache dir: /home/test/workspace/ulmfit-multilingual/data/mldoc/en-1/models/sp15k
Model dir: /home/test/workspace/ulmfit-multilingual/data/mldoc/en-1/models/sp15k/qrnn_nl4-sl.m
Training
Loading validation /home/test/workspace/ulmfit-multilingual/data/mldoc/en-1/en.dev.csv
Data lm, trn: 13500, val: 1500
Data cls, trn: 1000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁the', '▁,', 's', '▁.', '▁of', '▁and', '▁in', '▁to', '▁a', 'ed']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
Training lm from:  [PosixPath('/home/test/workspace/ulmfit-multilingual/data/wiki/en-100/models/sp15k/qrnn_nl4.m/lm_best'), PosixPath('/home/test/workspace/ulmfit-multilingual/data/wiki/en-100/models/sp15k/qrnn_nl4.m/../itos')]
epoch     train_loss  valid_loss  accuracy
1         5.361436    4.703674    0.343924

Total time: 02:55
epoch     train_loss  valid_loss  accuracy
1         4.757253    4.493189    0.372715

2         4.515738    4.291261    0.404511

3         4.301139    4.110973    0.430612

4         4.162306    3.964101    0.450053

5         4.062487    3.841125    0.466488

6         3.898028    3.740108    0.480628

7         3.876914    3.660982    0.493164

8         3.793781    3.593925    0.502977

9         3.736873    3.528259    0.513241

10        3.695738    3.477659    0.521709

11        3.668749    3.431972    0.529821

12        3.642119    3.385145    0.537860

13        3.556678    3.343567    0.545521

14        3.548823    3.305735    0.552502

15        3.520068    3.272878    0.558736

16        3.439619    3.247021    0.563504

17        3.391731    3.228240    0.567151

18        3.398134    3.217466    0.569319

19        3.402400    3.212110    0.570352

20        3.375334    3.210727    0.570589

Total time: 1:19:32
/home/test/workspace/ulmfit-multilingual/data/mldoc/en-1/models/sp15k
Saving info /home/test/workspace/ulmfit-multilingual/data/mldoc/en-1/models/sp15k/qrnn_nl4-sl.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.855421    0.644234    0.903000

2         0.722607    0.665451    0.954000

3         0.629362    0.590857    0.945000

4         0.555874    0.562738    0.950000

5         0.522701    0.549048    0.953000

6         0.506758    0.536445    0.961000

7         0.489250    0.527361    0.963000

8         0.482953    0.528452    0.961000

Total time: 02:54
Saving models at /home/test/workspace/ulmfit-multilingual/data/mldoc/en-1/models/sp15k/qrnn_nl4-sl.m
Loss and accuracy using (cls_best): [0.20434816, tensor(0.9555)]

Processing data/wiki/es-100/models/sp15k/qrnn_nl4.m
../mldoc/es-1
Max vocab: 15000
Cache dir: /home/test/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp15k
Model dir: /home/test/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp15k/qrnn_nl4-sl.m
Training
Loading validation /home/test/workspace/ulmfit-multilingual/data/mldoc/es-1/es.dev.csv
Data lm, trn: 13013, val: 1445
Data cls, trn: 1000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁.', '▁la', 's', '▁el', '▁en', '▁y', '▁a', "▁&'"]
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
Training lm from:  [PosixPath('/home/test/workspace/ulmfit-multilingual/data/wiki/es-100/models/sp15k/qrnn_nl4.m/lm_best'), PosixPath('/home/test/workspace/ulmfit-multilingual/data/wiki/es-100/models/sp15k/qrnn_nl4.m/../itos')]
epoch     train_loss  valid_loss  accuracy
1         3.925853    3.293504    0.535753

Total time: 01:42
epoch     train_loss  valid_loss  accuracy
1         3.584036    3.131868    0.563887

2         3.341769    2.924815    0.605171

3         3.064540    2.760458    0.637288

4         2.979203    2.651128    0.657270

5         2.913760    2.569732    0.670629

6         2.901400    2.507033    0.682515

7         2.884516    2.454021    0.692617

8         2.759039    2.404587    0.703479

9         2.730353    2.367218    0.711349

10        2.657660    2.325339    0.720632

11        2.638513    2.292851    0.728599

12        2.629284    2.258947    0.737086

13        2.542013    2.226581    0.744815

14        2.464086    2.202000    0.750827

15        2.489060    2.177043    0.757989

16        2.446775    2.158471    0.762447

17        2.388175    2.144888    0.766083

18        2.415777    2.136921    0.768290

19        2.454445    2.132960    0.769424

20        2.346935    2.132173    0.769667

Total time: 47:14
/home/test/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp15k
Saving info /home/test/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp15k/qrnn_nl4-sl.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.842615    0.671844    0.919000

2         0.705828    0.669006    0.948000

3         0.627629    0.572335    0.944000

4         0.576609    0.600053    0.953000

5         0.532899    0.542761    0.962000

6         0.503425    0.548742    0.961000

7         0.495654    0.535397    0.959000

8         0.487190    0.545798    0.958000

Total time: 02:16
Saving models at /home/test/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp15k/qrnn_nl4-sl.m
Loss and accuracy using (cls_best): [0.18526463, tensor(0.9582)]

Processing data/wiki/fr-100/models/sp15k/qrnn_nl4.m
../mldoc/fr-1
Max vocab: 15000
Cache dir: /home/test/workspace/ulmfit-multilingual/data/mldoc/fr-1/models/sp15k
Model dir: /home/test/workspace/ulmfit-multilingual/data/mldoc/fr-1/models/sp15k/qrnn_nl4-sl.m
Training
Loading validation /home/test/workspace/ulmfit-multilingual/data/mldoc/fr-1/fr.dev.csv
Data lm, trn: 13500, val: 1500
Running tokenization cls...
Data cls, trn: 1000, val: 1000

Running tokenization tst...
Data tst, trn: 1000, val: 4000

Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', 's', '▁.', "'", '▁la', '▁le', '▁et', '▁l', '▁à']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
Training lm from:  [PosixPath('/home/test/workspace/ulmfit-multilingual/data/wiki/fr-100/models/sp15k/qrnn_nl4.m/lm_best'), PosixPath('/home/test/workspace/ulmfit-multilingual/data/wiki/fr-100/models/sp15k/qrnn_nl4.m/../itos')]
epoch     train_loss  valid_loss  accuracy
1         4.408209    3.734384    0.451908

Total time: 02:09
epoch     train_loss  valid_loss  accuracy
1         3.911323    3.613580    0.472285

2         3.678445    3.445187    0.503284

3         3.489969    3.296292    0.528945

4         3.381153    3.180339    0.549296

5         3.291956    3.100773    0.562257

6         3.184217    3.027092    0.575632

7         3.215341    2.965142    0.586544

8         3.119935    2.915341    0.596039

9         3.081539    2.870155    0.605426

10        3.096917    2.826453    0.614306

11        3.024909    2.786344    0.622573

12        2.940282    2.743246    0.632480

13        2.939400    2.713417    0.639012

14        2.920471    2.682074    0.646323

15        2.836955    2.652518    0.653338

16        2.873827    2.631899    0.658241

17        2.847641    2.615588    0.662295

18        2.856253    2.605571    0.664535

19        2.817845    2.601670    0.665505

20        2.827199    2.600590    0.665696

Total time: 1:13:04
/home/test/workspace/ulmfit-multilingual/data/mldoc/fr-1/models/sp15k
Saving info /home/test/workspace/ulmfit-multilingual/data/mldoc/fr-1/models/sp15k/qrnn_nl4-sl.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.830361    0.645785    0.913000

2         0.718334    0.713889    0.903000

3         0.625460    0.608466    0.936000

4         0.549874    0.573567    0.938000

5         0.513573    0.563112    0.938000

6         0.497791    0.559489    0.948000

7         0.482823    0.547815    0.944000

8         0.473484    0.543763    0.946000

Total time: 02:36
Saving models at /home/test/workspace/ulmfit-multilingual/data/mldoc/fr-1/models/sp15k/qrnn_nl4-sl.m
Loss and accuracy using (cls_best): [0.21287616, tensor(0.9480)]

Processing data/wiki/it-100/models/sp15k/qrnn_nl4.m
../mldoc/it-1
Max vocab: 15000
Cache dir: /home/test/workspace/ulmfit-multilingual/data/mldoc/it-1/models/sp15k
Model dir: /home/test/workspace/ulmfit-multilingual/data/mldoc/it-1/models/sp15k/qrnn_nl4-sl.m
Training
Loading validation /home/test/workspace/ulmfit-multilingual/data/mldoc/it-1/it.dev.csv
Running tokenization lm...
Data lm, trn: 13500, val: 1500

Running tokenization cls...
Data cls, trn: 1000, val: 1000

Running tokenization tst...
Data tst, trn: 1000, val: 4000

Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁.', '▁di', '▁e', "▁&'", "'", '▁il', '▁la', '▁in', 'e']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
Training lm from:  [PosixPath('/home/test/workspace/ulmfit-multilingual/data/wiki/it-100/models/sp15k/qrnn_nl4.m/lm_best'), PosixPath('/home/test/workspace/ulmfit-multilingual/data/wiki/it-100/models/sp15k/qrnn_nl4.m/../itos')]
epoch     train_loss  valid_loss  accuracy
1         4.802729    3.930606    0.435333

Total time: 01:03
epoch     train_loss  valid_loss  accuracy
1         4.208684    3.763728    0.460475

2         3.899941    3.558932    0.495172

3         3.627373    3.365080    0.528145

4         3.479348    3.217953    0.552994

5         3.345057    3.106383    0.571176

6         3.203966    3.013706    0.587515

7         3.159807    2.928730    0.601662

8         3.123686    2.863181    0.613407

9         3.091257    2.802755    0.625887

10        2.993412    2.747775    0.636441

11        2.923112    2.694130    0.647843

12        2.910893    2.644428    0.658610

13        2.912500    2.606808    0.667447

14        2.809009    2.564682    0.676342

15        2.813561    2.530405    0.684753

16        2.768533    2.505601    0.691209

17        2.714267    2.487358    0.695374

18        2.704343    2.474279    0.698375

19        2.721650    2.469191    0.699875

20        2.692483    2.468237    0.700245

Total time: 43:31
/home/test/workspace/ulmfit-multilingual/data/mldoc/it-1/models/sp15k
Saving info /home/test/workspace/ulmfit-multilingual/data/mldoc/it-1/models/sp15k/qrnn_nl4-sl.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.926403    0.718438    0.853000

2         0.791397    0.850076    0.824000

3         0.710437    0.707848    0.902000

4         0.626952    0.700860    0.882000

5         0.551851    0.648725    0.900000

6         0.527778    0.632797    0.906000

7         0.502474    0.621409    0.911000

8         0.489953    0.621797    0.910000

Total time: 01:32
Saving models at /home/test/workspace/ulmfit-multilingual/data/mldoc/it-1/models/sp15k/qrnn_nl4-sl.m
Loss and accuracy using (cls_best): [0.3240368, tensor(0.9005)]

Processing data/wiki/ja-100/models/sp15k/qrnn_nl4.m
../mldoc/ja-1
Max vocab: 15000
Cache dir: /home/test/workspace/ulmfit-multilingual/data/mldoc/ja-1/models/sp15k
Model dir: /home/test/workspace/ulmfit-multilingual/data/mldoc/ja-1/models/sp15k/qrnn_nl4-sl.m
Training
Loading validation /home/test/workspace/ulmfit-multilingual/data/mldoc/ja-1/ja.dev.csv
Running tokenization lm...
Data lm, trn: 13500, val: 1500

Running tokenization cls...
Data cls, trn: 1000, val: 1000

Running tokenization tst...
Data tst, trn: 1000, val: 4000

Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁、', '▁。', '▁の', '▁に', '▁を', '▁年', 'の', '▁は', '▁・', '▁)']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
Training lm from:  [PosixPath('/home/test/workspace/ulmfit-multilingual/data/wiki/ja-100/models/sp15k/qrnn_nl4.m/lm_best'), PosixPath('/home/test/workspace/ulmfit-multilingual/data/wiki/ja-100/models/sp15k/qrnn_nl4.m/../itos')]
epoch     train_loss  valid_loss  accuracy
1         4.467716    3.639215    0.510628

Total time: 02:43
epoch     train_loss  valid_loss  accuracy
1         3.640961    3.393765    0.547888

2         3.245662    3.111493    0.597234

3         3.091152    2.893684    0.636629

4         2.874470    2.753393    0.660614

5         2.774781    2.660047    0.677299

6         2.818495    2.584401    0.690161

7         2.763403    2.525782    0.699487

8         2.689764    2.481472    0.708918

9         2.471829    2.443742    0.715523

10        2.558768    2.411052    0.722205

11        2.583986    2.380159    0.728743

12        2.416061    2.352447    0.734377

13        2.422695    2.327425    0.739690

14        2.447176    2.302086    0.745426

15        2.409782    2.280813    0.749981

16        2.431124    2.265426    0.753981

17        2.409947    2.255584    0.756481

18        2.426040    2.246758    0.758470

19        2.363041    2.244397    0.759102

20        2.397845    2.243302    0.759366

Total time: 1:20:37
/home/test/workspace/ulmfit-multilingual/data/mldoc/ja-1/models/sp15k
Saving info /home/test/workspace/ulmfit-multilingual/data/mldoc/ja-1/models/sp15k/qrnn_nl4-sl.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.962460    0.721570    0.861000

2         0.850875    0.732539    0.873000

3         0.733097    0.733598    0.880000

4         0.639531    0.743423    0.882000

5         0.570058    0.702896    0.870000

6         0.525673    0.663320    0.892000

7         0.514369    0.668241    0.887000

8         0.500725    0.663006    0.886000

Total time: 03:16
Saving models at /home/test/workspace/ulmfit-multilingual/data/mldoc/ja-1/models/sp15k/qrnn_nl4-sl.m
Loss and accuracy using (cls_best): [0.32758784, tensor(0.8988)]

Processing data/wiki/ru-100/models/sp15k/qrnn_nl4.m
../mldoc/ru-1
Max vocab: 15000
Cache dir: /home/test/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp15k
Model dir: /home/test/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp15k/qrnn_nl4-sl.m
Training
Loading validation /home/test/workspace/ulmfit-multilingual/data/mldoc/ru-1/ru.dev.csv
Running tokenization lm...
Data lm, trn: 9195, val: 1021

Running tokenization cls...
Data cls, trn: 1000, val: 1000

Running tokenization tst...
Data tst, trn: 1000, val: 4000

Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁.', '▁в', 'а', 'и', 'е', '▁и', 'й', '▁на', 'х']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
Training lm from:  [PosixPath('/home/test/workspace/ulmfit-multilingual/data/wiki/ru-100/models/sp15k/qrnn_nl4.m/lm_best'), PosixPath('/home/test/workspace/ulmfit-multilingual/data/wiki/ru-100/models/sp15k/qrnn_nl4.m/../itos')]
epoch     train_loss  valid_loss  accuracy
1         4.599347    3.756284    0.476954

Total time: 02:08
epoch     train_loss  valid_loss  accuracy
1         3.814071    3.501726    0.524080

2         3.543633    3.248534    0.571844

3         3.245142    3.051544    0.607123

4         3.074194    2.917155    0.630383

5         3.015727    2.814585    0.648273

6         2.936803    2.731585    0.663555

7         2.801445    2.659986    0.676864

8         2.829947    2.602137    0.688181

9         2.784838    2.547982    0.699379

10        2.705119    2.501527    0.709368

11        2.763923    2.456791    0.719173

12        2.597343    2.411362    0.730358

13        2.647648    2.374054    0.738579

14        2.527706    2.336248    0.747234

15        2.543835    2.306177    0.755421

16        2.478718    2.282218    0.761426

17        2.563173    2.261219    0.766674

18        2.480387    2.251179    0.769268

19        2.415822    2.244734    0.770911

20        2.459630    2.243680    0.771312

Total time: 1:04:34
/home/test/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp15k
Saving info /home/test/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp15k/qrnn_nl4-sl.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.010382    0.760864    0.848000

2         0.885819    0.800701    0.835000

3         0.768587    0.828160    0.844000

4         0.698592    0.751787    0.857000

5         0.620826    0.767858    0.856000

6         0.563309    0.727524    0.859000

7         0.524067    0.700363    0.878000

8         0.499860    0.707688    0.871000

Total time: 03:40
Saving models at /home/test/workspace/ulmfit-multilingual/data/mldoc/ru-1/models/sp15k/qrnn_nl4-sl.m
Loss and accuracy using (cls_best): [0.40361115, tensor(0.8717)]

Processing data/wiki/zh-100/models/sp15k/qrnn_nl4.m
../mldoc/zh-1
Max vocab: 15000
Cache dir: /home/test/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp15k
Model dir: /home/test/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp15k/qrnn_nl4-sl.m
Training
Loading validation /home/test/workspace/ulmfit-multilingual/data/mldoc/zh-1/zh.dev.csv
Running tokenization lm...
Data lm, trn: 13500, val: 1500

Running tokenization cls...
Data cls, trn: 1000, val: 1000

Running tokenization tst...
Data tst, trn: 1000, val: 4000

Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁的', '▁。', '▁年', '▁、', '▁在', '▁一', '▁是', '▁中', '▁有']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
Training lm from:  [PosixPath('/home/test/workspace/ulmfit-multilingual/data/wiki/zh-100/models/sp15k/qrnn_nl4.m/lm_best'), PosixPath('/home/test/workspace/ulmfit-multilingual/data/wiki/zh-100/models/sp15k/qrnn_nl4.m/../itos')]
epoch     train_loss  valid_loss  accuracy
1         3.804344    3.253225    0.571249

Total time: 02:19
epoch     train_loss  valid_loss  accuracy
1         3.309380    3.061855    0.600547

2         3.055568    2.865146    0.636541

3         2.877304    2.710331    0.663194

4         2.749444    2.604171    0.681028

5         2.715787    2.528694    0.693581

6         2.664271    2.477576    0.702474

7         2.598713    2.412279    0.715434

8         2.540510    2.367390    0.724008

9         2.499321    2.330683    0.731755

10        2.513472    2.290408    0.740227

11        2.397077    2.248312    0.749950

12        2.425433    2.212132    0.757908

13        2.364556    2.176752    0.767242

14        2.349984    2.142507    0.775855

15        2.321824    2.119729    0.781726

16        2.313458    2.095738    0.788297

17        2.239505    2.078650    0.792735

18        2.240292    2.069656    0.795083

19        2.250233    2.064083    0.796754

20        2.251804    2.063307    0.797006

Total time: 1:10:47
/home/test/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp15k
Saving info /home/test/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp15k/qrnn_nl4-sl.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.920672    0.685620    0.870000

2         0.773749    0.700495    0.907000

3         0.663505    0.669162    0.908000

4         0.591225    0.621341    0.915000

5         0.542783    0.622516    0.919000

6         0.517919    0.608709    0.911000

7         0.489793    0.605009    0.918000

8         0.476783    0.597071    0.916000

Total time: 02:37
Saving models at /home/test/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp15k/qrnn_nl4-sl.m
Loss and accuracy using (cls_best): [0.29685277, tensor(0.9190)]

OrderedDict([('data/mldoc/de-1/models/sp15k/qrnn_nl4-sl.m', 0.9610000252723694),
             ('data/mldoc/en-1/models/sp15k/qrnn_nl4-sl.m', 0.9555000066757202),
             ('data/mldoc/es-1/models/sp15k/qrnn_nl4-sl.m', 0.9582499861717224),
             ('data/mldoc/fr-1/models/sp15k/qrnn_nl4-sl.m', 0.9480000138282776),
             ('data/mldoc/it-1/models/sp15k/qrnn_nl4-sl.m', 0.9004999995231628),
             ('data/mldoc/ja-1/models/sp15k/qrnn_nl4-sl.m', 0.8987500071525574),
             ('data/mldoc/ru-1/models/sp15k/qrnn_nl4-sl.m', 0.871749997138977),
             ('data/mldoc/zh-1/models/sp15k/qrnn_nl4-sl.m',
              0.9190000295639038)])
data/mldoc/de-1/models/sp15k/qrnn_nl4-sl.m: 0.9610000252723694
data/mldoc/en-1/models/sp15k/qrnn_nl4-sl.m: 0.9555000066757202
data/mldoc/es-1/models/sp15k/qrnn_nl4-sl.m: 0.9582499861717224
data/mldoc/fr-1/models/sp15k/qrnn_nl4-sl.m: 0.9480000138282776
data/mldoc/it-1/models/sp15k/qrnn_nl4-sl.m: 0.9004999995231628
data/mldoc/ja-1/models/sp15k/qrnn_nl4-sl.m: 0.8987500071525574
data/mldoc/ru-1/models/sp15k/qrnn_nl4-sl.m: 0.871749997138977
data/mldoc/zh-1/models/sp15k/qrnn_nl4-sl.m: 0.9190000295639038
```


## CLS training
### all 
```
python -m ulmfit eval --glob="mldoc/*-1/models/sp15k/qrnn_nl4.m" --name nl4-1cyc-sl-e4  --num-cls-epochs=4 --bs=18 --lr_sched=1cycle --label-smoothing-eps=0.1
Processing data/mldoc/de-1/models/sp15k/qrnn_nl4.m
de-1
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1/models/sp15k/qrnn_nl4-1cyc-sl-e4.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1/de.dev.csv
Data lm, trn: 13500, val: 1500
Data cls, trn: 1000, val: 1000
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
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1/models/sp15k/qrnn_nl4-1cyc-sl-e4.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.806209    0.620243    0.938000
2         0.643088    0.608909    0.944000
3         0.552762    0.577317    0.943000
4         0.506282    0.566124    0.944000
Total time: 01:09
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1/models/sp15k/qrnn_nl4-1cyc-sl-e4.m
Loss and accuracy using (cls_best): [0.18408646, tensor(0.9597)]
Processing data/mldoc/en-1/models/sp15k/qrnn_nl4.m
en-1
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/en-1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/en-1/models/sp15k/qrnn_nl4-1cyc-sl-e4.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/en-1/en.dev.csv
Data lm, trn: 13500, val: 1500
Data cls, trn: 1000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁the', '▁,', 's', '▁.', '▁of', '▁and', '▁in', '▁to', '▁a', 'ed']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/en-1/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/en-1/models/sp15k/qrnn_nl4-1cyc-sl-e4.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.856259    0.637753    0.935000
2         0.713970    0.627611    0.918000
3         0.594603    0.550310    0.947000
4         0.526848    0.549133    0.954000
Total time: 01:15
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/en-1/models/sp15k/qrnn_nl4-1cyc-sl-e4.m
Loss and accuracy using (cls_best): [0.20320596, tensor(0.9500)]
Processing data/mldoc/es-1/models/sp15k/qrnn_nl4.m
es-1
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp15k/qrnn_nl4-1cyc-sl-e4.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/es.dev.csv
Data lm, trn: 13013, val: 1445
Data cls, trn: 1000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁la', '▁.', 's', '▁en', '▁el', '▁y', '▁a', '▁que']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp15k/qrnn_nl4-1cyc-sl-e4.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.804829    0.636135    0.906000
2         0.712160    0.579012    0.952000
3         0.605291    0.543731    0.965000
4         0.531676    0.546145    0.966000
Total time: 01:01
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1/models/sp15k/qrnn_nl4-1cyc-sl-e4.m
Loss and accuracy using (cls_best): [0.18462537, tensor(0.9565)]
Processing data/mldoc/fr-1/models/sp15k/qrnn_nl4.m
fr-1
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/models/sp15k/qrnn_nl4-1cyc-sl-e4.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/fr.dev.csv
Data lm, trn: 13500, val: 1500
Data cls, trn: 1000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', 's', '▁.', "'", '▁la', '▁le', '▁et', '▁l', '▁à']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/models/sp15k/qrnn_nl4-1cyc-sl-e4.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.835704    0.657239    0.902000
2         0.688311    0.679450    0.924000
3         0.575970    0.579612    0.938000
4         0.515154    0.565664    0.939000
Total time: 01:11
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1/models/sp15k/qrnn_nl4-1cyc-sl-e4.m
Loss and accuracy using (cls_best): [0.20844615, tensor(0.9435)]
Processing data/mldoc/it-1/models/sp15k/qrnn_nl4.m
it-1
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1/models/sp15k/qrnn_nl4-1cyc-sl-e4.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1/it.dev.csv
Data lm, trn: 13500, val: 1500
Data cls, trn: 1000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁.', '▁di', '▁e', "▁&'", "'", '▁il', '▁la', '▁in', 'e']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1/models/sp15k/qrnn_nl4-1cyc-sl-e4.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.914871    0.758207    0.848000
2         0.799002    0.692626    0.877000
3         0.658110    0.646415    0.888000
4         0.567358    0.629301    0.912000
Total time: 00:42
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1/models/sp15k/qrnn_nl4-1cyc-sl-e4.m
Loss and accuracy using (cls_best): [0.30882642, tensor(0.9032)]
Processing data/mldoc/ja-1/models/sp15k/qrnn_nl4.m
ja-1
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ja-1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ja-1/models/sp15k/qrnn_nl4-1cyc-sl-e4.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ja-1/ja.dev.csv
Data lm, trn: 13500, val: 1500
Data cls, trn: 1000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁、', '▁。', '▁の', '▁に', '▁を', '▁年', 'の', '▁は', '▁・', '▁)']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ja-1/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ja-1/models/sp15k/qrnn_nl4-1cyc-sl-e4.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.959891    0.771793    0.834000
2         0.813159    0.684481    0.889000
3         0.675869    0.698423    0.878000
4         0.580597    0.689197    0.881000
Total time: 01:24
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ja-1/models/sp15k/qrnn_nl4-1cyc-sl-e4.m
Loss and accuracy using (cls_best): [0.3306819, tensor(0.8967)]
Processing data/mldoc/zh-1/models/sp15k/qrnn_nl4.m
zh-1
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp15k/qrnn_nl4-1cyc-sl-e4.m
Evaluating previously trained model
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/zh.dev.csv
Data lm, trn: 13500, val: 1500
Data cls, trn: 1000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁的', '▁。', '▁年', '▁、', '▁在', '▁一', '▁是', '▁中', '▁有']
Loss and accuracy using (cls_best): [0.28541276, tensor(0.9237)]
OrderedDict([('data/mldoc/de-1/models/sp15k/qrnn_nl4-1cyc-sl-e4.m',
              0.9597499966621399),
             ('data/mldoc/en-1/models/sp15k/qrnn_nl4-1cyc-sl-e4.m',
              0.949999988079071),
             ('data/mldoc/es-1/models/sp15k/qrnn_nl4-1cyc-sl-e4.m',
              0.9564999938011169),
             ('data/mldoc/fr-1/models/sp15k/qrnn_nl4-1cyc-sl-e4.m',
              0.9434999823570251),
             ('data/mldoc/it-1/models/sp15k/qrnn_nl4-1cyc-sl-e4.m',
              0.903249979019165),
             ('data/mldoc/ja-1/models/sp15k/qrnn_nl4-1cyc-sl-e4.m',
              0.8967499732971191),
             ('data/mldoc/zh-1/models/sp15k/qrnn_nl4-1cyc-sl-e4.m',
              0.9237499833106995)])
data/mldoc/de-1/models/sp15k/qrnn_nl4-1cyc-sl-e4.m: 0.9597499966621399
data/mldoc/en-1/models/sp15k/qrnn_nl4-1cyc-sl-e4.m: 0.949999988079071
data/mldoc/es-1/models/sp15k/qrnn_nl4-1cyc-sl-e4.m: 0.9564999938011169
data/mldoc/fr-1/models/sp15k/qrnn_nl4-1cyc-sl-e4.m: 0.9434999823570251
data/mldoc/it-1/models/sp15k/qrnn_nl4-1cyc-sl-e4.m: 0.903249979019165
data/mldoc/ja-1/models/sp15k/qrnn_nl4-1cyc-sl-e4.m: 0.8967499732971191
data/mldoc/zh-1/models/sp15k/qrnn_nl4-1cyc-sl-e4.m: 0.9237499833106995
```
### ZH
Exec 1
```
python -m ulmfit eval --glob="mldoc/zh-1/models/sp15k/qrnn_nl4.m" --name nl4-1cyc-sl  --num-cls-epochs=8 --bs=18 --lr_sched=1cycle --label-smoothing-eps=0.1
Processing data/mldoc/zh-1/models/sp15k/qrnn_nl4.m
zh-1
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp15k/qrnn_nl4-1cyc-sl.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/zh.dev.csv
Data lm, trn: 13500, val: 1500
Data cls, trn: 1000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁的', '▁。', '▁年', '▁、', '▁在', '▁一', '▁是', '▁中', '▁有']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
/home/pczapla/anaconda3/envs/fastaiv1/lib/python3.7/site-packages/torch/utils/cpp_extension.py:152: UserWarning:
Loading pretrained model
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp15k/qrnn_nl4-1cyc-sl.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.859153    0.767961    0.864000
2         0.768888    0.775161    0.904000
3         0.658956    0.685653    0.902000
4         0.589073    0.618438    0.923000
5         0.540008    0.622157    0.915000
6         0.508080    0.606979    0.914000
7         0.487228    0.599491    0.918000
8         0.477516    0.602196    0.923000
Total time: 02:18
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp15k/qrnn_nl4-1cyc-sl.m
Loss and accuracy using (cls_best): [0.2829206, tensor(0.9205)]
OrderedDict([('data/mldoc/zh-1/models/sp15k/qrnn_nl4-1cyc-sl.m',
              0.9204999804496765)])
data/mldoc/zh-1/models/sp15k/qrnn_nl4-1cyc-sl.m: 0.9204999804496765
```
Exec 2
````python -m ulmfit eval --glob="mldoc/zh-1/models/sp15k/qrnn_nl4.m" --name nl4-1cyc-sl1  --num-cls-epochs=8 --bs=18 --lr_sched=1cycle --label-smoothing-eps=0.1
Processing data/mldoc/zh-1/models/sp15k/qrnn_nl4.m
zh-1
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp15k/qrnn_nl4-1cyc-sl1.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/zh.dev.csv
Data lm, trn: 13500, val: 1500
Data cls, trn: 1000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁的', '▁。', '▁年', '▁、', '▁在', '▁一', '▁是', '▁中', '▁有']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp15k/qrnn_nl4-1cyc-sl1.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.881513    0.712176    0.865000
2         0.743687    0.665091    0.906000
3         0.677436    0.687689    0.873000
4         0.595139    0.626483    0.920000
5         0.542732    0.600652    0.914000
6         0.512080    0.597546    0.916000
7         0.487021    0.597065    0.912000
8         0.476598    0.596792    0.914000
Total time: 02:20
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp15k/qrnn_nl4-1cyc-sl1.m
Loss and accuracy using (cls_best): [0.29172945, tensor(0.9178)]
OrderedDict([('data/mldoc/zh-1/models/sp15k/qrnn_nl4-1cyc-sl1.m',
              0.9177500009536743)])
data/mldoc/zh-1/models/sp15k/qrnn_nl4-1cyc-sl1.m: 0.9177500009536743
````
Exec 4
```bash
python -m ulmfit eval --glob="mldoc/zh-1/models/sp15k/qrnn_nl4.m" --name nl4-1cyc-sl-e4  --num-cls-epochs=4 --bs=18 --lr_sched=1cycle --label-smoothing-eps=0.1               ✘ 130
Processing data/mldoc/zh-1/models/sp15k/qrnn_nl4.m
zh-1
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp15k/qrnn_nl4-1cyc-sl-e4.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/zh.dev.csv
Data lm, trn: 13500, val: 1500
Data cls, trn: 1000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁的', '▁。', '▁年', '▁、', '▁在', '▁一', '▁是', '▁中', '▁有']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp15k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp15k/qrnn_nl4-1cyc-sl-e4.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.880415    0.677291    0.901000
2         0.729670    0.659975    0.911000
3         0.624817    0.603056    0.921000
4         0.542027    0.601961    0.921000
Total time: 01:08
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp15k/qrnn_nl4-1cyc-sl-e4.m
Loss and accuracy using (cls_best): [0.28558904, tensor(0.9222)]
OrderedDict([('data/mldoc/zh-1/models/sp15k/qrnn_nl4-1cyc-sl-e4.m',
              0.922249972820282)])
data/mldoc/zh-1/models/sp15k/qrnn_nl4-1cyc-sl-e4.m: 0.922249972820282
```
## LSTM sp30k
### 0.1
```bash
 python -m ulmfit eval --glob="mldoc/zh-1/models/sp30k/lstm_nl4.m" --name nl4-1cyc-sl  --num-cls-epochs=8 --bs=18 --lr_sched=1cycle --label-smoothing-eps=0.1
Processing data/mldoc/zh-1/models/sp30k/lstm_nl4.m
zh-1
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp30k/lstm_nl4-1cyc-sl.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/zh.dev.csv
Data lm, trn: 13500, val: 1500
Data cls, trn: 1000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁的', '▁。', '▁年', '▁、', '▁在', '▁一', '▁中', '▁人', '▁是']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp30k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp30k/lstm_nl4-1cyc-sl.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.870432    0.670671    0.882000
2         0.754248    0.824157    0.895000
3         0.654601    0.727428    0.885000
4         0.602772    0.668668    0.901000
5         0.542110    0.625137    0.903000
6         0.506150    0.617842    0.913000
7         0.480944    0.616885    0.912000
8         0.472876    0.614381    0.911000
Total time: 06:38
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp30k/lstm_nl4-1cyc-sl.m
Loss and accuracy using (cls_best): [0.2977172, tensor(0.9233)]
OrderedDict([('data/mldoc/zh-1/models/sp30k/lstm_nl4-1cyc-sl.m',
              0.9232500195503235)])
data/mldoc/zh-1/models/sp30k/lstm_nl4-1cyc-sl.m: 0.9232500195503235
```
### 0.2
```bash
python -m ulmfit eval --glob="mldoc/zh-1/models/sp30k/lstm_nl4.m" --name nl4-1cyc-sl2  --num-cls-epochs=8 --bs=18 --lr_sched=1cycle --label-smoothing-eps=0.2
Processing data/mldoc/zh-1/models/sp30k/lstm_nl4.m
zh-1
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp30k/lstm_nl4-1cyc-sl2.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/zh.dev.csv
Data lm, trn: 13500, val: 1500
Data cls, trn: 1000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁的', '▁。', '▁年', '▁、', '▁在', '▁一', '▁中', '▁人', '▁是']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp30k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp30k/lstm_nl4-1cyc-sl2.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.045619    0.908213    0.874000
2         0.957379    0.857977    0.921000
3         0.891791    0.852157    0.905000
4         0.845289    0.849923    0.914000
5         0.818228    0.848613    0.921000
6         0.787021    0.840483    0.920000
7         0.776123    0.844006    0.919000
8         0.762384    0.857240    0.916000
Total time: 06:33
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp30k/lstm_nl4-1cyc-sl2.m
Loss and accuracy using (cls_best): [0.40299156, tensor(0.9170)]
OrderedDict([('data/mldoc/zh-1/models/sp30k/lstm_nl4-1cyc-sl2.m',
              0.9169999957084656)])
data/mldoc/zh-1/models/sp30k/lstm_nl4-1cyc-sl2.m: 0.9169999957084656
```
### 0.4
```bash
 python -m ulmfit eval --glob="mldoc/zh-1/models/sp30k/lstm_nl4.m" --name nl4-1cyc-sl4  --num-cls-epochs=8 --bs=18 --lr_sched=1cycle --label-smoothing-eps=0.4
Processing data/mldoc/zh-1/models/sp30k/lstm_nl4.m
zh-1
Max vocab: 30000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp30k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp30k/lstm_nl4-1cyc-sl4.m
Training
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/zh.dev.csv
Data lm, trn: 13500, val: 1500
Data cls, trn: 1000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 30000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁的', '▁。', '▁年', '▁、', '▁在', '▁一', '▁中', '▁人', '▁是']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0.3} dps:  {'output_p': 0.25, 'hidden_p': 0.1, 'input_p': 0.2, 'embed_p': 0.02, 'weight_p': 0.15}
Loading pretrained model
Unknown tokens 0, first 100: []
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp30k
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp30k/lstm_nl4-1cyc-sl4.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         1.251581    1.183341    0.898000
2         1.214358    1.201266    0.834000
3         1.190343    1.165525    0.919000
4         1.168018    1.172510    0.903000
5         1.149965    1.161660    0.914000
6         1.140140    1.161689    0.915000
7         1.135877    1.159853    0.912000
8         1.134425    1.160039    0.911000
Total time: 06:34
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1/models/sp30k/lstm_nl4-1cyc-sl4.m
Loss and accuracy using (cls_best): [0.64041936, tensor(0.9195)]
OrderedDict([('data/mldoc/zh-1/models/sp30k/lstm_nl4-1cyc-sl4.m',
              0.9194999933242798)])
data/mldoc/zh-1/models/sp30k/lstm_nl4-1cyc-sl4.m: 0.9194999933242798
```