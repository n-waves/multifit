```
python -m ulmfit cls --dataset-path data/mldoc/${LANG}-1 --tokenizer sp --max-vocab 16000 --qrnn True --lang=${LANG} --name nl4 - train 20 --bs 18 --num-cls-epochs=4 --lr_sched=1cycle --label-smoothing-eps=0.1
Max vocab: 16000
Cache dir: data/mldoc/ru-1/models/sp16k
Model dir: data/mldoc/ru-1/models/sp16k/qrnn_nl4.m
Loading validation data/mldoc/ru-1/ru.dev.csv
/sentencepiece/src/sentencepiece_trainer.cc(185) LOG(INFO) Running command: --input=data/mldoc/ru-1/models/sp16k/all_text.txt --character_coverage=0.99 --unk_id=8 --pad_id=-1 --bos_id=-1 --eos_id=-1 --max_sentence_length=20480 --input_sentence_size=10000000 --user_defined_symbols=xxunk,xxpad,xxbos,xxfld,xxmaj,xxup,xxrep,xxwrep --model_prefix=data/mldoc/ru-1/models/sp16k/spm --vocab_size=16000 --model_type=unigram
/sentencepiece/src/unigram_model_trainer.cc(481) LOG(INFO) Starts training with :
input: "data/mldoc/ru-1/models/sp16k/all_text.txt"
model_prefix: "data/mldoc/ru-1/models/sp16k/spm"
model_type: UNIGRAM
vocab_size: 16000
character_coverage: 0.99
input_sentence_size: 10000000
max_sentence_length: 20480
user_defined_symbols: "xxunk"
user_defined_symbols: "xxpad"
user_defined_symbols: "xxbos"
user_defined_symbols: "xxfld"
user_defined_symbols: "xxmaj"
user_defined_symbols: "xxup"
user_defined_symbols: "xxrep"
user_defined_symbols: "xxwrep"
unk_id: 8
bos_id: -1
eos_id: -1
pad_id: -1

/sentencepiece/src/trainer_interface.cc(183) LOG(INFO) Loading corpus: data/mldoc/ru-1/models/sp16k/all_text.txt
/sentencepiece/src/trainer_interface.cc(216) LOG(INFO) Loading: ▁	▁киев▁,▁20▁июн▁(▁	▁рейтер▁)▁-▁	▁нацбанк▁	▁украины▁планирует▁постепенно▁отказаться▁от▁кредитных▁аукционов▁и▁использовать▁для▁рефинансирования▁банков▁только▁операции▁репо▁и▁ломбардное▁кредитование▁,▁сказала▁директор▁департамента▁	▁нбу▁	▁наталия▁	▁гребеник▁.▁&'▁	▁от▁кредитных▁аукционов▁	▁нбу▁будет▁в▁дальнейшем▁отказываться▁,▁используя▁репо▁и▁ломбардное▁кредитование▁&'▁,▁-▁сказала▁директор▁кредитно-эмиссионного▁департамента▁.▁	▁по▁ее▁словам▁,▁в▁настоящее▁время▁	▁нацбанк▁использует▁все▁три▁канала▁рефинансирования▁банков▁.▁	▁удельный▁вес▁рефинансирования▁через▁операции▁репо▁составляет▁50▁процентов▁,▁через▁кредитные▁аукционы▁и▁ломбардное▁кредитование▁под▁залог▁гособлигаций▁по▁25▁процентов▁.▁в▁частности▁,▁с▁начала▁года▁были▁проведены▁четыре▁кредитных▁аукционах▁на▁которых▁банкам▁было▁продано▁560▁миллионов▁гривен▁кредитов▁,▁сказала▁	▁гребеник▁.▁	▁по▁ее▁словам▁,▁средняя▁ставка▁продажи▁ресурсов▁на▁кредитных▁аукционах▁на▁3-4▁процента▁превышала▁ставку▁рефинансирования▁,▁действующую▁на▁день▁проведения▁аукциона▁.▁	▁действующая▁в▁настоящее▁время▁ставка▁рефинансирования▁	▁нбу▁составляет▁21▁процент▁годовых▁,▁ломбардная▁ставка▁-▁31▁процент▁.▁	▁по▁соглашениям▁репо▁ставка▁может▁быть▁ниже▁ставки▁рефинансирования▁,▁но▁не▁более▁,▁чем▁на▁5▁процентных▁пунктов▁,▁сказал▁	▁гребеник▁.▁	▁по▁ее▁словам▁,▁в▁будущем▁	▁нбу▁также▁планирует▁освоить▁инструмент▁векселей▁при▁рефинансировании▁коммерческих▁банков▁.▁&'▁	▁мы▁будем▁переходить▁к▁использованию▁векселей▁как▁залога▁,▁что▁даст▁нам▁возможность▁более▁четко▁определять▁стоимость▁денежных▁ресурсов▁&'▁,▁-▁сказала▁	▁гребеник▁.▁-▁	▁наталия▁	▁зинец▁,▁	▁киевское▁бюро▁,▁(▁044▁)▁244▁9150▁.▁(▁c▁)▁	▁reuters▁	▁limited▁1997▁.	size=0
/sentencepiece/src/trainer_interface.cc(200) LOG(INFO) Too long lines (>=20480 bytes (it can be changed with --max_sentence_length flag). Skipped.
/sentencepiece/src/trainer_interface.cc(200) LOG(INFO) Too long lines (>=20480 bytes (it can be changed with --max_sentence_length flag). Skipped.
/sentencepiece/src/trainer_interface.cc(240) LOG(INFO) Loaded 998 sentences
/sentencepiece/src/trainer_interface.cc(241) LOG(INFO) Loaded 0 test sentences
/sentencepiece/src/trainer_interface.cc(265) LOG(INFO) all chars count=1565524
/sentencepiece/src/trainer_interface.cc(273) LOG(INFO) Done: 99.1426% characters are covered.
/sentencepiece/src/trainer_interface.cc(283) LOG(INFO) Alphabet size=68
/sentencepiece/src/trainer_interface.cc(284) LOG(INFO) Final character coverage=0.991426
/sentencepiece/src/trainer_interface.cc(316) LOG(INFO) Done! 998 sentences are loaded
/sentencepiece/src/unigram_model_trainer.cc(127) LOG(INFO) Using 998 sentences for making seed sentencepieces
/sentencepiece/src/unigram_model_trainer.cc(155) LOG(INFO) Making suffix array...
/sentencepiece/src/unigram_model_trainer.cc(159) LOG(INFO) Extracting frequent sub strings...
/sentencepiece/src/unigram_model_trainer.cc(210) LOG(INFO) Initialized 67755 seed sentencepieces
/sentencepiece/src/trainer_interface.cc(322) LOG(INFO) Tokenizing input sentences with whitespace: 998
/sentencepiece/src/trainer_interface.cc(331) LOG(INFO) Done! 31975
/sentencepiece/src/unigram_model_trainer.cc(502) LOG(INFO) Using 31975 sentences for EM training
/sentencepiece/src/unigram_model_trainer.cc(518) LOG(INFO) EM sub_iter=0 size=22877 obj=16.6184 num_tokens=70560 num_tokens/piece=3.08432
/sentencepiece/src/unigram_model_trainer.cc(518) LOG(INFO) EM sub_iter=1 size=19595 obj=14.256 num_tokens=71915 num_tokens/piece=3.67007
/sentencepiece/src/unigram_model_trainer.cc(518) LOG(INFO) EM sub_iter=0 size=17579 obj=14.2013 num_tokens=73054 num_tokens/piece=4.15575
/sentencepiece/src/unigram_model_trainer.cc(518) LOG(INFO) EM sub_iter=1 size=17469 obj=14.1655 num_tokens=73390 num_tokens/piece=4.20116
/sentencepiece/src/trainer_interface.cc(387) LOG(INFO) Saving model: data/mldoc/ru-1/models/sp16k/spm.model
/sentencepiece/src/trainer_interface.cc(411) LOG(INFO) Saving vocabs: data/mldoc/ru-1/models/sp16k/spm.vocab
Running tokenization lm...
Data lm, trn: 9195, val: 1021
Running tokenization cls...
Data cls, trn: 1000, val: 1000
Running tokenization tst...
Data tst, trn: 1000, val: 4000
Size of vocabulary: 16000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', ',', '▁.', 'и', 'е', '▁в', 'й', '▁-', 'а', ')', '(']
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
Bptt 70
Training lm from random weights
epoch     train_loss  valid_loss  accuracy
1         4.311670    4.217185    0.471835
2         3.735446    3.595414    0.552916
3         3.553164    3.354127    0.581794
4         3.363475    3.259169    0.593828
5         3.514256    3.261860    0.590224
6         3.413725    3.223500    0.597156
7         3.453391    3.182702    0.601941
8         3.317564    3.131130    0.610511
9         3.398653    3.092810    0.616117
10        3.276093    3.037282    0.624851
11        3.207109    2.980038    0.634575
12        3.141415    2.928465    0.643130
13        3.164837    2.878245    0.653095
14        3.093078    2.823911    0.662821
15        3.026668    2.770853    0.673216
16        2.968236    2.723534    0.682577
17        2.983422    2.690081    0.689747
18        2.862256    2.666973    0.694282
19        2.876733    2.656204    0.696821
20        2.853209    2.654935    0.696994
Total time: 39:43
data/mldoc/ru-1/models/sp16k
Saving info data/mldoc/ru-1/models/sp16k/qrnn_nl4.m/info.json
Single training schedule
epoch     train_loss  valid_loss  accuracy
1         0.993379    0.788644    0.807000
2         0.832733    0.773031    0.864000
3         0.706515    0.715565    0.864000
4         0.618606    0.720445    0.868000
Total time: 00:56
Saving models at data/mldoc/ru-1/models/sp16k/qrnn_nl4.m
Loss and accuracy using (cls_best): [0.39357555, tensor(0.8685)]
0.3935755491256714
0.8684999942779541
```