````
LANG=ja                                                                                 ✘ 130
python -m multifit lm --dataset-path data/wiki/${LANG}-100 --tokenizer='fsp' --nl 4 --name 'nl4-1152' --max-vocab 15000 --lang ${LANG} --qrnn=True --lmseed=1  --nh=1152 - train 10 --bs=50 --drop_mult=0 --label-smoothing-eps=0.1
Training lm
Max vocab: 15000
Cache dir: data/wiki/ja-100/models/fsp15k
Model dir: data/wiki/ja-100/models/fsp15k/qrnn_nl4-1152_lmseed-1.m
Setting LM seed to 1
Wiki text was split to 120037 articles
Wiki text was split to 63 articles
Running tokenization lm...
sentencepiece_trainer.cc(116) LOG(INFO) Running command: --input=/home/pczapla/workspace/ulmfit-multilingual/data/wiki/ja-100/models/fsp15k/all_text.out --max_sentence_length=20480 --character_coverage=0.9998 --unk_id=9 --pad_id=-1 --bos_id=-1 --eos_id=-1 --user_defined_symbols=▁xxunk,▁xxpad,▁xxbos,▁xxeos,▁xxfld,▁xxmaj,▁xxup,▁xxrep,▁xxwrep --model_prefix=/home/pczapla/workspace/ulmfit-multilingual/data/wiki/ja-100/models/fsp15k/spm --vocab_size=15000 --model_type=unigram
sentencepiece_trainer.cc(49) LOG(INFO) Starts training with :
TrainerSpec {
  input: /home/pczapla/workspace/ulmfit-multilingual/data/wiki/ja-100/models/fsp15k/all_text.out
  input_format:
  model_prefix: /home/pczapla/workspace/ulmfit-multilingual/data/wiki/ja-100/models/fsp15k/spm
  model_type: UNIGRAM
  vocab_size: 15000
  self_test_sample_size: 0
  character_coverage: 0.9998
  input_sentence_size: 0
  shuffle_input_sentence: 1
  seed_sentencepiece_size: 1000000
  shrinking_factor: 0.75
  max_sentence_length: 20480
  num_threads: 16
  num_sub_iterations: 2
  max_sentencepiece_length: 16
  split_by_unicode_script: 1
  split_by_number: 1
  split_by_whitespace: 1
  treat_whitespace_as_suffix: 0
  user_defined_symbols: ▁xxunk
  user_defined_symbols: ▁xxpad
  user_defined_symbols: ▁xxbos
  user_defined_symbols: ▁xxeos
  user_defined_symbols: ▁xxfld
  user_defined_symbols: ▁xxmaj
  user_defined_symbols: ▁xxup
  user_defined_symbols: ▁xxrep
  user_defined_symbols: ▁xxwrep
  hard_vocab_limit: 1
  use_all_vocab: 0
  unk_id: 9
  bos_id: -1
  eos_id: -1
  pad_id: -1
  unk_piece: <unk>
  bos_piece: <s>
  eos_piece: </s>
  pad_piece: <pad>
  unk_surface:  ⁇
}
NormalizerSpec {
  name: nmt_nfkc
  add_dummy_prefix: 1
  remove_extra_whitespaces: 1
  escape_whitespaces: 1
  normalization_rule_tsv:
}

trainer_interface.cc(267) LOG(INFO) Loading corpus: /home/pczapla/workspace/ulmfit-multilingual/data/wiki/ja-100/models/fsp15k/all_text.out
trainer_interface.cc(287) LOG(WARNING) Found too long line (74468 > 20480).
trainer_interface.cc(289) LOG(WARNING) Too long lines are skipped in the training.
trainer_interface.cc(290) LOG(WARNING) The maximum length can be changed with --max_sentence_length=<size> flag.
trainer_interface.cc(315) LOG(INFO) Loaded all 115812 sentences
trainer_interface.cc(321) LOG(INFO) Skipped 4225 too long sentences.
trainer_interface.cc(330) LOG(INFO) Adding meta_piece: ▁xxunk
trainer_interface.cc(330) LOG(INFO) Adding meta_piece: ▁xxpad
trainer_interface.cc(330) LOG(INFO) Adding meta_piece: ▁xxbos
trainer_interface.cc(330) LOG(INFO) Adding meta_piece: ▁xxeos
trainer_interface.cc(330) LOG(INFO) Adding meta_piece: ▁xxfld
trainer_interface.cc(330) LOG(INFO) Adding meta_piece: ▁xxmaj
trainer_interface.cc(330) LOG(INFO) Adding meta_piece: ▁xxup
trainer_interface.cc(330) LOG(INFO) Adding meta_piece: ▁xxrep
trainer_interface.cc(330) LOG(INFO) Adding meta_piece: ▁xxwrep
trainer_interface.cc(330) LOG(INFO) Adding meta_piece: <unk>
trainer_interface.cc(335) LOG(INFO) Normalizing sentences...
trainer_interface.cc(385) LOG(INFO) all chars count=179924674
trainer_interface.cc(393) LOG(INFO) Done: 99.98% characters are covered.
trainer_interface.cc(403) LOG(INFO) Alphabet size=4440
trainer_interface.cc(404) LOG(INFO) Final character coverage=0.9998
trainer_interface.cc(436) LOG(INFO) Done! preprocessed 115812 sentences.
unigram_model_trainer.cc(129) LOG(INFO) Making suffix array...
unigram_model_trainer.cc(133) LOG(INFO) Extracting frequent sub strings...
unigram_model_trainer.cc(184) LOG(INFO) Initialized 1000000 seed sentencepieces
trainer_interface.cc(442) LOG(INFO) Tokenizing input sentences with whitespace: 115812
trainer_interface.cc(452) LOG(INFO) Done! 2256013
unigram_model_trainer.cc(470) LOG(INFO) Using 2256013 sentences for EM training
unigram_model_trainer.cc(486) LOG(INFO) EM sub_iter=0 size=589366 obj=9.67447 num_tokens=5565112 num_tokens/piece=9.44254
unigram_model_trainer.cc(486) LOG(INFO) EM sub_iter=1 size=499262 obj=8.54312 num_tokens=5565195 num_tokens/piece=11.1468
unigram_model_trainer.cc(486) LOG(INFO) EM sub_iter=0 size=374209 obj=8.48394 num_tokens=5655360 num_tokens/piece=15.1128
unigram_model_trainer.cc(486) LOG(INFO) EM sub_iter=1 size=373594 obj=8.47578 num_tokens=5655268 num_tokens/piece=15.1375
unigram_model_trainer.cc(486) LOG(INFO) EM sub_iter=0 size=280179 obj=8.57339 num_tokens=5837695 num_tokens/piece=20.8356
unigram_model_trainer.cc(486) LOG(INFO) EM sub_iter=1 size=280142 obj=8.57282 num_tokens=5839303 num_tokens/piece=20.8441
unigram_model_trainer.cc(486) LOG(INFO) EM sub_iter=0 size=210103 obj=8.62784 num_tokens=6061221 num_tokens/piece=28.8488
unigram_model_trainer.cc(486) LOG(INFO) EM sub_iter=1 size=210098 obj=8.62292 num_tokens=6062624 num_tokens/piece=28.8562
unigram_model_trainer.cc(486) LOG(INFO) EM sub_iter=0 size=157573 obj=8.72461 num_tokens=6293760 num_tokens/piece=39.9419
unigram_model_trainer.cc(486) LOG(INFO) EM sub_iter=1 size=157573 obj=8.71413 num_tokens=6295112 num_tokens/piece=39.9504
unigram_model_trainer.cc(486) LOG(INFO) EM sub_iter=0 size=118179 obj=8.87183 num_tokens=6531351 num_tokens/piece=55.2666
unigram_model_trainer.cc(486) LOG(INFO) EM sub_iter=1 size=118178 obj=8.86018 num_tokens=6532542 num_tokens/piece=55.2771
unigram_model_trainer.cc(486) LOG(INFO) EM sub_iter=0 size=88632 obj=9.09891 num_tokens=6780015 num_tokens/piece=76.4962
unigram_model_trainer.cc(486) LOG(INFO) EM sub_iter=1 size=88632 obj=9.08523 num_tokens=6781876 num_tokens/piece=76.5172
unigram_model_trainer.cc(486) LOG(INFO) EM sub_iter=0 size=66474 obj=9.39972 num_tokens=7048669 num_tokens/piece=106.036
unigram_model_trainer.cc(486) LOG(INFO) EM sub_iter=1 size=66474 obj=9.38439 num_tokens=7050941 num_tokens/piece=106.071
unigram_model_trainer.cc(486) LOG(INFO) EM sub_iter=0 size=49855 obj=9.80014 num_tokens=7334836 num_tokens/piece=147.123
unigram_model_trainer.cc(486) LOG(INFO) EM sub_iter=1 size=49855 obj=9.78241 num_tokens=7339902 num_tokens/piece=147.225
unigram_model_trainer.cc(486) LOG(INFO) EM sub_iter=0 size=37391 obj=10.5579 num_tokens=7654233 num_tokens/piece=204.708
unigram_model_trainer.cc(486) LOG(INFO) EM sub_iter=1 size=37391 obj=10.5324 num_tokens=7672187 num_tokens/piece=205.188
unigram_model_trainer.cc(486) LOG(INFO) EM sub_iter=0 size=28043 obj=11.6071 num_tokens=8021953 num_tokens/piece=286.059
unigram_model_trainer.cc(486) LOG(INFO) EM sub_iter=1 size=28043 obj=11.5755 num_tokens=8063782 num_tokens/piece=287.551
unigram_model_trainer.cc(486) LOG(INFO) EM sub_iter=0 size=21032 obj=13.1808 num_tokens=8464260 num_tokens/piece=402.447
unigram_model_trainer.cc(486) LOG(INFO) EM sub_iter=1 size=21032 obj=13.1392 num_tokens=8549719 num_tokens/piece=406.51
unigram_model_trainer.cc(486) LOG(INFO) EM sub_iter=0 size=16500 obj=14.8618 num_tokens=8932725 num_tokens/piece=541.377
unigram_model_trainer.cc(486) LOG(INFO) EM sub_iter=1 size=16500 obj=14.8185 num_tokens=8996465 num_tokens/piece=545.24
trainer_interface.cc(508) LOG(INFO) Saving model: /home/pczapla/workspace/ulmfit-multilingual/data/wiki/ja-100/models/fsp15k/spm.model
trainer_interface.cc(532) LOG(INFO) Saving vocabs: /home/pczapla/workspace/ulmfit-multilingual/data/wiki/ja-100/models/fsp15k/spm.vocab
Data lm, trn: 120037, val: 63
Size of vocabulary: 15000
First 20 words in vocab: ['▁xxunk', '▁xxpad', '▁xxbos', '▁xxeos', '▁xxfld', '▁xxmaj', '▁xxup', '▁xxrep', '▁xxwrep', '<unk>', '▁', '▁、', '▁。', '▁の', '▁に', '▁を', '▁年', '▁は', '▁・', '▁(']
Training args:  {'clip': 0.12, 'alpha': 2, 'beta': 1, 'drop_mult': 0} config:  {'emb_sz': 400, 'n_hid': 1152, 'n_layers': 4, 'pad_token': 1, 'qrnn': True, 'bidir': False, 'output_p': 0.1, 'hidden_p': 0.15, 'input_p': 0.25, 'embed_p': 0.02, 'weight_p': 0.2, 'tie_weights': True, 'out_bias': True}
Bptt 70
Training lm from random weights
epoch     train_loss  valid_loss  accuracy  time
0         3.886147    3.948560    0.430394  47:41
1         3.835572    3.931224    0.429429  47:37
2         3.830497    3.871733    0.439411  47:25
3         3.740645    3.824856    0.446847  47:25
4         3.715881    3.770223    0.455526  47:25
5         3.698531    3.708276    0.463582  47:25
6         3.600051    3.639733    0.476275  47:25
7         3.540591    3.576993    0.487240  47:25
8         3.487257    3.533188    0.496038  47:25
9         3.503373    3.515034    0.499575  47:31
Total time: 7:54:45
data/wiki/ja-100/models/fsp15k
Saving info data/wiki/ja-100/models/fsp15k/qrnn_nl4-1152_lmseed-1.m/info.json
````