= FR =
== VF60k LSTM nl 3 ==
=== LM ===
```
```
=== MLDocs ===
```
```

== SP30k LSTM nl 4 ==
=== LM ===
```
python -m ulmfit lm --dataset-path data/wiki/fr-100 --cuda-id=1 --tokenizer='sp' --nl 4 --name 'nl4' --max-vocab 30000 \
--lang fr --qrnn=False - train 10 --bs=50 --drop_mult=0
```

=== MLDocs ===
```
```
 
