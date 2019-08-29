```
$ python -m ulmfit ensemble --glob="data/mldoc/*laser*" --file_template='${dataset_path}/${lang}.train.csv' --gold_labels_template='data/mldoc/${lang}-1/${lang}.train.csv'  --key_template='${lang}' --out_template='data/mldoc/${key}-1-ensemble/${key}.train.csv' --exclude_re=".*([a-z][a-z])-1-laser-probs-\1.*"
Skipping /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1-laser-probs-es1
Skipping /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ja-1-laser-probs-ja1
Skipping /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1-laser-probs-fr1
Skipping /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1-laser-probs-zh1
Skipping /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/en-1-laser-probs-en1
Skipping /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1-laser-probs-it1
Skipping /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1-laser-probs-de1
Skipping /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1-laser-probs-ru1
{'Key': 'ru', 'Test Accuracy': 0.682, 'on': PosixPath('data/mldoc/ru-1/ru.train.csv'), 'files_count': 7}
{'File saved to': PosixPath('data/mldoc/ru-1-ensemble/ru.train.csv')}
{'Key': 'en', 'Test Accuracy': 0.82, 'on': PosixPath('data/mldoc/en-1/en.train.csv'), 'files_count': 7}
{'File saved to': PosixPath('data/mldoc/en-1-ensemble/en.train.csv')}
{'Key': 'es', 'Test Accuracy': 0.821, 'on': PosixPath('data/mldoc/es-1/es.train.csv'), 'files_count': 7}
{'File saved to': PosixPath('data/mldoc/es-1-ensemble/es.train.csv')}
{'Key': 'it', 'Test Accuracy': 0.782, 'on': PosixPath('data/mldoc/it-1/it.train.csv'), 'files_count': 7}
{'File saved to': PosixPath('data/mldoc/it-1-ensemble/it.train.csv')}
{'Key': 'ja', 'Test Accuracy': 0.685, 'on': PosixPath('data/mldoc/ja-1/ja.train.csv'), 'files_count': 7}
{'File saved to': PosixPath('data/mldoc/ja-1-ensemble/ja.train.csv')}
{'Key': 'zh', 'Test Accuracy': 0.789, 'on': PosixPath('data/mldoc/zh-1/zh.train.csv'), 'files_count': 7}
{'File saved to': PosixPath('data/mldoc/zh-1-ensemble/zh.train.csv')}
{'Key': 'de', 'Test Accuracy': 0.905, 'on': PosixPath('data/mldoc/de-1/de.train.csv'), 'files_count': 7}
{'File saved to': PosixPath('data/mldoc/de-1-ensemble/de.train.csv')}
{'Key': 'fr', 'Test Accuracy': 0.86, 'on': PosixPath('data/mldoc/fr-1/fr.train.csv'), 'files_count': 7}
{'File saved to': PosixPath('data/mldoc/fr-1-ensemble/fr.train.csv')}
```

```



ln -s /home/pczapla/workspace/ulmfit-multilingual/data-archive/mldoc/de-1/models/sp15k/qrnn_nl4_0.m  data-archive/mldoc/de-1/models/sp15k/qrnn_base.m
ln -s /home/pczapla/workspace/ulmfit-multilingual/data-archive/mldoc/en-1/models/sp15k/qrnn_nl4_tls.m  data-archive/mldoc/en-1/models/sp15k/qrnn_base.m
ln -s /home/pczapla/workspace/ulmfit-multilingual/data-archive/mldoc/es-1/models/sp15k/qrnn_nl4_0.m  data-archive/mldoc/es-1/models/sp15k/qrnn_base.m
ln -s /home/pczapla/workspace/ulmfit-multilingual/data-archive/mldoc/fr-1/models/sp15k/qrnn_nl4_tls.m  data-archive/mldoc/fr-1/models/sp15k/qrnn_base.m
ln -s /home/pczapla/workspace/ulmfit-multilingual/data-archive/mldoc/it-1/models/sp15k/qrnn_nl4_tls.m  data-archive/mldoc/it-1/models/sp15k/qrnn_base.m
ln -s /home/pczapla/workspace/ulmfit-multilingual/data-archive/mldoc/ja-1/models/sp15k/qrnn_nl4_tls.m  data-archive/mldoc/ja-1/models/sp15k/qrnn_base.m
ln -s /home/pczapla/workspace/ulmfit-multilingual/data-archive/mldoc/ru-1/models/sp15k/qrnn_nl4_tls.m  data-archive/mldoc/ru-1/models/sp15k/qrnn_base.m
ln -s /home/pczapla/workspace/ulmfit-multilingual/data-archive/mldoc/zh-1/models/sp15k/qrnn_nl4_tls.m  data-archive/mldoc/zh-1/models/sp15k/qrnn_base.m


for a in data-archive/mldoc/*-1; do cp $a/*unsup.csv $a/*test.csv $a/*dev.csv ${a/-archive/}-ensemble; done
python -m ulmfit ls --glob 'data-archive/mldoc/*-1/models/sp15k/qrnn_base.m' --dataset_template='data/mldoc/${lang}-ensemble'


python -m ulmfit eval --glob 'data-archive/mldoc/*-1/models/sp15k/qrnn_base.m' --dataset_template='../../data/mldoc/${lang}-ensemble' --num_lm_epochs=0 --num_cls_epochs=8 --early_stopping=False --bs=20 --label-smoothing-eps=0.1 --lr_sched=1cycle --skip_on_error=False

```