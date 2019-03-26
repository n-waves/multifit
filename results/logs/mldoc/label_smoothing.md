# Multifit Best results after label smoothing
 
|     |  de-1 |  en-1 |  es-1 |  fr-1 |  it-1 |  ja-1 |  ru-1 |  zh-1|
|-----|-------|-------|-------|-------|-------|-------|-------|------|
|best | 95.90 | 95.17 | 96.07 | 94.75 | 90.25 | 90.03 | 87.65 | 92.52|
|max  | 95.90 | 95.55 | 96.07 | 94.75 | 90.38 | 90.03 | 87.65 | 92.52|
|avg  | 95.77 | 95.27 | 95.92 | 94.75 | 90.24 | 89.89 | 87.28 | 92.31|
 
 
## Log
```
python -m ulmfit eval --glob="mldoc/ru-1/models/sp15k/qrnn_nl4.m" --lr_sched=1cycle --bs=18 --num-cls-epochs=8 --name "nl4_tls4" --label-smoothing-eps=0.1

                                            name  tst_accuracy  tst_loss  val_accuracy  val_loss
0    data/mldoc/de-1/models/sp15k/qrnn_nl4_tls.m       0.95850  0.254842         0.946  0.320358
1   data/mldoc/de-1/models/sp15k/qrnn_nl4_tls2.m       0.95900  0.245983         0.947  0.303949
2   data/mldoc/de-1/models/sp15k/qrnn_nl4_tls3.m       0.95550  0.270527         0.938  0.323216
3    data/mldoc/en-1/models/sp15k/qrnn_nl4_tls.m       0.95550  0.246017         0.959  0.237861
4   data/mldoc/en-1/models/sp15k/qrnn_nl4_tls2.m       0.95075  0.258219         0.959  0.235698
5   data/mldoc/en-1/models/sp15k/qrnn_nl4_tls3.m       0.95175  0.249414         0.960  0.245007
6    data/mldoc/es-1/models/sp15k/qrnn_nl4_tls.m       0.95875  0.258491         0.961  0.255865
7   data/mldoc/es-1/models/sp15k/qrnn_nl4_tls2.m       0.95825  0.263527         0.959  0.274785
8   data/mldoc/es-1/models/sp15k/qrnn_nl4_tls3.m       0.96075  0.253370         0.965  0.254268
9    data/mldoc/fr-1/models/sp15k/qrnn_nl4_tls.m       0.94750  0.277039         0.942  0.295544
10  data/mldoc/fr-1/models/sp15k/qrnn_nl4_tls2.m       0.94750  0.284394         0.943  0.288495
11  data/mldoc/fr-1/models/sp15k/qrnn_nl4_tls3.m       0.94750  0.268739         0.938  0.274793
12   data/mldoc/it-1/models/sp15k/qrnn_nl4_tls.m       0.90100  0.424416         0.899  0.386466
13  data/mldoc/it-1/models/sp15k/qrnn_nl4_tls2.m       0.90375  0.410442         0.913  0.381761
14  data/mldoc/it-1/models/sp15k/qrnn_nl4_tls3.m       0.90250  0.416314         0.917  0.378864
15   data/mldoc/ja-1/models/sp15k/qrnn_nl4_tls.m       0.89850  0.456913         0.887  0.507895
16  data/mldoc/ja-1/models/sp15k/qrnn_nl4_tls2.m       0.90025  0.426836         0.897  0.469335
17  data/mldoc/ja-1/models/sp15k/qrnn_nl4_tls3.m       0.89800  0.449715         0.890  0.502422
18   data/mldoc/ru-1/models/sp15k/qrnn_nl4_tls.m       0.86550  0.571294         0.870  0.548535
19  data/mldoc/ru-1/models/sp15k/qrnn_nl4_tls2.m       0.87650  0.587116         0.877  0.585862
20  data/mldoc/ru-1/models/sp15k/qrnn_nl4_tls3.m       0.87625  0.550317         0.866  0.574534
21   data/mldoc/zh-1/models/sp15k/qrnn_nl4_tls.m       0.92525  0.347967         0.921  0.350878
22  data/mldoc/zh-1/models/sp15k/qrnn_nl4_tls2.m       0.92175  0.377572         0.917  0.380295
23  data/mldoc/zh-1/models/sp15k/qrnn_nl4_tls3.m       0.92225  0.350547         0.916  0.362135
```