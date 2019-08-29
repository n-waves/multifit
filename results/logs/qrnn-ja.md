LANG=ja
python -m ulmfit lm --dataset-path data/wiki/${LANG}-100 --tokenizer='sp' --nl 4 --name 'nl4' --max-vocab 30000 --lang ${LANG} --qrnn=True --lmseed=1 - train 10 --bs=50 --drop_mult=0

LANG=ja
python -m ulmfit lm --dataset-path data/wiki/${LANG}-100 --tokenizer='sp' --nl 4 --name 'nl4' --max-vocab 15000 --lang ${LANG} --qrnn=True "--tokenizer-mod=-fix" --lmseed=1 - train 10 --bs=50 --drop_mult=0

python -m ulmfit cls --dataset-path data/mldoc/ja-1 --base-lm-path data/wiki/ja-100/models/sp15k-fix/qrnn_nl4_lmseed-1.m  --lang=ja --name 'nl4' - train 20 --bs 20  --lr_sched=1cycle --label-smoothing-eps=0.1set-path data/mldoc/es-1 --base-lm-path data/wiki/ja-100/models/sp30k-fix/qrnn_nl4.m  --lang=ja --name 'nl4' - train 20 --bs 40 

0         4.875985    3.940408    0.479504  02:40
Total time: 02:40
epoch     train_loss  valid_loss  accuracy  time
0         3.814340    3.574515    0.524143  03:43
1         3.344660    3.174727    0.591801  03:42
2         3.062797    2.909801    0.638397  03:42
3         2.924287    2.753593    0.662699  03:42
4         2.832728    2.648505    0.679922  03:41
5         2.673981    2.575237    0.692270  03:41
6         2.727100    2.521090    0.702180  03:42
7         2.647422    2.474463    0.710956  03:42
8         2.557694    2.437784    0.717445  03:41
9         2.619366    2.398306    0.725727  03:42
10        2.501441    2.368656    0.731127  03:41
11        2.501446    2.340000    0.737203  03:41
12        2.539080    2.316010    0.742567  03:42
13        2.441686    2.290955    0.748064  03:41
14        2.406307    2.273114    0.752770  03:41
15        2.421776    2.256135    0.756699  03:42
16        2.399470    2.245539    0.758821  03:42
17        2.336457    2.237878    0.760780  03:42
18        2.383474    2.234359    0.761501  03:41
19        2.407631    2.233689    0.761739  03:42
Total time: 1:14:01
/home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ja-1/models/sp15k-fix
Saving info /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ja-1/models/sp15k-fix/qrnn_nl4_lmseed-1.m/info.json
Single training schedule
epoch     train_loss  valid_loss  f_beta    precision  recall    kappa_score  matthews_correff  accuracy  time
/home/pczapla/workspace/_oss/fastai/fastai/fastai/metrics.py:179: UserWarning: average=`binary` was selected for a non binary case. Value for average has now been set to `macro` instead.
  def _precision(self):
0         0.872342    0.839480    0.739856  0.836258   0.741572  0.649270     0.682293          0.737000  00:19
Better model found at epoch 0 with f_beta value: 0.739856481552124.
/home/pczapla/workspace/_oss/fastai/fastai/fastai/metrics.py:179: UserWarning: average=`binary` was selected for a non binary case. Value for average has now been set to `macro` instead.
  def _precision(self):
1         0.678405    0.587759    0.886923  0.888504   0.885348  0.845242     0.846027          0.884000  00:19
Better model found at epoch 1 with f_beta value: 0.8869231939315796.
Total time: 00:39
Saving models at /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ja-1/models/sp15k-fix/qrnn_nl4_lmseed-1.m
/home/pczapla/workspace/_oss/fastai/fastai/fastai/metrics.py:179: UserWarning: average=`binary` was selected for a non binary case. Value for average has now been set to `macro` instead.
  def _precision(self):
Model: nl4
Validation on: test
F1 score bin: 0.9002974033355713
Loss: 0.3307778239250183
Precision: 0.9020541906356812
Recall: 0.9002037048339844
Accuracy: 0.9007499814033508
test F1 score bin:     0.9002974033355713
test Loss:             0.33077782
test Precision:        0.9020541906356812
test Recall:           0.9002037048339844
test Kappa Linear:     0.8676621913909912
test Matthews Correff: 0.868194043636322
test Accuracy:         0.9007499814033508



eval --glob="mldoc/ja-1/models/sp15k/qrnn_nl4.m" --name nl4-1cyc-sl  --num-cls-epochs=8 --bs=18 --lr_sched=1cycle --label-smoothing-eps=0.1



