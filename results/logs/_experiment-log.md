


## QRNN sp15k
```
cd fastai # go to fast ai
git checkout ulfit_multilingual
git pull

cd ../ulmfit-multilingual # go to ulmfit
git checkout master
git pull

export CUDA_VISIBLE_DEVICES=1
LANG=fr
python -m ulmfit lm --dataset-path data/wiki/${LANG}-100 --tokenizer='sp' --nl 4 --name 'nl4' --max-vocab 15000 --lang ${LANG} --qrnn=True - train 10 --bs=50 --drop_mult=0

## Jeremy
export CUDA_VISIBLE_DEVICES=2
LANG=it
python -m ulmfit lm --dataset-path data/wiki/${LANG}-100 --tokenizer='sp' --nl 4 --name 'nl4' --max-vocab 15000 --lang ${LANG} --qrnn=True - train 10 --bs=50 --drop_mult=0

export CUDA_VISIBLE_DEVICES=3
LANG=ru
python -m ulmfit lm --dataset-path data/wiki/${LANG}-100 --tokenizer='sp' --nl 4 --name 'nl4' --max-vocab 15000 --lang ${LANG} --qrnn=True - train 10 --bs=50 --drop_mult=0
```



```
#
export CUDA_VISIBLE_DEVICES=0
LANG=de
python -m ulmfit lm --dataset-path data/wiki/${LANG}-100 --tokenizer='sp' --nl 4 --name 'nl4' --max-vocab 15000 --lang ${LANG} --qrnn=True - train 10 --bs=50 --drop_mult=0
#

## Piotr
export CUDA_VISIBLE_DEVICES=1
LANG=es
python -m ulmfit lm --dataset-path data/wiki/${LANG}-100 --tokenizer='sp' --nl 4 --name 'nl4' --max-vocab 15000 --lang ${LANG} --qrnn=True - train 10 --bs=50 --drop_mult=0

export CUDA_VISIBLE_DEVICES=0
LANG=zh
python -m ulmfit lm --dataset-path data/wiki/${LANG}-100 --tokenizer='sp' --nl 4 --name 'nl4' --max-vocab 15000 --lang ${LANG} --qrnn=True - train 10 --bs=50 --drop_mult=0
```

# trained
export CUDA_VISIBLE_DEVICES=0
LANG=ja
python -m ulmfit lm --dataset-path data/wiki/${LANG}-100 --tokenizer='sp' --nl 4 --name 'nl4' --max-vocab 15000 --lang ${LANG} --qrnn=True - train 10 --bs=50 --drop_mult=0

done V100
```
export CUDA_VISIBLE_DEVICES=0
LANG=en
python -m ulmfit lm --dataset-path data/wiki/${LANG}-100 --tokenizer='sp' --nl 4 --name 'nl4' --max-vocab 15000 --lang ${LANG} --qrnn=True - train 10 --bs=50 --drop_mult=0
```