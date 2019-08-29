# RU

#CSL direct
export CUDA_VISIBLE_DEVICES=0
LANG=ru
python -m ulmfit cls --dataset-path data/mldoc-e/${LANG}-1 --bidir=False --qrnn=True --nl 4 \
    --tokenizer='sp' --max-vocab 15000 --lang $LANG --name 'nowiki' --lmseed=$CUDA_VISIBLE_DEVICES --ftseed=0 --clsweightseed=0 --clstrainseed=0 \ 
    - train 20 --num_cls_epochs=8 --early_stopping=False --drop-mult=0 --bs=50 --label-smoothing-eps=0.1 --lr_sched=1cycle
    
    
for LANG in de es fr ; do 
 python -m ulmfit cls --dataset-path data/mldoc/${LANG}-1 --bidir=False --qrnn=True --nl 4 --use_tst_for_lm=False \
    --tokenizer='sp' --max-vocab 15000 --lang $LANG --name 'nowiki-bst' --lmseed=6 --ftseed=0 --clsweightseed=5 --clstrainseed=0 \
    - train 20 --num_cls_epochs=8 --early_stopping=False --drop-mult=0 --bs=20 --label-smoothing-eps=0.1 --lr_sched=1cycle
done
    
for LANG in it ja zh; do 
 python -m ulmfit cls --dataset-path data/mldoc/${LANG}-1 --bidir=False --qrnn=True --nl 4 --use_tst_for_lm=False \
    --tokenizer='sp' --max-vocab 15000 --lang $LANG --name 'nowiki-bst' --lmseed=6 --ftseed=0 --clsweightseed=5 --clstrainseed=0 \
    - train 20 --num_cls_epochs=8 --early_stopping=False --drop-mult=0 --bs=20 --label-smoothing-eps=0.1 --lr_sched=1cycle
done
----------------

for LANG in de es fr ; do 
 python -m ulmfit cls --dataset-path data/mldoc/${LANG}-1 --bidir=False --qrnn=True --nl 4 --use_tst_for_lm=False \
    --tokenizer='sp' --max-vocab 15000 --lang $LANG --name 'nowiki-wrst' --lmseed=10 --ftseed=0 --clsweightseed=0 --clstrainseed=0 \
    - train 20 --num_cls_epochs=8 --early_stopping=False --drop-mult=0 --bs=20 --label-smoothing-eps=0.1 --lr_sched=1cycle
done
    
for LANG in it ja zh; do 
 python -m ulmfit cls --dataset-path data/mldoc/${LANG}-1 --bidir=False --qrnn=True --nl 4 --use_tst_for_lm=False \
    --tokenizer='sp' --max-vocab 15000 --lang $LANG --name 'nowiki-wrst' --lmseed=10 --ftseed=0 --clsweightseed=0 --clstrainseed=0 \
    - train 20 --num_cls_epochs=8 --early_stopping=False --drop-mult=0 --bs=20 --label-smoothing-eps=0.1 --lr_sched=1cycle
done

--------------------



for LANG in de es fr ; do 
python -m ulmfit cls --dataset-path data/mldoc/${LANG}-1 --bidir=False --qrnn=True --nl 4 \
    --tokenizer='sp' --max-vocab 15000 --lang $LANG --name 'nowiki-wrst' --lmseed=10 --ftseed=0 --clsweightseed=0 --clstrainseed=0 \ 
    - train 20 --num_cls_epochs=8 --early_stopping=False --drop-mult=0 --bs=50 --label-smoothing-eps=0.1 --lr_sched=1cycle
done
    
    















    
    
# test downstream    
    

for seed in {1..5}; do
  echo $seed;
  python -m ulmfit eval --glob 'data/mldoc-e/ru-1/models/sp15k/qrnn_nowiki-notst_lmseed-*-ftseed-0-clsweightseed-0-clstrainseed-0.m' \
    --clsweightseed=$seed --clstrainseed=0 --num_lm_epochs=0 --num_cls_epochs=8 --early_stopping=False  --bs=20 --label-smoothing-eps=0.1 --lr_sched=1cycle 
done


for seed in {6..10}; do
  echo $seed;
  python -m ulmfit eval --glob 'data/mldoc-e/ru-1/models/sp15k/qrnn_nowiki-notst_lmseed-*-ftseed-0-clsweightseed-0-clstrainseed-0.m' \
    --clsweightseed=$seed --clstrainseed=0 --num_lm_epochs=0 --num_cls_epochs=8 --early_stopping=False  --bs=20 --label-smoothing-eps=0.1 --lr_sched=1cycle 
done


for seed in {1..10}; do
  echo $seed;
  python -m ulmfit eval --glob 'data/mldoc-e/ru-1/models/sp15k/qrnn_nowiki_lmseed-*-ftseed-0-clsweightseed-0-clstrainseed-0.m' \
    --clsweightseed=$seed --clstrainseed=0 --num_lm_epochs=0 --num_cls_epochs=8 --early_stopping=False  --bs=20 --label-smoothing-eps=0.1 --lr_sched=1cycle
done 





ln -s /home/pczapla/workspace/ulmfit-multilingual/data-archive/mldoc/de-1/models/sp15k/qrnn_nl4_0.m  data-archive/mldoc/de-1/models/sp15k/qrnn_base.m
ln -s /home/pczapla/workspace/ulmfit-multilingual/data-archive/mldoc/en-1/models/sp15k/qrnn_nl4_tls.m  data-archive/mldoc/en-1/models/sp15k/qrnn_base.m
ln -s /home/pczapla/workspace/ulmfit-multilingual/data-archive/mldoc/es-1/models/sp15k/qrnn_nl4_0.m  data-archive/mldoc/es-1/models/sp15k/qrnn_base.m
ln -s /home/pczapla/workspace/ulmfit-multilingual/data-archive/mldoc/fr-1/models/sp15k/qrnn_nl4_tls.m  data-archive/mldoc/fr-1/models/sp15k/qrnn_base.m
ln -s /home/pczapla/workspace/ulmfit-multilingual/data-archive/mldoc/it-1/models/sp15k/qrnn_nl4_tls.m  data-archive/mldoc/it-1/models/sp15k/qrnn_base.m
ln -s /home/pczapla/workspace/ulmfit-multilingual/data-archive/mldoc/ja-1/models/sp15k/qrnn_nl4_tls.m  data-archive/mldoc/ja-1/models/sp15k/qrnn_base.m
ln -s /home/pczapla/workspace/ulmfit-multilingual/data-archive/mldoc/ru-1/models/sp15k/qrnn_nl4_tls.m  data-archive/mldoc/ru-1/models/sp15k/qrnn_base.m
ln -s /home/pczapla/workspace/ulmfit-multilingual/data-archive/mldoc/zh-1/models/sp15k/qrnn_nl4_tls.m  data-archive/mldoc/zh-1/models/sp15k/qrnn_base.m



