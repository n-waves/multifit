export CUDA_VISIBLE_DEVICES=0


python -m ulmfit cls --dataset-path data/hate/pl-10-wiki-halftest --bidir=False --qrnn=False --nl 4 \
    --tokenizer='sp' --max-vocab 25000 --lang pl --name 'tiny_test' --lmseed=0 --ftseed=0 --clsweightseed=0 --clstrainseed=0  - \
    train 6 --num_cls_epochs=8 --drop-mult=0 --bs=160 --lr_sched=1cycle
    
    
    
    
    
python -m ulmfit cls --dataset-path data/hate/pl-10-reddit --bidir=False --qrnn=False --nl 4 \
    --tokenizer='sp' --max-vocab 25000 --lang pl --name 'tiny_test' --lmseed=1 --ftseed=0 --clsweightseed=0 --clstrainseed=0  - \
    train 6 --num_cls_epochs=8 --drop-mult=0 --bs=160 --lr_sched=1cycle
    
    


python -m ulmfit poleval19_seeds data/hate/pl-10-reddit/models/sp25k/lstm_tiny_test_lmseed-1-ftseed-0-clsweightseed-0-clstrainseed-0.m --seed_name='clsweightseed'


for seed in 3 4 5 6 7 8; do
    python -m ulmfit poleval19_full data/wiki/pl-100/models/sp25k/lstm_nl4_lmseed-${seed}.m --early_stopping=True --skip_train_seed=True --name "1ep"
done