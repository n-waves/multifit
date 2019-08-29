# wiki ft6_cl8 
export CUDA_VISIBLE_DEVICES=1 
python -m ulmfit cls \
    --dataset-path data/hate/pl-10 \
    --base-lm-path data/wiki/pl-100/models/sp25k/lstm_seed1.m \
    --lang=pl --name "ft6_cl8"\
    --lmseed 1 --ftseed 0 --clsweightseed 0 --clstrainseed 0\
    - train 6 --bs 160 --num-cls-epochs 8 --lr-sched 1cycle

## weightseed
for seed in {1..9} ; do
    python -m ulmfit eval --glob="hate/pl-10-wiki/models/sp25k/lstm_ft6_cl8_lmseed-1-ftseed-0-clsweightseed-0-clstrainseed-0.m"  --name="ft6_cl8" --num-lm-epochs 0 --bs 160 --num-cls-epochs 8 --clsweightseed $seed --lr-sched 1cycle;
done

## trainseed
for seed in {1..9} ; do
    python -m ulmfit eval --glob="hate/pl-10-wiki/models/sp25k/lstm_ft6_cl8_lmseed-1-ftseed-0-clsweightseed-0-clstrainseed-0.m"  --name="ft6_cl8" --num-lm-epochs 0 --bs 160 --num-cls-epochs 8 --clstrainseed $seed --lr-sched 1cycle;
done

export CUDA_VISIBLE_DEVICES=0
python -m ulmfit cls \
        --dataset-path data/hate/pl-10-wiki \
        --base-lm-path data/wiki/pl-100/models/sp25k/lstm_seed0.m \
        --lang=pl --name "ft6_cl8"\
        --lmseed 0 --ftseed 0 --clsweightseed 0 --clstrainseed 0\
        - train 6 --bs 160 --num-cls-epochs 8 --lr-sched 1cycle
        
for seed in {1..9} ; do 
    python -m ulmfit eval --glob="hate/pl-10-wiki/models/sp25k/lstm_ft6_cl8_lmseed-0-ftseed-0-clsweightseed-0-clstrainseed-0.m"  --name="ft6_cl8" --num-lm-epochs 0 --bs 160 --num-cls-epochs 8 --clsweightseed $seed --lr-sched 1cycle; 
done

## trainseed
for seed in {1..9} ; do
    python -m ulmfit eval --glob="hate/pl-10-wiki/models/sp25k/lstm_ft6_cl8_lmseed-0-ftseed-0-clsweightseed-0-clstrainseed-0.m"  --name="ft6_cl8" --num-lm-epochs 0 --bs 160 --num-cls-epochs 8 --clstrainseed $seed --lr-sched 1cycle;
done

#
python -m ulmfit cls \
    --dataset-path data/hate/pl-10 \
    --base-lm-path data/wiki/pl-100/models/sp25k/lstm_seed1.m \
    --lang=pl --name "ft6_cl8"\
    --lmseed 1 --ftseed 0 --clsweightseed 0 --clstrainseed 0\
    - train 6 --bs 160 --num-cls-epochs 8 --lr-sched 1cycle
    
    
########## evcal

python -m ulmfit poleval19_full data/wiki/pl-100/models/sp25k/lstm_seed0.m --num_lm_epochs=20  ; 
python -m ulmfit poleval19_full data/reddit/pl-100/models/sp25k/qrnn_v25k-nl4-0.m --lmseed=0 --num_lm_epochs=20



########## quick check
seed=0
python -m ulmfit lm --dataset-path data/wiki/pl-100 --tokenizer='sp' --nl 4 --name "seed${seed}-e1" --max-vocab 25000  --lang pl --qrnn=False - train 1 --bs=150 --drop_mult=0  --label-smoothing-eps=0.0 --lmseed $seed
seed=1
python -m ulmfit lm --dataset-path data/wiki/pl-100 --tokenizer='sp' --nl 4 --name "seed${seed}-e1" --max-vocab 25000  --lang pl --qrnn=False - train 1 --bs=150 --drop_mult=0  --label-smoothing-eps=0.0 --lmseed $seed


############## training time
```
Training lm from random weights
epoch     train_loss  valid_loss  accuracy  time
0         2.800385    3.201227    0.442155  1:51:13
Total time: 1:51:13
data/wiki/pl-100/models/sp25k
Saving info data/wiki/pl-100/models/sp25k/lstm_seed0-e1.m/info.json
-------------------------------------------------------------------
Training lm from random weights
epoch     train_loss  valid_loss  accuracy  time
0         2.813926    3.221857    0.439616  1:55:18
Total time: 1:55:18
data/wiki/pl-100/models/sp25k
Saving info data/wiki/pl-100/models/sp25k/lstm_seed1-e1.m/info.json
```

