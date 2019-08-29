#!/bin/bash

if [ "$seed" == "" ]; then
    echo "seed env is required"
    exit 1;
fi
echo "Training seed=$seed"
python -m ulmfit poleval19_full data/wiki/pl-100/models/sp25k/lstm_seed${seed}.m --num_lm_epochs=20
python -m ulmfit poleval19_full data/reddit/pl-100/models/sp25k/qrnn_v25k-nl4-${seed}.m --lmseed=$seed --num_lm_epochs=20
python -m ulmfit poleval19_full data/wiki/pl-100/models/sp25k/lstm_seed${seed}.m --num_lm_epochs=6
python -m ulmfit poleval19_full data/reddit/pl-100/models/sp25k/qrnn_v25k-nl4-${seed}.m --lmseed=$seed --num_lm_epochs=6
python -m ulmfit poleval19_full data/wiki/pl-100/models/sp25k/lstm_seed${seed}.m --num_lm_epochs=0
python -m ulmfit poleval19_full data/reddit/pl-100/models/sp25k/qrnn_v25k-nl4-${seed}.m --lmseed=$seed --num_lm_epochs=0


python -m ulmfit poleval19_full data/wiki/pl-100/models/sp25k/lstm_seed${seed}-e1.m --num_lm_epochs=6

# python -m ulmfit poleval19_full data/wiki/pl-100/models/sp25k/lstm_seed${seed}-e1.m --num_lm_epochs=6 --name "small_ft6_el20




