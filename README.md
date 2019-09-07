
# Retraining a multifit model from wikipedia

The whole training process from wikipedia to mldoc can be run as follows:  
```bash
python -m ulmfit new multifit_fp16 \
    pretrain-lm train- data/wiki/de-100 - \
    finetune-lm train- data/mldoc/de-1 - \
    classifier train- data/mldoc/de-1
```
You can evaulate any model with the following command: 
```bash
python -m ulmfit load data/mldoc/de-1/models/fsp15k/multfit_fp16 classifier validate data/mldoc/de-1
```
