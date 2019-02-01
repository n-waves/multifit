**ULMFiT - Russia - Experiments Readme** 
----------------------------------------
Hello. I'm working on ULMFiT for Russian language. I forked from https://github.com/n-waves/ulmfit-multilingual and mostly inspired by Piotr Czapla work  https://forums.fast.ai/t/multilingual-ulmfit/28117

As for now, I did:
- pretraining language model from wiki-dump on 100M
- finetuning LM and experimenting on ruSentEval task (http://www.dialog-21.ru/evaluation/2016/sentiment/) 
with different sizes of data samples and achieved ~ .98 F1 score, but only positive vs negative (data from http://study.mokoron.com/)

I'm currently working on "Rusentiment" classification task (http://text-machine.cs.uml.edu/projects/rusentiment/), 
which is multiclass and noisier than the previous one. I've performed some first experiments and get close to benchmark results.

I will continue to work on ruSentEval 2016 to perform classification task as in the original competition.

My fork is https://github.com/ademyanchuk/ulmfit-multilingual. 
It has all readmes from parent repo and my experiments are in experiments folder.
____________________
*Experiments for now:*
- all wiki downloading and processing as in parent repo
- all lm-scratch* files are toy examples on how to train LM on 2M wiki, tune it with ruSentEval data and classify on positive and negative
- ruSentiment-experiment.ipynb is baseline on ruSentiment data
- wiki-100-unk-lm-rusent-class* files are for training LM on 100M wiki, tune on tweets from ruSentEval and classify ruSentiment task
- some helpers to merge and fix text files (fix-data-from-db.ipynb, take-data-from-db.ipynb, merge-and-sample-rusent.ipynb)


This work is on fastai v1. All notebooks are self-explanatory and have some comments. 
Feel free to ask questions, comment and provide suggestion
