# Todo 
- [ ] Update these docs

Getting Started
---

## Download and Extract the Wikipedia corpus

In Linux, you can do all the following steps automatically with [prepare_wiki.sh](./prepare_wiki.sh)

**Manual Instructions**

We use the [WikiExtractor.py](http://medialab.di.unipi.it/wiki/Wikipedia_Extractor). It is a Python script that extracts and cleans text from a [Wikipedia database dump](http://download.wikimedia.org/).

At the end of this step, you should have the following directory structure inside ulmfit:
```bash

|- data
  |- wiki
  |- wiki_dumps
  |- wiki_extr
|- wikiextractor
```
The extracted data should be in the folder `wiki_extr` -> language name e.g.`en` (english), `fr` (french) `hi` (hindi) and so on. 

## Create and Post Process WikiText
Use the Python script [create_wikitext.py](./create_wikitext.py) to process the extracted Wikipedia documents. 

If you used the automated shell script from previous step, this might look something like
```bash
python create_wikitext.py -i data/wiki_extr/hi -o data/hindi -l hi
``` 
for hindi (unicode: 'hi')

This should create two splits of your Wikimedia Dumps: a small and large one. 

_**Then**_, use the [postprocess_wikitext.py](./postprocess_wikitext.py) script to finish post processing. This processes numbers, builds a vocab, and limits the vocabulary size. This might look following for Hindi (`hi`)
```bash
python postprocess_wikitext.py -i data/hindi -l hi
```
