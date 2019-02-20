import glob
import shutil
import time
from fastai.text import * 

orig_path = untar_data(URLs.IMDB)
path = Path('data') / 'imdb_small'
path.mkdir(parents=True, exist_ok=True)

for mode in ['train', 'test']:
    for label in ['pos', 'neg']:
        tgt_path = path / mode / label
        tgt_path.mkdir(parents=True, exist_ok=True)
        # Keep just 10% of the files
        pattern = str(orig_path / mode / label / '3*.txt')
        for file in glob.glob(pattern):
            shutil.copy(file, tgt_path)

data_lm = TextLMDataBunch.from_folder(path)
data_clas = TextClasDataBunch.from_folder(path, bs=32, vocab=data_lm.train_ds.vocab)

print('Vocab size', len(data_lm.train_ds.vocab.itos))


def count_parameters(model, requires_grad):
    return sum(p.numel() for p in model.parameters() if p.requires_grad == requires_grad)


def test(qrnn, func, config, data, arch=AWD_LSTM):
    total = len(list(data.train_dl))
    config = config.copy()
    config['qrnn'] = qrnn
    
    learn = func(data, AWD_LSTM, config=config, pretrained=False)
    learn.unfreeze()
    params = count_parameters(learn.model, True)
    total = len(list(data.train_dl))
    start_time = time.clock()
    learn.fit(1)
    diff = time.clock() - start_time
    
    print('Batch size', data.one_batch()[0].shape)
    print(f'Params = {params // 1000000} MM')
    print(f'Training time is {1000 * diff // total} ms per batch')


for qrnn in [True, False]:
    print('QRNN' if qrnn else 'LSTM')
    print('LM')
    test(qrnn, language_model_learner, config=awd_lstm_lm_config, data=data_lm)
    print('CLAS')
    test(qrnn, text_classifier_learner, config=awd_lstm_clas_config, data=data_clas)