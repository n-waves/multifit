from multifit import ULMFiT
from fastai.text import *
from sotabencheval.language_modelling import WikiText103Evaluator
from sotabencheval.utils import is_server

def iterate_over_batches(data, bs, bptt):
    def batched(Z, bptt):
        sz = Z.shape[-1]
        for s in range(0, sz, bptt):
            yield Z[..., s:s+bptt]
    size = data.numel()
    batched_size = ((size-1) // bs) * bs
    # filp - to be able to switch to batch_size 1 later and maintain trasnfoxl memory
    X = data[:batched_size].view(bs, -1).flip(0,)
    Y = data[1:batched_size+1].view(bs, -1).flip(0,)
    yield from zip(batched(X, bptt), batched(Y, bptt))
    X = data[None, batched_size:-1]
    Y = data[None, batched_size+1:]
    yield from zip(batched(X, bptt), batched(Y, bptt))

#TODO the tokenization removes new lines so te perplexity coalculation is off
def evaluate(pretrained_name):
    model = ULMFiT().from_pretrained_(pretrained_name)
    if is_server():
        wikitext_folder = WikiText103Evaluator.dataset.get_path(local_root="unused")
    else:
        wikitext_folder = untar_data(URLs.WIKITEXT)
    ds = model.arch.dataset(wikitext_folder, tokenizer=model.pretrain_lm.tokenizer)

    test_df = ds.read_data(ds.tst_path)
    data_lm = ds.databunch_from_df(TextLMDataBunch, test_df, test_df, bs=20, bptt=70)
    learn = model.finetune_lm.get_learner(data_lm)

    full_data = np.concatenate(data_lm.valid_ds.items)

    evaluator = WikiText103Evaluator(
        model_name="Multifit (slim)",
        model_description=pretrained_name,
        paper_arxiv_id="1909.04761",
        local_root=str(wikitext_folder)
    )

    learn.loss_func = None

    dev = torch.device("cuda")
    evaluator.reset()
    batches = iterate_over_batches(torch.tensor(full_data), bs=200, bptt=70)
    for x,y in progress_bar(batches, total=len(full_data)//200//70):
        logits = learn.pred_batch(batch=[x.to(dev), y.to(dev)])
        log_probs = torch.log_softmax(logits, -1)
        evaluator.add(log_probs, y)
        if evaluator.cache_exists:
            break
    evaluator.save()
    print(pretrained_name)
    evaluator.print_results()
    return evaluator.results

evaluate("en_multifit_nl3_wiki103")