import random
import torch
from d2l import torch as d2l
from textPreprocessing import *
tokens = d2l.tokenize(read_time_machine())
corpus = [token for line in tokens for token in line]
vocab = d2l.Vocab(corpus)
print(vocab.token_freqs[:10])
# using bigram
bigram_tokens = [pair for pair in zip(corpus[:-1], corpus[1:])]
bigram_vocab = Vocabulary(bigram_tokens)
print(bigram_vocab.token_freqs()[:10])
trigram_tokens = [triple for triple in zip(
    corpus[:-2], corpus[1:-1], corpus[2:]
)]
trigram_vocab = Vocabulary(trigram_tokens)
print(trigram_vocab.token_freqs()[:10])

"""randomly generate sequence_data"""
def seq_data_iter_random(corpus, batch_size, num_steps):
    corpus = corpus[random.randint(0, num_steps - 1):]
    """ minus 1 because considering label (Y equals X + 1) """
    num_subseqs = (len(corpus) - 1) // num_steps
    intial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    random.shuffle(intial_indices)
    def data(pos):
        return corpus[pos : pos + num_steps]

    num_batches = num_subseqs / batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        intial_indices_per_batch = intial_indices[i : i + batch_size]
        X = [data(j) for j in intial_indices_per_batch]
        Y = [data(j + 1) for j in intial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)

"""sequentially generate sequence_data"""
def seq_data_iter_sequential(corpus, batch_size, num_steps):
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = torch.tensor(corpus[offset: offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1 : offset + 1 + num_tokens])
    Xs , Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i : i + num_steps]
        Y = Ys[:, i : i + num_steps]
        yield X, Y
class SeqDataLoader:  #@save
    """load iterator"""
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter:
            self.data_iter_fn = seq_data_iter_random
        else:
            self.data_iter_fn = seq_data_iter_sequential
        self.corpus, self.vocab = load_corpus(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)
def load_data_time_machine(batch_size, num_steps,  #@save
                           use_random_iter=False, max_tokens=10000):
    """return iterator and vocabulary"""
    data_iter = SeqDataLoader(
        batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab
