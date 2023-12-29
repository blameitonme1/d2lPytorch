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