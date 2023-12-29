import collections
import re
from d2l import torch as d2l
#@save
d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt',
                                '090b5e7e70c295757f55df93cb0a180b9691891a')
def read_time_machine():
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines()
    # convert non-alphabet characters to space.
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]
lines = read_time_machine()
print(f'# text lines: {len(lines)}')
print(lines[0])
print(lines[10])

def tokenize(lines, token='word'):
    '''
    tokenize the lines
    '''
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('error: unkown token type : ' + token)
    
tokens = tokenize(lines)
for i in range(11):
    print(tokens[i])

class Vocabulary:
    ''' words' dict'''
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        # special case
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        
        # rate by freqs
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key = lambda x : x[1], reverse=True)
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {
            token : idx 
            for idx, token in enumerate(self.idx_to_token)d
        }
        for token, freq in self._token_freqs:
            if freq < min_freq:
                # sorted value less than threshold, no need further going
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1
    
    def __len__(self):
        return len(self.idx_to_token)
    def __getitem__(self, tokens):
        """return idx of givven token"""
        if not isinstance(tokens, (list, tuple)):
            ''' this syntax self.unk is for when tokens not in the list'''
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]
    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]
    def unk(self):
        """unknown token's index is 0"""
        return 0
    def token_freqs(self):
        return self._token_freqs
    
def count_corpus(tokens):
    """ count the frequencies of words"""
    if len(tokens) == 0 or isinstance(tokens[0], list):
        """flat the char tokenized list"""
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)

Vocab = Vocabulary(tokens)
print(list(Vocab.token_to_idx.items())[:10])

def load_corpus(max_tokens=-1):
    lines = read_time_machine()
    tokens = tokenize(lines, 'char')
    vocab = Vocabulary(tokens)
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab
""" notice that here len(vocab will be 28 since it's char tokenized.)"""
corpus, vocab = load_corpus()