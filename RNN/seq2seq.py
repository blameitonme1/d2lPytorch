import torch
from torch import nn
from d2l import torch as d2l
import collections
import math
from machine_translation_dataset import load_data_nmt
d2l.Seq2SeqEncoder
class Seq2SeqEncoder(d2l.Encoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0):
        super().__init__()
        # embedding layer, think it as somewhat similar to one-hot previously told.
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers, dropout=dropout)
    def forward(self, X, *args):
        """shape of output X (batch_size, num_steps, embed_size)"""
        X = self.embedding(X)
        """need to swtich dimentions to make num_steps the first dimention"""
        X = X.permute(1, 0, 2)
        """shape of output (num_steps, batch_size, num_hiddens), since it's encoder, no need to use a classifier"""
        """shape of state (num_layers, batch_size, num_hiddens), num_layers because using multi-layers RNN"""
        output, state = self.rnn(X)
        return output, state
class Seq2SeqDecoder(d2l.Decoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                dropout=0, **kwargs):
        super().__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers, dropout=dropout)
        """ need output layer because decoder will predict """
        self.dense = nn.Linear(num_hiddens, vocab_size)
    def init_state(self, enc_outputs, *args):
        return enc_outputs[1] # end_outputs is output, state.
    def forward(self, X, state):
        """ X shape (batch_size,num_steps,embed_size) """
        X = self.embedding(X)
        X = X.permute(1, 0, 2)
        """ context is the last layer of the last RNN cell of the encoder """
        context = state[-1].repeat(X.shape[0], 1, 1)
        X_and_context = torch.cat((X, context), 2)
        output, state = self.rnn(X_and_context, state)
        """switch batch size to front """
        output = self.dense(output).permute(1, 0, 2)
        """ output shape (batch_size, num_steps, vocab_size)
        state shape (num_layers, batch_size, num_hiddens)"""
        return output, state
def sequence_mask(X, valid_len, value=0):
    maxlen = X.size(1) # X shape (batchsize, longth of each sentence)
    mask = torch.arange((maxlen), dtype=torch.float32, device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X 

class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    def forward(self, pred, label, valid_len):
        """creating masks, dont count padding when calculating loss """
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction = 'none'
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(
            pred.permute(0, 2, 1), label
        )
        """for each sentence, calculate mean (dim = 1)"""
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss

def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    def xavier_init_weights(m):
        """ init module m depending using xavier depending on which type it is """
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])
    
    net.apply(xavier_init_weights)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = MaskedSoftmaxCELoss()
    net.train()
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                     xlim=[10, num_epochs])
    for epoch in range(num_epochs):
        timer = d2l.Timer()
        metric = d2l.Accumulator(2) # sum of all loss and numbers of tokens
        for batch in data_iter:
            optimizer.zero_grad()
            """HERE X IS ORIGINAL SENTENCE AND Y IS TRANLATED SENTENCE!"""
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            """ 因为训练的时候 decoder每一次都要输入真实的文本数据, 所以要dec_input """
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0], device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], 1) # forceful teaching
            Y_hat = net(X, dec_input, X_valid_len) # x is enc_input, while dec_input is obvious.
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward()
            d2l.grad_clipping(net, 1)
            num_tokens = Y_valid_len.sum()
            optimizer.step()
            with torch.no_grad():
                metric.add(l.sum(), num_tokens)
        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, (metric[0] / metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} '
        f'tokens/sec on {str(device)}')

embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
batch_size, num_steps = 64, 10
lr, num_epochs, device = 0.005, 300, d2l.try_gpu()

train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size, num_steps)
encoder = Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
decoder = Seq2SeqDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
net = d2l.EncoderDecoder(encoder, decoder)
train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)

def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps, device, save_attention_weights=False):
    net.eval()
    """ 加入<eos>表示句子结尾 """
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [src_vocab['<eos>']]
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = d2l.truncate_pad(src_tokens, num_steps, src_vocab['<pad>']) # 裁剪或者pad句子，根据num_steps
    # 将原始句子包装成一个张量，dim = 0是为了处理批量 shape of enc_X (1, num_steps)
    enc_X = torch.unsqueeze(
        torch.tensor(src_tokens, dtype=torch.long, device=device),dim=0
    )
    enc_outputs = net.encoder(enc_X, enc_valid_len) # valid_len现在还用不上，之后会用到
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len) # 使用encoder最后一层的hidden state初始化decoder
    dec_X = torch.unsqueeze(
        torch.tensor([tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0
    )
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        # 最多预测num_steps个词
        Y, dec_state = net.decoder(dec_X, dec_state)
        # 取预测的最大可能性的词作为下一个时间步的输入
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        # 注意力机制，现在暂时不用
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        if pred == tgt_vocab['<eos>']:
            # 预测到了句子结尾
            break        
        output_seq.append(pred)
    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq

def bleu(pred_seq, label_seq, k):
    """计算 BLEU, 直接抄公式即可 """
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred)) # 惩罚过短的预测
    # 统计n gram的准确率
    for n in range(1, k + 1):
        num_mathes, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i : i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i : i + n])] > 0:
                # label含有这个n gram
                num_mathes += 1
                label_subs[' '.join(pred_tokens[i : i + n])] -= 1
        score *= math.pow(num_mathes / (len_pred - n + 1), math.pow(0.5, n))
    return score


engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
for eng, fra in zip(engs, fras):
    translation, attention_weight_seq = predict_seq2seq(
        net, eng, src_vocab, tgt_vocab, num_steps, device)
    print(f'{eng} => {translation}, bleu {bleu(translation, fra, k=2):.3f}')






    