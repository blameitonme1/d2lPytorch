import torch
from d2l import torch as d2l
from torch import nn
class AttentionDecoder(d2l.Decoder):
    """ 注意力机制的解码器接口"""
    def __init__(self):
        super().__init__()
    
    def attention_weights(self):
        raise NotImplementedError

class Seq2SeqAttentionDecoder(AttentionDecoder):
    """有注意力机制实现的Seq2Seq的decoder"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs):
        super().__init__()
        self.attention = d2l.AdditiveAttention(
            num_hiddens, dropout
        ) # 使用加性注意力，因为可以学习参数效果会更好
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(
            embed_size + num_hiddens, num_hiddens, num_layers, dropout=dropout
        )
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_output, enc_valid_lens, *args):
        # 要使用valid_len防止考虑pad的token，pad的都不用考虑
        outputs, hidden_state = enc_output
        # permute是因为考虑的是相对位置，也就是在时间轴上应用注意力
        return (outputs.permute(1, 0, 2), hidden_state, enc_valid_lens)
    
    def forward(self, X, state):
        enc_outputs, hidden_state, enc_valid_lens = state
        # 输出X的形状是(num_steps,batch_size,embed_size)
        X = self.embedding(X).permute(1, 0, 2)
        outputs, self._attention_weights = [], []
        for x in X:
            # 因为每一次要处理一个query，所以不能一口气放进去
            #  query的形状为(batch_size,1,num_hiddens), 采用上一个时间的预测的最后一层隐藏层状态
            query = torch.unsqueeze(hidden_state[-1], dim=1)
            # context根据注意力池化进行加权平均计算， batch_size的维度没有管，直接计算后面两个维度
            # context的形状为(batch_size,1,num_hiddens) 1 代表只有一个query
            context = self.attention(
                query, enc_outputs, enc_outputs, enc_valid_lens
            ) # 有num_steps个keyvalue pair，有一个query，他们大小都相等但是使用加性注意力可以学习参数
            x = torch.cat((context, torch.unsqueeze(x, dim=1)), dim=-1) #注意API，在最后一个维度拼接
            # 将x变形为(1,batch_size,embed_size+num_hiddens), 因为PyTorch要求时间维度在前面
            out, hidden_state = self.rnn(x.permute(1, 0, 2), hidden_state)
            outputs.append(out)
            # 把学习的参数append
            self._attention_weights.append(self.attention.attention_weights)
        # 先第一个维度cat，得到三维张量，最后形状为(num_steps,batch_size,vocab_size)
        outputs = self.dense(torch.cat(outputs, dim=0))
        # 转换维度，batch_size主维度分别表示每个句子的预测结果
        return outputs.permute(1, 0, 2), [enc_outputs, hidden_state, enc_valid_lens]
    
    def attention_weights(self):
        return self._attention_weights
# 下面训练，直接cv代码
embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
batch_size, num_steps = 64, 10
lr, num_epochs, device = 0.005, 250, d2l.try_gpu()

train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)
encoder = d2l.Seq2SeqEncoder(
    len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
decoder = Seq2SeqAttentionDecoder(
    len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
net = d2l.EncoderDecoder(encoder, decoder)
d2l.train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)
engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
for eng, fra in zip(engs, fras):
    translation, dec_attention_weight_seq = d2l.predict_seq2seq(
        net, eng, src_vocab, tgt_vocab, num_steps, device, True)
    print(f'{eng} => {translation}, ',
          f'bleu {d2l.bleu(translation, fra, k=2):.3f}')