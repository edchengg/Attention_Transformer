import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchtext import data, datasets
import numpy as np
import math
import copy
import time

def clone(module, N):
    res = nn.ModuleList()
    for _ in range(N):
        res.append(module)
    return res

def subsequent_mask(size):
    attn_shape = (1, size, size)
    mask = np.tril(np.ones(attn_shape), k=0).astype('uint8')
    return torch.from_numpy(mask)

''' ================= Sub layer architecture =============== 
    ======== Scaled Dot-Product Attention           ======== 
    ======== Multi-Head Attention                   ======== 
    ======== Position Wise Feed Forward Network     ======== 
    ======== Add & Norm                             ======== 
    ================== Encoder + Decoder ===================
    ======== Encoder layer                          ========
    ======== Encoder                                ========
    ======== Decoder layer                          ========
    ======== Decoder                                ========
    ==================== Input + Output ====================
    ======== Output Linear + Softmax                ========
    ======== Input Embedding                        ========
    ======== Positional Encoding                    ========
    ==================== Overall =========================== 
    ======== Transformer                            ========
    ======================================================== 
'''


''' ======== Scaled Dot-Product Attention ========'''

def Attention(Q, K, V, mask=None, dropout=None):
    '''
    Attention(Q, K, V) = softmax((QK^T)/sqrt(dk))V
    '''
    #dk.size(): (batch, h, -1, d_k)
    dk = K.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(dk) #(batch, h, -1, -1)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    weight = F.softmax(scores, dim = -1) #right most dimension, (batch, h, -1, -1)
    if dropout is not None:
        weight = dropout(weight)
    res = torch.matmul(weight, V) # (batch, h, -1, d_k)
    return res

''' ======== Multi-Head Attention ========'''

class MutiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MutiHeadAttention, self).__init__()
        self.d_k = d_model // h
        self.d_v = d_model // h
        self.d_model = d_model
        self.h = h
        self.W_QKV = clone(nn.Linear(d_model, d_model, bias=False), 3)
        self.W_0 = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)

        # Q.size(): (batch, -1, d_model)
        n_batch = Q.size(0)
        # 1) (QWi, KWi, VWi)
        Q, K ,V = \
            [linear(x).view(n_batch, -1, self.h, self.d_k).transpose(1, 2)
             for linear, x in zip(self.W_QKV, (Q, K, V))]
        # 2) headi = Attention()
        X = Attention(Q, K, V, mask=mask, dropout=self.dropout)
        # 3) Concat(head1, ..., head_h)
        X = X.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        # 4) *W0
        X = self.W_0(X)
        return X

''' ======== Position Wise Feed Forward Network ======='''

class PositionWiseFFN(nn.Module):
    '''
    Position-wise Feed-Forward Networks
    '''
    def __init__(self, d_model=512, d_ff=2048, dropout=0.1):
        super(PositionWiseFFN, self).__init__()
        self.W_1 = nn.Linear(d_model, d_ff)
        self.W_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        res = self.W_2(self.dropout(F.relu(self.W_1(x))))
        return res

''' ======== Add & Norm ========='''

class AddNorm(nn.Module):
    '''
    A residual connection followed by a layer normalization
    '''
    def __init__(self, size, dropout=0.1):
        super(AddNorm, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x))) # norm first instead of last


''' ======== Encoder and Decoder ======== '''
''' ======== Encoder layer ======= '''

class EncoderLayer(nn.Module):
    def __init__(self, size, attention, feed_forward, dropout=0.1):
        super(EncoderLayer, self).__init__()

        self.feed_forward = feed_forward
        self.multi_head_attention = attention
        self.add_norm_1 = AddNorm(size, dropout)
        self.add_norm_2 = AddNorm(size, dropout)
        self.size = size

    def forward(self, x, mask):
        output = self.add_norm_1(x, lambda x: self.multi_head_attention(x, x, x, mask))
        output = self.add_norm_2(output, self.feed_forward)
        return output

''' ======== Encoder ======= '''

class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clone(layer, N)
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

''' ======== Decoder layer ======= '''

class DecoderLayer(nn.Module):
    def __init__(self, size, self_attention, src_attention, feed_forward, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.add_norm_1 = AddNorm(size, dropout)
        self.add_norm_2 = AddNorm(size, dropout)
        self.add_norm_3 = AddNorm(size, dropout)
        self.muti_head_attention = src_attention
        self.masked_muti_head_attention = self_attention
        self.feed_forward = feed_forward

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.add_norm_1(x, lambda x: self.masked_muti_head_attention(x, x, x, tgt_mask))
        x = self.add_norm_2(x, lambda x: self.muti_head_attention(x, m, m, src_mask))
        output = self.add_norm_3(x, self.feed_forward)
        return output

''' ======== Decoder ======= '''

class Decoder(nn.Module):
    def __init__(self, DecoderLayer, N):
        super(Decoder, self).__init__()
        self.layers = clone(DecoderLayer, N)
        self.norm = nn.LayerNorm(DecoderLayer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

''' ======== Output Linear + Softmax ======= '''

class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        res = F.log_softmax(self.proj(x), dim=-1)
        return res

''' ======== Input Embedding ======= '''

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.emb = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        # In the embeddings layers, we multiply whose weights by sqrt(d_model)
        out = self.emb(x) * math.sqrt(self.d_model)
        return out

''' ======== Input Positional Encoding ========= '''

class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

''' ========== Transformer ========== '''

class Transformer(nn.Module):
    def __init__(self, Encoder, Decoder, src_embed, tgt_embed, Generator):
        super(Transformer, self).__init__()
        self.encoder = Encoder
        self.decoder = Decoder
        self.generator = Generator
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed

    def forward(self, src, tgt, src_mask, tgt_mask):
        memory = self.encoding(src, src_mask)
        out = self.decoding(memory, src_mask, tgt, tgt_mask)
        return out

    def encoding(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decoding(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
