import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math


class ResidualConnection(nn.Module):
    def __init__(self, d_model, dropout):
        super(ResidualConnection, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.fc_1 = nn.Linear(d_model, d_ff)
        self.fc_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc_2(self.dropout(F.gelu(self.fc_1(x))))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=256):
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
        self.register_buffer('pos_enc', pe)
    def forward(self, x):
        x = x + self.pos_enc[:, :x.size(1)]
        return self.dropout(x)


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab_size, device):
        super(Embeddings, self).__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
        self.scale = torch.sqrt(torch.FloatTensor([d_model])).to(device)
    def forward(self, x):
        return self.emb(x) * self.scale


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.res_con_1 = ResidualConnection(d_model, dropout)
        self.res_con_2 = ResidualConnection(d_model, dropout)
        self.size = d_model

    def forward(self, x, mask):
        x = self.res_con_1(x, lambda x: self.self_attn(x, x, x, mask))
        return self.res_con_2(x, self.feed_forward)


class Decoder(nn.Module):
    def __init__(self, d_model, output_dim, layer, N):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.size)
        self.fc_output = nn.Linear(d_model, output_dim)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        x = self.norm(x)
        return self.fc_output(x)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, enc_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = d_model
        self.self_attn = self_attn
        self.enc_attn = enc_attn
        self.feed_forward = feed_forward
        self.res_con_1 = ResidualConnection(d_model, dropout)
        self.res_con_2 = ResidualConnection(d_model, dropout)
        self.res_con_3 = ResidualConnection(d_model, dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        x = self.res_con_1(x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.res_con_2(x, lambda x: self.enc_attn(x, enc_output, enc_output, src_mask))
        return self.res_con_3(x, self.feed_forward)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, dropout, device):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.fc_q = nn.Linear(d_model, d_model)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        self.fc_o = nn.Linear(d_model, d_model)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.d_k])).to(device)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        batch_size = query.size(0)
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        #Q = [batch size, query len, d_k]
        #K = [batch size, key len, d_k]
        #V = [batch size, value len, d_k]
        Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        #Q = [batch size, n_heads, query len, d_k]
        #K = [batch size, n_heads, key len, d_k]
        #V = [batch size, n_heads, value len, d_k]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask==0, -1e10)
        p_attn = F.softmax(scores, dim=-1)
        self.attn = p_attn
        x = torch.matmul(self.dropout(p_attn), V)
        #x = [batch size, n heads, query len, d_k]
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)
        #x = [batch size, query len, d_model]
        return self.fc_o(x)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, pad_idx, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.device = device
        self.pad_idx = pad_idx


    def make_src_mask(self, src):
        src_mask = (src != self.pad_idx).unsqueeze(-2)
        return src_mask

    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.pad_idx).unsqueeze(-2)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = self.device)).bool()
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask

    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_trg_mask(tgt)
        return self.decode(self.encode(src, src_mask), src_mask,
                            tgt, tgt_mask)
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, enc_output, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), enc_output, src_mask, tgt_mask)
