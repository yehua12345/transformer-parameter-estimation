import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from thop import profile
from functools import partial

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)



    def forward(self, x):
        residual = x
        x = F.relu(self.w_1(x))
        x = self.dropout(x)
        x = F.relu(self.w_2(x))
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)

        return x



class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature

    def forward(self, q, k, v):
        attn = torch.matmul(q / self.temperature, k.transpose(1, 2))
        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, v)

        return output
class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.6):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v):
        residual = q

        q = rearrange(self.w_qs(q), 'b n (h d) -> (b h) n d', h=self.n_head)
        k = rearrange(self.w_ks(k), 'b n (h d) -> (b h) n d', h=self.n_head)
        v = rearrange(self.w_vs(v), 'b n (h d) -> (b h) n d', h=self.n_head)

        context = self.attention(q, k, v)
        context = rearrange(context, '(b h) n d -> b n (h d)', h=self.n_head)
        context= self.dropout(F.relu(self.fc(context)))
        context += residual
        context = self.layer_norm(context)
        return context
class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.6):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input):
        enc_output= self.slf_attn(enc_input, enc_input, enc_input)
        enc_output = self.pos_ffn(enc_output)
        return enc_output

class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''
    def __init__( self, n_layers, n_head, d_k, d_v,d_model, d_inner, dropout=0.6):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.d_model = d_model

    def forward(self, src_seq):
        for enc_layer in self.layer_stack:
            src_seq= enc_layer(src_seq)
        return src_seq

class ViT(nn.Module):
    def __init__(self, *,dim=256, depth=8, heads=6, mlp_dim=128,dim_head = 64, dropout = 0.6):
        super().__init__()
        self.to_patch_embedding1 = nn.Sequential(
            Rearrange('b c (h n) w -> b h (n w c)',n=8),
            nn.LayerNorm(16),
            nn.Linear(16,256)
        )

        self.pe = torch.tensor([[pos / (10000.0 ** (i // 2 * 2.0 / dim)) for i in range(dim)] for pos in range(dim_head)])
        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])
        self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])
        self.Positional_Encoding=nn.Parameter(self.pe, requires_grad=False).to(device="cuda:0")
        self.encoder = Encoder(depth,heads,dim_head,dim_head,dim,mlp_dim, dropout)
        self.to_latent = nn.Identity()
        self.mlp_head1 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 12)
        )
        self.mlp_head2 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 21)
        )
        self.mlp_head3 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 10)
        )
        self.mlp_head4 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 5)
        )

    def forward(self, img, mask = None):
        x = img

        x = self.to_patch_embedding1(x)

        x = x + self.Positional_Encoding
        enc_output= self.encoder(x)

        x = enc_output.mean(dim=1)
        x = self.to_latent(x)
        return self.mlp_head1(x), self.mlp_head2(x), self.mlp_head3(x), self.mlp_head4(x)

if __name__== '__main__':
    model=ViT().to(device="cuda:1")
    input1 = torch.randn(1, 1, 512, 2).to(device="cuda:1")
    flops,para=profile(model,inputs=(input1,))
    print(flops,para)