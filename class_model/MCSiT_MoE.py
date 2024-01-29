import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from einops import rearrange, repeat
# from task_moe.layer import TaskMoE
from task_moe.layer import TaskMoE
from einops.layers.torch import Rearrange
from thop import profile
from functools import partial


def one_hot_with_dtype(data, num_classes, dtype):
    result = torch.zeros([data.size(0),data.size(1), num_classes],
                         device=data.device,
                         dtype=dtype)
    result.scatter_(2, data, 1)
    return result


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.layer_norm1 = nn.LayerNorm(d_hid, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(d_hid, eps=1e-6)
        self.layer_norm3 = nn.LayerNorm(d_hid, eps=1e-6)
        self.layer_norm4 = nn.LayerNorm(d_hid, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

        MoE_layer = partial(
            TaskMoE,
            num_experts=24,
            k=2,
            capacity_factor=1.0,
            eval_capacity_factor=1.0,
            min_capacity=1.0,
            noisy_gate_policy='vmoe',
            use_rts=False,
            use_tutel=False,
            cfg=None,
        )
        self.linear1 = MoE_layer(hidden_size=d_in, expert=self.w_1).cuda()
        self.linear2 = MoE_layer(hidden_size=d_hid, expert=self.w_2).cuda()

    def forward(self, x):
        residual = x
        x = rearrange(x, 'b (h n) c -> b h n c', h=4)
        x1, gate_decision = self.linear1(x[:, 0, :, :])
        x1=self.layer_norm1(x1)
        x1, _ = self.linear2(self.dropout(F.relu(x1)),gate_decision=gate_decision)
        x2, gate_decision = self.linear1(x[:, 1, :, :])
        x2 = self.layer_norm2(x2)
        x2, _ = self.linear2(self.dropout(F.relu(x2)),gate_decision=gate_decision)
        x3, gate_decision = self.linear1(x[:, 2, :, :])
        x3 = self.layer_norm3(x3)
        x3, _ = self.linear2(self.dropout(F.relu(x3)),gate_decision=gate_decision)
        x4, gate_decision = self.linear1(x[:, 3, :, :])
        x4 = self.layer_norm4(x4)
        x4, _ = self.linear2(self.dropout(F.relu(x4)),gate_decision=gate_decision)

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = x+residual
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
    def __init__( self, n_layers, n_head, d_k, d_v,d_model, d_inner, dropout):
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
    def __init__(self, *,dim=128, depth=4, heads=6, mlp_dim=64,dim_head = 32*4, dropout = 0.6):
        super().__init__()
        self.to_patch_embedding1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(15, 2), stride=(1, 2), padding=(7, 0)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(32, 64, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(64, 128, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(128, 128, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),
            Rearrange('b c h w -> b (h w) c'),
        )

        self.to_patch_embedding2 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(15, 2), stride=(1, 2), padding=(7, 0)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(32, 64, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(64, 128, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(128, 128, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),
            Rearrange('b c h w -> b (h w) c'),
        )

        self.to_patch_embedding3 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(15, 2), stride=(1, 2), padding=(7, 0)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(32, 64, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(64, 128, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(128, 128, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),
            Rearrange('b c h w -> b (h w) c'),
        )

        self.to_patch_embedding4 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(15, 2), stride=(1, 2), padding=(7, 0)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(32, 64, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(64, 128, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(128, 128, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),

            Rearrange('b c h w -> b (h w) c'),
        )

        self.pe = torch.tensor([[pos / (10000.0 ** (i // 2 * 2.0 / dim)) for i in range(dim)] for pos in range(32)])
        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])
        self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])
        self.Positional_Encoding=nn.Parameter(self.pe, requires_grad=False).to(device="cuda:0")
        # self.Positional_Encoding = nn.Parameter(torch.zeros(1,32,dim)).to(device="cuda:0")
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

        x1 = self.to_patch_embedding1(x)
        x2 = self.to_patch_embedding2(x)
        x3 = self.to_patch_embedding3(x)
        x4 = self.to_patch_embedding4(x)
        x1 = x1 + self.Positional_Encoding
        x2 = x2 + self.Positional_Encoding
        x3 = x3 + self.Positional_Encoding
        x4 = x4 + self.Positional_Encoding
        x = torch.cat([x1, x2, x3, x4], 1)
        # x = x + self.Positional_Encoding
        enc_output= self.encoder(x)
        enc_output = rearrange(enc_output, 'b (h n) c -> b h n c', h=4)
        x = enc_output.mean(dim=2)
        x = self.to_latent(x)
        x1, x2, x3, x4 = x[:, 0, :], x[:, 1, :], x[:, 2, :], x[:, 3, :]
        return self.mlp_head1(x1), self.mlp_head2(x2), self.mlp_head3(x3), self.mlp_head4(x4)

if __name__== '__main__':
    model=ViT().cuda()
    input1 = torch.randn(2, 1, 512, 2).cuda()
    flops,para=profile(model,inputs=(input1,))
    print(flops,para)