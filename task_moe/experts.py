'''
Copyright 2020 The Microsoft DeepSpeed Team
'''

import torch
import copy
from .gate import one_hot_with_dtype

import torch.nn.functional as F

from torch.cuda.amp import autocast


class FusedExperts(torch.nn.Module):
    def __init__(self, expert, cfg,  num_local_experts=1):
        super(FusedExperts, self).__init__()
        self.cfg = cfg

        self.deepspeed_experts = torch.nn.ModuleList(
            [copy.deepcopy(expert) for i in range(num_local_experts)])
        self.num_local_experts = num_local_experts

        # self.bias_merge = self.deepspeed_experts[0].bias is not None


    def top1_expert_forward(self, x, indice, gate, mode=None, **kwargs):
        assert  mode is None, "unified qkv inference is not supported for top1"
            #unimodal
        x = self.deepspeed_experts[indice[0]](x) * gate[0].to(x)
        return x

    def mergelayer(self, x,  index1, index2, gate1, gate2, mode=None):
        



            return self.deepspeed_experts[index1](x) * gate1 + self.deepspeed_experts[index2](x) * gate2

        

    def top2_expert_forward(self, x, indices, gates, mode=None, **kwargs):

        # caption eval mode
        # 2×128 #2×128
        # unimodal
        #print(indices[0].shape,gates[0].shape)
        #print(indices[0])
        k1=torch.mode(indices[0])[1]
        k2=torch.mode(indices[1])[1]
        #print(k1,k2)

        # x = self.mergelayer(x, k1, k2, k1, k2, mode=mode)
        x = self.mergelayer(x, indices[0][0], indices[1][0], gates[0][0], gates[1][0], mode=mode)


        return x

    def forward(self, hidden_states, top_indices=None, gates=None, **kwargs):

        # top1#2×128×20 #2×128
        if len(top_indices) == 1:
            out = self.top1_expert_forward(hidden_states, top_indices[0], gates[0], **kwargs)

        # top2
        elif len(top_indices) == 2:

            out = self.top2_expert_forward(hidden_states, top_indices, gates, **kwargs)

        else:
            raise NotImplementedError("only support top1 and top2 ")
        # print(out.shape,hidden_states.shape)



        # assert out.shape[1] == hidden_states.shape[1]

        return out
