'''
Copyright 2021 The Microsoft DeepSpeed Team
'''
# The file has been adapted from two fairscale files:
# (1) https://github.com/facebookresearch/fairscale/blob/master/fairscale/nn/moe/moe_layer.py
# (2) https://github.com/facebookresearch/fairscale/blob/master/fairscale/nn/moe/top2gate.py
# Git commit hash: 34df606902a240567a0d898037ece55c2f1336cf
# We retain the following license from the original files:

# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


from typing import Callable, Dict, TYPE_CHECKING, Any, Optional, Tuple, Union, cast

import time
from time import perf_counter
import torch
from torch import nn
from torch import Tensor
import torch.distributed as dist
from torch.nn import Module, ModuleList
import torch.nn.functional as F
from torch.cuda.amp import autocast
import numpy as np


if TYPE_CHECKING:
    Base = Module[Tensor]
else:
    Base = Module

uniform_map: Dict[torch.device, Callable] = {}
gumbel_map: Dict[torch.device, Callable] = {}
normal_map: Dict[torch.device, Callable] = {}
exp_selection_uniform_map: Dict[torch.device, Callable] = {}




def multiplicative_jitter(x, device: torch.device, epsilon=1e-2):
    """
    Modified from switch transformer paper. mesh transformers
    Multiply values by a random number between 1-epsilon and 1+epsilon.
    Makes models more resilient to rounding errors introduced by bfloat16.
    This seems particularly important for logits.
    Args:
        x: a torch.tensor
        device: torch.device
        epsilon: a floating point value
    Returns:
        a jittered x.
    """
    if epsilon == 0:
        return x
    uniform = uniform_map.get(device)
    if uniform is None:
        uniform = torch.distributions.uniform.Uniform(
            low=torch.tensor(1.0 - epsilon, device=device),
            high=torch.tensor(1.0 + epsilon,
                              device=device)).rsample  # type: ignore
        uniform_map[device] = uniform
    return x * uniform(x.shape)


def gumbel_rsample(shape: Tuple, device: torch.device) -> Tensor:
    gumbel = gumbel_map.get(device)
    if gumbel is None:
        one = torch.tensor(1.0, device=device)
        zero = torch.tensor(0.0, device=device)
        gumbel = torch.distributions.gumbel.Gumbel(zero,
                                                   one).rsample  # type: ignore
        gumbel_map[device] = gumbel
    return gumbel(shape)


def normal_rsample(shape: Tuple, device: torch.device, num_expert: int) -> Tensor:
    normal = normal_map.get(device)
    if normal is None:
        std = torch.tensor(1.0/num_expert, device=device)
        mean = torch.tensor(0.0, device=device)
        normal = torch.distributions.normal.Normal(mean, std).rsample  # type: ignore
        normal_map[device] = normal
    return normal(shape)


def one_hot_with_dtype(data, num_classes, dtype):
    result = torch.zeros([data.size(0), num_classes],
                         device=data.device,
                         dtype=dtype)
    result.scatter_(1, data.unsqueeze(-1), 1)
    return result

@torch.jit.script
def _top_idx(source, k):
    return torch.topk(source, k=k, dim=0)[1]


@torch.jit.script
def _one_hot_to_float(x, num_classes):
    return F.one_hot(x, num_classes=num_classes).float()




class TopKGate(nn.Module):
    """Gate module which implements Top2Gating as described in Gshard_.
    ::

        gate = TopKGate(model_dim, num_experts)
        l_aux, combine_weights, dispatch_mask = gate(input)

    .. Gshard_: https://arxiv.org/pdf/2006.16668.pdf

    Args:
        model_dim (int):
            size of model embedding dimension
        num_experts (ints):
            number of experts in model
    """

    # wg: torch.nn.Linear

    def __init__(self,
                 model_dim: int,
                 num_experts: int,
                 k: int = 1,
                 noisy_gate_policy: Optional[str] = None,
                 cfg: dict = None,
                 moe_type: str = None,
                 **kwargs):
        super().__init__( )

        if k != 1 and k != 2:
            raise ValueError('Only top-1 and top-2 gatings are supported.')
        # self.model_dim = model_dim
        self.k = k

        self.cfg = cfg

        self.noisy_gate_policy = noisy_gate_policy
        self.noise_std = 1


        #'deepspeed'



        self.layer_type = kwargs.pop('moe_type', 'ffn')


        self.moe_type = moe_type


        LayerNormModule = torch.nn.LayerNorm


        self.wg = torch.nn.Linear(model_dim, num_experts, bias=False).float()

        pass

    def forward(
        self,
        input,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Tensor]:  # type: ignore

        if self.wg.weight.dtype != torch.float32:
            self.wg = self.wg.float()
        input_fp32 = input.float()

        # input jittering
        if self.noisy_gate_policy == 'Jitter' and self.training:
            input_fp32 = multiplicative_jitter(input_fp32, device=input.device)

        logits = self.wg(input_fp32)
        # logits = logits.mean(1)
        #print(input.shape,'lofits is ',logits.shape)

        if self.k == 1:
            gate_output = self.top1gating(
                logits,
                self.noisy_gate_policy if self.training else None,
                **kwargs)

        # tutel gate function
        else:
            gate_output = self.top2gating(
                logits,
                self.noisy_gate_policy if self.training else None,
                **kwargs )
        return gate_output



    def top1gating(
            self,
            logits: Tensor,
            noisy_gate_policy: Optional[str] = None,
            **kwargs,
            ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Implements Top1Gating on logits."""

        logits_w_noise = None
        if noisy_gate_policy == 'RSample':
            logits_w_noise = logits + gumbel_rsample(logits.shape,
                                                     device=logits.device)
        else:
            logits_w_noise=logits
        # elif noisy_gate_policy == 'vmoe':
        #     num_experts = int(logits.shape[-1])
        #     logits_w_noise = logits + normal_rsample(logits.shape,
        #                                              device=logits.device,
        #                                              num_expert=num_experts/self.noise_std)

        # everything is in fp32 in this function
        gates = F.softmax(logits, dim=-1)
        # Create a mask for 1st's expert per token
        # noisy gating
        indices1_s = torch.argmax(logits_w_noise if logits_w_noise is not None else gates, dim=-1)

        num_experts = int(gates.shape[-1])
        mask1 = F.one_hot(indices1_s, num_classes=num_experts)



        gates = (gates*mask1).sum(dim=-1)


        return [indices1_s], [gates]




    def top2gating(
        self,
        logits: Tensor,
        noisy_gate_policy: Optional[str] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Implements Top2Gating on logits."""
        # everything is in fp32 in this function

        num_experts = int(logits.shape[-1])

        logits_w_noise = None
        if noisy_gate_policy == 'RSample':
            logits_w_noise = logits + gumbel_rsample(logits.shape,
                                                     device=logits.device) * self.noise_std
        elif noisy_gate_policy == 'vmoe':
            logits_w_noise = logits + normal_rsample(logits.shape,
                                                     device=logits.device,
                                                     num_expert=num_experts/self.noise_std)

        # topk_indices = torch.topk(logits, self.k, dim=1).indices
        # print('加噪的logit',logits.shape)
        topk_indices = torch.topk(
            logits_w_noise
            if logits_w_noise is not None else logits,
            self.k,
            dim=-1).indices
        #print('选择的topk',topk_indices.shape)
        #64, 2

        indices_s = [x.view(-1) for x in topk_indices.chunk(self.k, dim=-1)]
        #indices_s = [x.squeeze(-1) for x in topk_indices.chunk(self.k, dim=-1)]
        #print('***',indices_s[0].shape,indices_s[0])
        #2×64

        masks_se = [
            one_hot_with_dtype(x, num_classes=num_experts, dtype=x.dtype)
            for x in indices_s
        ]



        if noisy_gate_policy == 'vmoe':
            gates = F.softmax(logits_w_noise, dim=-1)

        else:
            gates = F.softmax(logits, dim=-1)

        #print('x is', masks_se[0].shape,'gates is', gates.shape)
        # self.load_balance(gates, masks_se[0], num_experts)
        gates_s = [(gates * x).sum(dim=-1) for x in masks_se]
        #print('gate的shape',gates_s[0].shape)

        # 2×128



        if self.k > 1:

            # Normalize Gate
            denom_s = torch.clamp(sum(gates_s),
                                  min=torch.finfo(gates_s[0].dtype).eps)
            gates_s = [x / denom_s for x in gates_s]

        # self.tb_output(mask1=None, exp_counts=None, gates=gates_s)

        return indices_s, gates_s


