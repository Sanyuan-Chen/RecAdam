# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""RecAdam optimizer"""

import logging
import math
import numpy as np

import torch
from torch.optim import Optimizer


logger = logging.getLogger(__name__)


def anneal_function(function, step, k, t0, weight):
    if function == 'sigmoid':
        return float(1 / (1 + np.exp(-k * (step - t0)))) * weight
    elif function == 'linear':
        return min(1, step / t0) * weight
    elif function == 'constant':
        return weight
    else:
        ValueError


class RecAdam(Optimizer):
    """ Implementation of RecAdam optimizer, a variant of Adam optimizer.

    Parameters:
        lr (float): learning rate. Default 1e-3.
        betas (tuple of 2 floats): Adams beta parameters (b1, b2). Default: (0.9, 0.999)
        eps (float): Adams epsilon. Default: 1e-6
        weight_decay (float): Weight decay. Default: 0.0
        correct_bias (bool): can be set to False to avoid correcting bias in Adam (e.g. like in Bert TF repository). Default True.
        anneal_fun (str): a hyperparam for the anneal function, decide the function of the curve. Default 'sigmoid'.
        anneal_k (float): a hyperparam for the anneal function, decide the slop of the curve. Choice: [0.05, 0.1, 0.2, 0.5, 1]
        anneal_t0 (float): a hyperparam for the anneal function, decide the middle point of the curve. Choice: [100, 250, 500, 1000]
        anneal_w (float): a hyperparam for the anneal function, decide the scale of the curve. Default 1.0.
        pretrain_cof (float): the coefficient of the quadratic penalty. Default 5000.0.
        pretrain_params (list of tensors): the corresponding group of params in the pretrained model.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.0, correct_bias=True,
                 anneal_fun='sigmoid', anneal_k=0, anneal_t0=0, anneal_w=1.0, pretrain_cof=5000.0, pretrain_params=None):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias,
                        anneal_fun=anneal_fun, anneal_k=anneal_k, anneal_t0=anneal_t0, anneal_w=anneal_w,
                        pretrain_cof=pretrain_cof, pretrain_params=pretrain_params)
        super().__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p, pp in zip(group["params"], group["pretrain_params"]):
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(1.0 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1.0 - beta2, grad, grad)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                # With RecAdam method, the optimization objective is
                # Loss = lambda(t)*Loss_T + (1-lambda(t))*Loss_S
                # Loss = lambda(t)*Loss_T + (1-lambda(t))*\gamma/2*\sum((\theta_i-\theta_i^*)^2)
                if group['anneal_w'] > 0.0:
                    # We calculate the lambda as the annealing function
                    anneal_lambda = anneal_function(group['anneal_fun'], state["step"], group['anneal_k'],
                                                    group['anneal_t0'], group['anneal_w'])
                    assert anneal_lambda <= group['anneal_w']
                    # The loss of the target task is multiplied by lambda(t)
                    p.data.addcdiv_(-step_size * anneal_lambda, exp_avg, denom)
                    # Add the quadratic penalty to simulate the pretraining tasks
                    p.data.add_(-group["lr"] * (group['anneal_w'] - anneal_lambda) * group["pretrain_cof"], p.data - pp.data)
                else:
                    p.data.addcdiv_(-step_size, exp_avg, denom)

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                if group["weight_decay"] > 0.0:
                    p.data.add_(-group["lr"] * group["weight_decay"], p.data)

        return loss
