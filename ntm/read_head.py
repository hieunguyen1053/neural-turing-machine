import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .memory import Memory


def _split_cols(mat, lengths):
    assert mat.size()[1] == sum(lengths), "Lengths must be summed to num columns"
    l = np.cumsum([0] + lengths)
    results = []
    for s, e in zip(l[:-1], l[1:]):
        results += [mat[:, s:e]]
    return results


class ReadHead(nn.Module):
    def __init__(self, memory: Memory, controller_size: int):
        super(ReadHead, self).__init__()
        self.memory = memory
        self.controller_size = controller_size

        self.N, self.M = memory.size
        self.read_lengths = [self.M, 1, 1, 3, 1]
        self.fc_read = nn.Linear(controller_size, sum(self.read_lengths))
        self.reset_parameters()

    def _address_memory(self, k, beta, g, s, gamma, w_prev):
        k = k.clone()
        beta = F.softplus(beta)
        g = F.sigmoid(g)
        s = F.softmax(s, dim=1)
        gamma = 1 + F.softplus(gamma)

        w = self.memory.address(k, beta, g, s, gamma, w_prev)

        return w

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc_read.weight, gain=1.4)
        nn.init.normal_(self.fc_read.bias, std=0.01)

    def create_new_state(self, batch_size: int):
        return torch.zero(batch_size, self.N)

    def foward(self, input: Tensor, w_prev: Tensor):
        o = self.fc_read(input)
        k, beta, g, s, gamma = _split_cols(o, self.read_lengths)

        w = self._address_memory(k, beta, g, s, gamma, w_prev)
        r = self.memory.read(w)

        return r, w
