import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Memory(nn.Module):
    def __init__(self, N: int, M: int):
        super(Memory, self).__init__()
        self.N = N
        self.M = M

        self.register_buffer('mem_bias', torch.tensor(N, M))
        stdev = 1/np.sqrt(N + M)
        nn.init.uniform_(self.mem_bias, -stdev, stdev)

    def reset(self, batch_size: int):
        self.batch_size = batch_size
        self.memory = self.mem_bias.clone().repeat(batch_size, 1, 1)

    @property
    def size(self):
        return self.N, self.M

    def read(self, w: Tensor):
        return torch.matmul(w.unsqueeze(1), self.memory).squeeze(1)

    def write(self, w: Tensor, e: Tensor, a: Tensor):
        self.prev_mem = self.memory
        self.memory = torch.tensor(self.batch_size, self.N, self.M)

        erase = torch.matmul(w.unsqueeze(-1), e.unsqueeze(1))
        add = torch.matmul(w.unsqueeze(-1), a.unsqueeze(1))
        self.memory = self.prev_mem * (1 - erase) + add

    def address(self, k, beta, g, s, gamma, w_prev):
        # Content focus
        wc = self._similarity(k, beta)

        # Location focus
        wg = self._interpolate(w_prev, wc, g)
        w_hat = self._shift(wg, s)
        w = self._sharpen(w_hat, gamma)
        return w

    def _similarity(self, k, beta):
        k = k.view(self.batch_size, 1, -1)
        w = F.softmax(beta * F.cosine_similarity(self.memory +
                      1e-16, k + 1e-16, dim=-1), dim=1)
        return w

    def _interpolate(self, w_prev, wc, g):
        return g * wc + (1 - g) * w_prev

    def _shift(self, wg, s):
        result = torch.zeros(wg.size())
        for b in range(self.batch_size):
            result[b] = self._convolve(wg[b], s[b])
        return result

    def _sharpen(self, w_hat, gamma):
        w = w_hat ** gamma
        w = torch.div(w, torch.sum(w, dim=1).view(-1, 1) + 1e-16)
        return w

    def _convolve(w, s):
        assert s.size(0) == 3
        t = torch.cat([w[-1:], w, w[:1]])
        c = F.conv1d(t.view(1, 1, -1), s.view(1, 1, -1)).view(-1)
        return c
