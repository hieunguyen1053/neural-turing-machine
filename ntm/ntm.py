from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .controller import Controller
from .memory import Memory
from .read_head import ReadHead
from .write_head import WriteHead


class NTM(nn.Module):
    def __init__(self, num_inputs, num_outputs, controller: Controller, memory: Memory, heads: List[Union[ReadHead, WriteHead]]):
        super(NTM, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.controller = controller
        self.memory = memory
        self.heads = heads

        self.N, self.M = memory.size
        _, self.controller_size = controller.size
        self.fc = nn.Linear(self.controller_size + self.num_read_heads * self.M, num_outputs)
        self.reset_parameters()

        self.num_read_heads = 0
        self.init_r = []
        for head in heads:
            if head.is_read_head():
                init_r_bias = torch.randn(1, self.M) * 0.01
                self.register_buffer("read{}_bias".format(
                    self.num_read_heads), init_r_bias.data)
                self.init_r += [init_r_bias]
                self.num_read_heads += 1

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc.weight, gain=1)
        nn.init.normal_(self.fc.bias, std=0.01)

    def create_new_state(self, batch_size: int):
        init_r = [r.clone().repeat(batch_size, 1) for r in self.init_r]
        controller_state = self.controller.create_new_state(batch_size)
        heads_state = [head.create_new_state(batch_size) for head in self.heads]

        return init_r, controller_state, heads_state

    def forward(self, x: Tensor, prev_state):
        # Unpack the previous state
        prev_reads, prev_controller_state, prev_heads_states = prev_state

        # Use the controller to get an embeddings
        inp = torch.cat([x] + prev_reads, dim=1)
        controller_outp, controller_state = self.controller(inp, prev_controller_state)

        # Read/Write from the list of heads
        reads = []
        heads_states = []
        for head, prev_head_state in zip(self.heads, prev_heads_states):
            if isinstance(head, ReadHead):
                r, head_state = head(controller_outp, prev_head_state)
                reads += [r]
                heads_states += [head_state]
            elif isinstance(head, WriteHead):
                head_state = head(controller_outp, prev_head_state)
                heads_states += [head_state]

        # Generate Output
        inp2 = torch.cat([controller_outp] + reads, dim=1)
        o = F.sigmoid(self.fc(inp2))

        # Pack the current state
        state = (reads, controller_state, heads_states)

        return o, state
