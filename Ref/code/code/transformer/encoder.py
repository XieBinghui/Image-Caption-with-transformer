# -*- coding: utf-8 -*-
# date: 2018-11-29 20:07
import torch.nn as nn

from .functional import clones
from .layer_norm import LayerNorm
from .positional_encoding import PositionalEncoding

class Encoder(nn.Module):
    """
    Core encoder is a stack of N layers
    """
    def __init__(self, d_model, in_channels=2048, att=False, layer=None, n=None, src_posi=False, dropout=0.1):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.att = att
        self.src_posi = src_posi
        if att:
            self.layers = clones(layer, n)
            self.norm = LayerNorm(layer.size)
        if d_model != in_channels:
            self.linear = nn.Linear(in_channels, d_model)
            self.ReLU = nn.ReLU()
        if src_posi:
            self.position = PositionalEncoding(d_model, dropout)
    def forward(self, x, att=None):
        """
        Pass the input (and mask) through each layer in turn.
        """
        if self.d_model != x.size(-1):
            x = self.ReLU(self.linear(x))
            if self.src_posi:
                x = self.position(x)
        if self.att:
            for layer in self.layers:
                x = layer(x)
            return self.norm(x)
        return x