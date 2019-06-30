# -*- coding: utf-8 -*-
# date: 2018-12-02 11:35
import copy
import time

import torch.nn as nn

from .decoder import Decoder
from .decoder_layer import DecoderLayer
from .embeddings import Embeddings
from .encoder import Encoder
from .encoder_decoder import EncoderDecoder
from .encoder_layer import EncoderLayer
from .generator import Generator
from .multihead_attention import MultiHeadAttention
from .pointerwise_feedforward import PointerwiseFeedforward
from .positional_encoding import PositionalEncoding


def make_model(tgt_vocab, n=6, d_model=512, d_ff=2048, h=8, dropout=0.1, att=False, src_position=False):
    """
    Helper: Construct a model from hyperparameters.
    """
    c = copy.deepcopy
    attn = MultiHeadAttention(h, d_model)
    ff = PointerwiseFeedforward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(d_model, in_channels=2048, att=att,
                layer=EncoderLayer(d_model, c(attn), c(ff), dropout), n=n, src_posi=src_position),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), n),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab)
    )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model

