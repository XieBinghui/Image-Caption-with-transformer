# -*- coding: utf-8 -*-
# date: 2018-11-29 19:56
import torch.nn as nn


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, tgt_mask):
        """
        Take in and process masked src and target sequences.
        """
        return self.decode(self.encode(src, att=None), tgt, tgt_mask)


    def encode(self, src, att=None):
        return self.encoder(src, att)

    def decode(self, memory, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, tgt_mask)
