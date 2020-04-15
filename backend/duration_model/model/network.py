#!/usr/bin/env python3

# Copyright 2020 The Johns Hopkins University (author: Jiatong Shi)

import torch
import torch.nn as nn
import numpy as np
import math
import model.module as module
from torch.nn.init import xavier_uniform_
from torch.nn import Module, LayerNorm, Linear, Dropout

class TransformerDuration(Module):
    """Transformer encoder based diarization model.
    Args:
        d_model: the number of expected features in the encoder/decoder inputs.
        nhead: the number of heads in the multiheadattention models.
        num_encoder_layers: the number of sub-encoder-layers in the encoder.
        dim_feedforward: the dimension of the feedforward network model.
        dropout: the dropout value.
        pos_enc: True if positional encoding is used.
    """

    def __init__(self, embed_size=512, d_model=512, d_output=1,
                 nhead=4, num_block=6, phone_size=87,
                 dim_feedforward=2048, dropout=0.1, pos_enc=True,
                 ):
        super(TransformerDuration, self).__init__()

        self.input_embed = nn.Embedding(embed_size, phone_size)
        self.input_fc = nn.Linear(embed_size, d_model)
        self.speed_fc = nn.Linear(1, d_model)
        self.pos_encoder = module.PositionalEncoding(d_model, dropout)
        encoder_layer = module.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout)
        self.input_norm = LayerNorm(d_model)
        encoder_norm = LayerNorm(d_model)
        self.encoder = module.TransformerEncoder(
            encoder_layer, num_block, encoder_norm)
        self.output_fc = Linear(d_model, d_output)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead
        self.pos_enc = pos_enc

    def forward(self, src, speed, src_mask=None,
                src_key_padding_mask=None):
        embed = self.input_embed(src)
        embed = self.input_fc(embed)
        speed = self.speed_fc(embed)
        embed = embed + speed
        embed = self.input_norm(embed)
        if self.pos_enc:
            embed, att = self.encoder(embed) * math.sqrt(self.d_model)
            embed = self.pos_encoder(embed)
        memory, att = self.encoder(
            embed,
            mask=src_mask,
            src_key_padding_mask=src_key_padding_mask)
        output = self.output_fc(memory)
        return output, att

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)


def _test():
    # debug test

    import random
    random.seed(7)
    batch_size = 16
    max_length = 500
    char_max_length = 50
    feat_dim = 1324
    phone_size = 67
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    seq_len_list = []
    for i in range(batch_size):
        seq_len_list.append(random.randint(0, max_length))

    char_seq_len_list = []
    for i in range(batch_size):
        char_seq_len_list.append(random.randint(0, char_max_length))

    spec = torch.zeros(batch_size, max_length, feat_dim)
    phone = torch.zeros(batch_size, max_length, 1).long()
    pitch = torch.zeros(batch_size, max_length, 1).long()
    beat = torch.zeros(batch_size, max_length, 1).long()
    char = torch.zeros([batch_size, char_max_length, 1]).long()
    for i in range(batch_size):
        length = seq_len_list[i]
        char_length = char_seq_len_list[i]
        spec[i, :length, :] = torch.randn(length, feat_dim)
        phone[i, :length, :] = torch.randint(0, phone_size, (length, 1)).long()
        pitch[i, :length, :] = torch.randint(0, 200, (length, 1)).long()
        beat[i, :length, :] = torch.randint(0, 2, (length, 1)).long()
        char[i, :char_length, :] = torch.randint(0, phone_size, (char_length, 1)).long()

    seq_len = torch.from_numpy(np.array(seq_len_list)).to(device)
    char_seq_len = torch.from_numpy(np.array(char_seq_len_list)).to(device)
    spec = spec.to(device)
    phone = phone.to(device)
    pitch = pitch.to(device)
    beat = beat.to(device)
    print(seq_len.size())
    print(char_seq_len.size())
    print(spec.size())
    print(phone.size())
    print(pitch.size())
    print(beat.size())
    print(type(beat))

    hidden_size = 256
    embed_size = 256
    nhead = 4
    dropout = 0.1
    activation = 'relu'
    glu_kernel = 3
    num_dec_block = 3
    glu_num_layers = 1
    num_glu_block = 3
    
    #test encoder and encoer_postnet
    encoder = Encoder(phone_size, embed_size, hidden_size, dropout, num_glu_block, num_layers=glu_num_layers, glu_kernel=glu_kernel)
    encoder_out, text_phone = encoder(phone.squeeze(2))
    print('encoder_out.size():',encoder_out.size())
    
    post = Encoder_Postnet(embed_size)
    post_out = post(encoder_out, phone, text_phone, pitch.float(), beat)
    print('post_net_out.size():',post_out.size())
    
    
    # test model as a whole
    # model = GLU_Transformer(phone_size, hidden_size, embed_size, glu_num_layers, dropout, num_dec_block, nhead, feat_dim)
    # spec_pred = model(char, phone, pitch, beat, src_key_padding_mask=seq_len, char_key_padding_mask=char_seq_len)
    # print(spec_pred)

    # test decoder
    out_from_encoder = torch.zeros(batch_size, max_length, hidden_size)
    for i in range(batch_size):
        length = seq_len_list[i]
        out_from_encoder[i, :length, :] = torch.randn(length, hidden_size)
    decoder = Decoder(num_dec_block, embed_size, feat_dim, nhead, dropout)
    decoder_out, att = decoder(out_from_encoder, src_key_padding_mask=seq_len)
    print(decoder_out.size())
    print(att.size())



if __name__ == "__main__":
    _test()




