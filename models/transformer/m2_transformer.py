from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from icecream import ic

from .containers import Module
from .decoders import MeshedDecoder
from .encoders import MemoryAugmentedEncoder, DualPathMemoryAugmentedEncoder
from .attention import ScaledDotProductAttentionMemory, ScaledDotProductAttention
from .beam_search import BeamSearch
from .utils import TensorOrSequence, get_batch_size, get_device


class M2Transformer(Module):
    def __init__(self, vocab, max_seq_len, object_latent_dim, padding_idx):
        super(M2Transformer, self).__init__()
        self.padding_idx = padding_idx
        self.bos_idx = vocab['word2idx']['sos']
        self.eos_idx = vocab['word2idx']['eos']
        self.vocab = vocab

        self.encoder = MemoryAugmentedEncoder(3, 0, d_in=object_latent_dim,
                                              attention_module=ScaledDotProductAttentionMemory,
                                              attention_module_kwargs={'m': 40}
                                              )
        self.decoder = MeshedDecoder(len(vocab["word2idx"]), max_seq_len, 1, padding_idx)

        self.register_state('enc_output', None)
        self.register_state('mask_enc', None)

    def forward(self, objects_features, tokens):
        # input (b_s, seq_len, d_in)
        mask_enc = (torch.sum(objects_features, -1) == self.padding_idx).unsqueeze(1).unsqueeze(
            1)  # (b_s, 1, 1, seq_len)

        objects_features = self.encoder(objects_features, mask_enc)  # (B, 3, n_object, 512)

        dec_outputs, intermediate_feats = self.decoder(tokens, objects_features, mask_enc)  # (B, max_len, vocab_size)

        return dec_outputs, intermediate_feats, objects_features

    def step(self, t, prev_output, visual, seq, mode='teacher_forcing', **kwargs):
        it = None
        if mode == 'teacher_forcing':
            raise NotImplementedError
        elif mode == 'feedback':
            if t == 0:
                self.mask_enc = (torch.sum(visual, -1) == self.padding_idx).unsqueeze(1).unsqueeze(
                    1)  # (b_s, 1, 1, seq_len)
                self.enc_output = self.encoder(visual, self.mask_enc)
                if isinstance(visual, torch.Tensor):
                    it = visual.data.new_full((visual.shape[0], 1), self.bos_idx).long()
                else:
                    it = visual[0].data.new_full((visual[0].shape[0], 1), self.bos_idx).long()
            else:
                it = prev_output

        output = self.decoder(it, self.enc_output, self.mask_enc)[0]
        return F.log_softmax(output, dim=-1)

    def beam_search(self, visual: TensorOrSequence, max_len: int, beam_size: int, out_size=1,
                    return_probs=False, **kwargs):
        bs = BeamSearch(self, max_len, self.eos_idx, beam_size)
        return bs.apply(visual, out_size, return_probs, **kwargs)


class DualM2Transformer(Module):
    def __init__(self, vocab, max_seq_len, object_latent_dim, padding_idx):
        super(DualM2Transformer, self).__init__()
        self.padding_idx = padding_idx
        self.bos_idx = vocab['word2idx']['sos']
        self.eos_idx = vocab['word2idx']['eos']
        self.vocab = vocab

        self.encoder = DualPathMemoryAugmentedEncoder(3, 0, d_in=object_latent_dim,
                                              attention_module=ScaledDotProductAttentionMemory,
                                              attention_module_kwargs={'m': 40})
        # self.decoder_t = MeshedDecoder(len(vocab["word2idx"]), max_seq_len, 1, padding_idx)
        self.decoder = MeshedDecoder(len(vocab["word2idx"]), max_seq_len, 1, padding_idx)

        self.register_state('enc_output', None)
        self.register_state('mask_enc', None)

    def forward(self, feats, extra_feats, tokens):
        # input (b_s, seq_len, d_in)
        mask_enc = (torch.sum(feats, -1) == self.padding_idx).unsqueeze(1).unsqueeze(
            1)  # (b_s, 1, 1, seq_len)

        feats = self.encoder(feats, extra_feats, mask_enc)  # (B, 3, n_object, 512)

        dec_outputs, intermediate_feats = self.decoder(tokens, feats, mask_enc)  # (B, max_len, vocab_size)

        return dec_outputs, intermediate_feats

    def step(self, t, prev_output, visual, seq, mode='teacher_forcing', **kwargs):
        it = None
        if mode == 'teacher_forcing':
            raise NotImplementedError
        elif mode == 'feedback':
            if t == 0:
                self.mask_enc = (torch.sum(visual[0], -1) == self.padding_idx).unsqueeze(1).unsqueeze(
                    1)  # (b_s, 1, 1, seq_len)
                self.enc_output = self.encoder(visual[0], visual[1], self.mask_enc)
                if isinstance(visual, torch.Tensor):
                    it = visual.data.new_full((visual.shape[0], 1), self.bos_idx).long()
                else:
                    it = visual[0].data.new_full((visual[0].shape[0], 1), self.bos_idx).long()
            else:
                it = prev_output

        output = self.decoder(it, self.enc_output, self.mask_enc)[0]
        return F.log_softmax(output, dim=-1)

    def beam_search(self, visual: TensorOrSequence, max_len: int, beam_size: int, out_size=1,
                    return_probs=False, **kwargs):
        bs = BeamSearch(self, max_len, self.eos_idx, beam_size)
        return bs.apply(visual, out_size, return_probs, **kwargs)