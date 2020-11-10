# coding=utf-8

"""
A very basic implementation of neural machine translation

Usage:
    nmt.py --mode=<mode> --train-src=<file> --train-tgt=<file> --dev=<file> --vocab=<file> [options]
    nmt.py --mode=<mode> [options] MODEL_PATH TEST_FILE TEST_FILE_SP OUTPUT_FILE
    nmt.py --mode=<mode> [options] MODEL_PATH TEST_FILE OUTPUT_FILE

Options:
    -h --help                               show this screen.
    --gpu_id=<int>                          GPU ID
    --mode=<mode>                           train or decode
    --cuda                                  use GPU
    --copy=<bool>                           apply copy mechanism [default: False]
    --coverage=<bool>                       apply coverage mechanism [default: False]
    --train-src=<file>                      train source file
    --train-tgt=<file>                      train target file
    --dev=<file>                            dev source and targets file
    --vocab=<file>                          vocab file
    --seed=<int>                            seed [default: 5783287]
    --batch-size=<int>                      batch size [default: 64]
    --embed-size=<int>                      embedding size [default: 256]
    --hidden-size=<int>                     hidden size [default: 256]
    --clip-grad=<float>                     gradient clipping [default: 5.0]
    --label-smoothing=<float>               use label smoothing [default: 0.0]
    --log-every=<int>                       log every [default: 500]
    --max-epoch=<int>                       max epoch [default: 999]
    --input-feed                            use input feeding
    --patience=<int>                        wait for how many iterations to decay learning rate [default: 5]
    --max-num-trial=<int>                   terminate training after how many trials [default: 5]
    --lr-decay=<float>                      learning rate decay [default: 0.5]
    --beam-size=<int>                       beam size [default: 10]
    --lr=<float>                            learning rate [default: 0.0005]
    --uniform-init=<float>                  uniformly initialize all parameters [default: 0.1]
    --save-to=<file>                        model save path [default: model.bin]
    --valid-niter=<int>                     perform validation after how many iterations [default: 2000]
    --dropout=<float>                       dropout [default: 0.2]
    --max-decoding-time-step=<int>          maximum number of decoding time steps [default: 80]
    --load-model=<str>                      continue training [default: False]
    --num-trial=<int>                       having trained for how many trials [default: 0]
    --best-r2=<float>                       best rouge2 f1 score for the trained models [default: 0.0]
    --best-ppl=<float>                      minimum ppl score for the trained models [default: 1]

    --stop-words-file=<file>                stop-word file path [default: data/stopwords_ch]
    --exclusive-words-file=<file>           exclusive-words file path [default: data/exclusive_words]

    --train-kb-table-file=<file>            train kb kv-pairs file path [default: data/train_kb_acl]
    --dev-batch-size=<int>                  dev batch size [default: 20]
"""
from __future__ import print_function
import math
import pickle
import sys
import time
from collections import namedtuple

import numpy as np
from typing import List, Tuple, Dict, Set, Union
from docopt import docopt
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction

import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from utils import read_corpus_json, read_corpus, batch_iter, LabelSmoothingLoss, \
    read_stopwords, replace_digit, read_kb_table, forbid_duplicate, batch_iter_dev
from vocab import Vocab, VocabEntry

from rougescore import *
import copy
import re

Hypothesis = namedtuple('Hypothesis', ['value', 'score'])


class NMT(nn.Module):

    def __init__(self, embed_size, hidden_size, vocab, dropout_rate=0.2, input_feed=True, label_smoothing=0.,
                 copy=False, coverage=False):
        super(NMT, self).__init__()

        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.vocab = vocab
        self.input_feed = input_feed
        self.copy = copy
        self.coverage = coverage

        # initialize neural network layers...

        self.src_embed = nn.Embedding(len(vocab.src), embed_size, padding_idx=vocab.src['<pad>'])
        self.tgt_embed = nn.Embedding(len(vocab.tgt), embed_size, padding_idx=vocab.tgt['<pad>'])

        self.encoder_lstm = nn.LSTM(embed_size, hidden_size, bidirectional=True)
        decoder_lstm_input = embed_size + hidden_size if self.input_feed else embed_size
        self.decoder_lstm = nn.LSTMCell(decoder_lstm_input, hidden_size)

        # attention: dot product attention
        # project source encoding to decoder rnn's state space
        self.att_src_linear = nn.Linear(hidden_size * 2, hidden_size, bias=False)

        # transformation of decoder hidden states and context vectors before reading out target words
        # this produces the `attentional vector` in (Luong et al., 2015)
        self.att_vec_linear = nn.Linear(hidden_size * 2 + hidden_size, hidden_size, bias=False)

        self.att_ht_linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.att_v_linear = nn.Linear(hidden_size, 1, bias=False)

        # prediction layer of the target vocabulary
        self.readout = nn.Linear(hidden_size, len(vocab.tgt), bias=False)

        # dropout layer
        self.dropout = nn.Dropout(self.dropout_rate)

        # initialize the decoder's state and cells with encoder hidden states
        self.decoder_cell_init = nn.Linear(hidden_size * 2, hidden_size)

        if self.copy:
            self.p_linear = nn.Linear(hidden_size * 3 + embed_size, 1)

        self.label_smoothing = label_smoothing
        if label_smoothing > 0.:
            self.label_smoothing_loss = LabelSmoothingLoss(label_smoothing,
                                                           tgt_vocab_size=len(vocab.tgt),
                                                           padding_idx=vocab.tgt['<pad>'])

    @property
    def device(self):
        return self.src_embed.weight.device

    def forward(self, src_sents, tgt_sents, tgt_vocab_mask):
        """
        take a mini-batch of source and target sentences, compute the log-likelihood of
        target sentences.

        Args:
            src_sents: list of source sentence tokens
            tgt_sents: list of target sentence tokens, wrapped by `<s>` and `</s>`

        Returns:
            scores: a variable/tensor of shape (batch_size, ) representing the
                log-likelihood of generating the gold-standard target sentence for
                each example in the input batch
        """
        
        # (src_sent_len, batch_size); (tgt_sent_len, batch_size)
        src_sents_var = self.vocab.src.to_input_tensor_src(src_sents, device=self.device)
        if self.copy:
            (tgt_sents_var, src_complete_sents_var, tgt_complete_sents_var,
             word_oovs, max_oov_num) = self.vocab.tgt.to_input_tensor_tgt(src_sents, tgt_sents, device=self.device)
        else:  # TO DO
            src_sents_var = self.vocab.src.to_input_tensor_src(src_sents, device=self.device)
            tgt_sents_var = self.vocab.tgt.to_input_tensor_src(tgt_sents, device=self.device)

        src_sents_len = [len(s) for s in src_sents]

        src_encodings, decoder_init_vec = self.encode(src_sents_var, src_sents_len)

        src_sent_masks = self.get_attention_mask(src_encodings, src_sents_len)

        # (tgt_sent_len - 1, batch_size, hidden_size)
        if self.copy and self.coverage:
            h_ts, ctx_ts, alpha_ts, att_vecs, tgt_word_embeds, coverages = self.decode(src_encodings, src_sent_masks,
                                                                                       decoder_init_vec,
                                                                                       tgt_sents_var[:-1])
        elif self.coverage:
            att_vecs, coverages = self.decode(src_encodings, src_sent_masks, decoder_init_vec, tgt_sents_var[:-1])
        elif self.copy:
            h_ts, ctx_ts, alpha_ts, att_vecs, tgt_word_embeds = self.decode(src_encodings, src_sent_masks,
                                                                            decoder_init_vec, tgt_sents_var[:-1])
            # h_ts:            (tgt_sent_len - 1, batch_size, hidden_size)
            # ctx_ts:          (tgt_sent_len - 1, batch_size, hidden_size * 2)
            # alpha_ts:        (tgt_sent_len - 1, batch_size, src_sent_len)
            # att_vecs:        (tgt_sent_len - 1, batch_size, hidden_size)
            # tgt_word_embeds: (tgt_sent_len - 1, batch_size, embed-size)
        else:
            att_vecs = self.decode(src_encodings, src_sent_masks, decoder_init_vec, tgt_sents_var[:-1])

        # (tgt_sent_len - 1, batch_size, tgt_vocab_size)
        if self.copy:
            scores = self.readout(att_vecs)
            scores.data.masked_fill_(tgt_vocab_mask.byte(), -float('inf')) 
            tgt_words_log_prob = F.softmax(scores, dim=-1)
            tgt_words_log_prob = torch.clamp(tgt_words_log_prob, 1e-9, 1 - 1e-9)
#            print(tgt_words_log_prob)
#            tgt_words_log_prob = tgt_words_log_prob.data.masked_fill_(tgt_vocab_mask.byte(), float(1e-9))
#            print("---------------------------------------")
#            print(tgt_words_log_prob)
#            print("#######################################")
            if max_oov_num > 0:
                oov_zeros = torch.zeros(tgt_words_log_prob.size(0), tgt_words_log_prob.size(1), max_oov_num,
                                        device=self.device)
                tgt_words_log_prob = torch.cat([tgt_words_log_prob, oov_zeros], dim=-1)

            p = torch.cat([h_ts, ctx_ts, tgt_word_embeds], dim=-1)
            g = torch.sigmoid(self.p_linear(p))
            """ avoid loss nan """
            w = torch.clone(g)
            w[w == 0] = 1e-6
            w[w == 1] = 1 - 1e-6
            g = w

            src_complete_sents_var_expanded = src_complete_sents_var.permute(1, 0).expand(alpha_ts.size(0), -1, -1)
            tgt_words_log_prob = (g * tgt_words_log_prob).scatter_add(2, src_complete_sents_var_expanded,
                                                                      (1 - g) * alpha_ts)

            # tgt_words_log_prob = torch.log(tgt_words_log_prob)

        else:
            tgt_words_log_prob = F.log_softmax(self.readout(att_vecs), dim=-1)

        # (tgt_sent_len, batch_size)
        tgt_words_mask = (tgt_sents_var != self.vocab.tgt['<pad>']).float()

        if self.label_smoothing:
            if self.copy:
                # (tgt_sent_len - 1, batch_size)
                tgt_gold_words_log_prob = self.label_smoothing_loss(
                    tgt_words_log_prob.view(-1, tgt_words_log_prob.size(-1)),
                    tgt_complete_sents_var[1:].view(-1), max_oov_num, self.copy).view(-1, len(tgt_sents))
            else:
                tgt_gold_words_log_prob = self.label_smoothing_loss(
                    tgt_words_log_prob.view(-1, tgt_words_log_prob.size(-1)),
                    tgt_sents_var[1:].view(-1)).view(-1, len(tgt_sents))
        else:

            # (tgt_sent_len - 1, batch_size)
            if self.copy:
                tgt_gold_words_log_prob = torch.gather(tgt_words_log_prob,
                                                       index=tgt_complete_sents_var[1:].unsqueeze(-1), dim=-1).squeeze(
                    -1)
                tgt_gold_words_log_prob = torch.log(tgt_gold_words_log_prob) * tgt_words_mask[1:]
            else:
                tgt_gold_words_log_prob = torch.gather(tgt_words_log_prob, index=tgt_sents_var[1:].unsqueeze(-1),
                                                       dim=-1).squeeze(-1) * tgt_words_mask[1:]

        # (batch_size)
        # scores = tgt_gold_words_log_prob.sum(dim=0)
        scores = tgt_gold_words_log_prob.sum(dim=0) / torch.sum(tgt_words_mask[1:], 0)

        if self.coverage:
            return scores, coverages
        else:
            return scores

    def get_attention_mask(self, src_encodings, src_sents_len):
        src_sent_masks = torch.zeros(src_encodings.size(0), src_encodings.size(1), dtype=torch.float)
        for e_id, src_len in enumerate(src_sents_len):
            src_sent_masks[e_id, src_len:] = 1

        return src_sent_masks.to(self.device)

    def encode(self, src_sents_var, src_sent_lens):
        """
        Use a GRU/LSTM to encode source sentences into hidden states

        Args:
            src_sents: list of source sentence tokens

        Returns:
            src_encodings: hidden states of tokens in source sentences, this could be a variable
                with shape (batch_size, source_sentence_length, encoding_dim), or in orther formats
            decoder_init_state: decoder GRU/LSTM's initial state, computed from source encodings
        """

        # (src_sent_len, batch_size, embed_size)
        src_word_embeds = self.src_embed(src_sents_var)
        packed_src_embed = pack_padded_sequence(src_word_embeds, src_sent_lens)

        # src_encodings: (src_sent_len, batch_size, hidden_size * 2)
        src_encodings, (last_state, last_cell) = self.encoder_lstm(packed_src_embed)
        src_encodings, _ = pad_packed_sequence(src_encodings)

        # (batch_size, src_sent_len, hidden_size * 2)
        src_encodings = src_encodings.permute(1, 0, 2)

        dec_init_cell = self.decoder_cell_init(torch.cat([last_cell[0], last_cell[1]], dim=1))
        dec_init_state = torch.tanh(dec_init_cell)

        return src_encodings, (dec_init_state, dec_init_cell)

    def decode(self, src_encodings, src_sent_masks, decoder_init_vec, tgt_sents_var):
        """
        Given source encodings, compute the log-likelihood of predicting the gold-standard target
        sentence tokens

        Args:
            src_encodings: hidden states of tokens in source sentences
            decoder_init_state: decoder GRU/LSTM's initial state
            tgt_sents: list of gold-standard target sentences, wrapped by `<s>` and `</s>`

        Returns:
            scores: could be a variable of shape (batch_size, ) representing the
                log-likelihood of generating the gold-standard target sentence for
                each example in the input batch
        """

        # (batch_size, src_sent_len, hidden_size)
        src_encoding_att_linear = self.att_src_linear(src_encodings)

        batch_size = src_encodings.size(0)

        # initialize the attentional vector
        att_tm1 = torch.zeros(batch_size, self.hidden_size, device=self.device)

        # (tgt_sent_len, batch_size, embed_size)
        # here we omit the last word, which is always </s>.
        # Note that the embedding of </s> is not used in decoding
        tgt_word_embeds = self.tgt_embed(tgt_sents_var)

        h_tm1 = decoder_init_vec

        att_ves = []

        if self.copy:
            h_ts = []
            ctx_ts = []
            alpha_ts = []

        if self.coverage:
            att_history = None
            coverages = []

        # start from y_0=`<s>`, iterate until y_{T-1}
        for y_tm1_embed in tgt_word_embeds.split(split_size=1):
            y_tm1_embed = y_tm1_embed.squeeze(0)
            if self.input_feed:
                # input feeding: concate y_tm1 and previous attentional vector
                # (batch_size, hidden_size + embed_size)

                x = torch.cat([y_tm1_embed, att_tm1], dim=-1)
            else:
                x = y_tm1_embed

            if self.coverage:
                (h_t, cell_t), ctx_t, alpha_t, att_t = self.step(x, h_tm1, src_encodings, src_encoding_att_linear,
                                                                 src_sent_masks, att_history)
                if att_history is None:
                    att_history = alpha_t
                else:
                    coverage = torch.min(alpha_t, att_history)
                    coverages.append(coverage)
                    att_history = att_history + alpha_t
            else:
                (h_t, cell_t), ctx_t, alpha_t, att_t = self.step(x, h_tm1, src_encodings, src_encoding_att_linear,
                                                                 src_sent_masks, None)

            att_tm1 = att_t
            h_tm1 = h_t, cell_t
            att_ves.append(att_t)
            if self.copy:
                h_ts.append(h_t)
                ctx_ts.append(ctx_t)
                alpha_ts.append(alpha_t)

        # (tgt_sent_len - 1, batch_size, tgt_vocab_size)
        att_ves = torch.stack(att_ves)

        if self.copy:
            h_ts = torch.stack(h_ts)
            ctx_ts = torch.stack(ctx_ts)
            alpha_ts = torch.stack(alpha_ts)

        if self.coverage:
            coverages = torch.stack(coverages)

        if self.copy and self.coverage:
            return h_ts, ctx_ts, alpha_ts, att_ves, tgt_word_embeds, coverages
        elif self.coverage:
            return att_ves, coverages
        elif self.copy:
            return h_ts, ctx_ts, alpha_ts, att_ves, tgt_word_embeds
        else:
            return att_ves

    def step(self, x, h_tm1, src_encodings, src_encoding_att_linear, src_sent_masks, att_history):
        # h_t: (batch_size, hidden_size)
        h_t, cell_t = self.decoder_lstm(x, h_tm1)  # h1 -> h2

        ctx_t, alpha_t = self.dot_prod_attention(h_t, src_encodings, src_encoding_att_linear, src_sent_masks,
                                                 att_history)

        att_t = torch.tanh(self.att_vec_linear(torch.cat([h_t, ctx_t], 1)))  # E.q. (5)
        att_t = self.dropout(att_t)

        return (h_t, cell_t), ctx_t, alpha_t, att_t

    def dot_prod_attention(self, h_t, src_encoding, src_encoding_att_linear, mask, att_history):
        # (batch_size, src_sent_len, hidden_size) * (batch_size, hidden_size, 1) = (batch_size, src_sent_len)
        # att_weight = torch.bmm(src_encoding_att_linear, h_t.unsqueeze(2)).squeeze(2)

        if self.coverage and att_history is not None:
            att_hidden_text = torch.tanh(self.att_ht_linear(h_t).unsqueeze(1).expand_as(src_encoding_att_linear) +
                                         att_history.unsqueeze(2).expand_as(
                                             src_encoding_att_linear) + src_encoding_att_linear)
        else:
            att_hidden_text = torch.tanh(
                self.att_ht_linear(h_t).unsqueeze(1).expand_as(src_encoding_att_linear) + src_encoding_att_linear)

        # (batch_size, src_sent_len)
        att_weight = self.att_v_linear(att_hidden_text).squeeze(2)

        if mask is not None:
            att_weight.data.masked_fill_(mask.byte(), -float('inf'))

        softmaxed_att_weight = F.softmax(att_weight, dim=-1)

        att_view = (att_weight.size(0), 1, att_weight.size(1))
        # (batch_size, hidden_size)
        ctx_vec = torch.bmm(softmaxed_att_weight.view(*att_view), src_encoding).squeeze(1)

        return ctx_vec, softmaxed_att_weight

    def beam_search(self, idx_unsort_dev, src_sents, first_uni2id, first_bi2id, tri2id, beam_size, max_decoding_time_step, stopwords_set, args, tgt_vocab_masks):
        """
        self, src_enc, src_len, tgt_lang_id, max_len=200, sample_temperature=None, x1=None):
        Decode a sentence given initial start.
        `x`:
            - LongTensor(bs, slen)
                <EOS> W1 W2 W3 <EOS> <PAD>
                <EOS> W1 W2 W3   W4  <EOS>
        `lengths`:
            - LongTensor(bs) [5, 6]
        `positions`:
            - False, for regular "arange" positions (LM)
            - True, to reset positions from the new generation (MT)
        `langs`:
            - must be None if the model only supports one language
            - lang_id if only one language is involved (LM)
            - (lang_id1, lang_id2) if two languages are involved (MT)
        """
        
        # check inputs
        # assert src_enc.size(0) == src_len.size(0)
        assert beam_size >= 1

        # batch size / number of words
        bs = len(src_sents)  # batch_size
        n_words = len(self.vocab.tgt)  # self.n_words
        src_sents_len = [len(s) for s in src_sents]
        """ alpha_w """

        src_sents_var = self.vocab.src.to_input_tensor_src(src_sents, device=self.device)
        src_encodings, dec_init_vec = self.encode(src_sents_var, src_sents_len)
        src_encodings_att_linear = self.att_src_linear(src_encodings)
        src_sent_masks = self.get_attention_mask(src_encodings, src_sents_len)

        if self.copy:
            (src_complete_sents_var, word_oovs, max_oov_num) = self.vocab.tgt.to_input_tensor_tgt_decode(src_sents, device=self.device)

            n_words += max_oov_num

        eos_id = self.vocab.tgt['</s>']

        h_tm1 = []
        for i in range(2):
            h_tm1.append(dec_init_vec[i].unsqueeze(1).expand((bs, beam_size) + dec_init_vec[i].shape[1:]).contiguous().view((bs * beam_size,) + dec_init_vec[i].shape[1:]))
        h_tm1 = tuple(h_tm1)

        att_tm1 = torch.zeros(bs * beam_size, self.hidden_size, device=self.device)

        src_encodings = src_encodings.unsqueeze(1).expand((bs, beam_size) + src_encodings.shape[1:]).contiguous().view((bs * beam_size,) + src_encodings.shape[1:])
        src_encodings_att_linear = src_encodings_att_linear.unsqueeze(1).expand((bs, beam_size) + src_encodings_att_linear.shape[1:]).contiguous().view((bs * beam_size,) + src_encodings_att_linear.shape[1:])
        src_sent_masks = src_sent_masks.unsqueeze(1).expand((bs, beam_size) + src_sent_masks.shape[1:]).contiguous().view((bs * beam_size,) + src_sent_masks.shape[1:])
        
        if self.copy:
            src_complete_sents_var = src_complete_sents_var.unsqueeze(2).expand(src_complete_sents_var.shape[:1] + (bs, beam_size)).contiguous().view(src_complete_sents_var.shape[:1] + (bs * beam_size,))

        src_sents_len = torch.tensor(src_sents_len, device=self.device)
        src_sents_len = src_sents_len.unsqueeze(1).expand(bs, beam_size).contiguous().view(-1)

        # generated sentences (batch with beam current hypotheses)
        generated = src_sents_len.new(max_decoding_time_step, bs * beam_size)  # upcoming output
        generated.fill_(self.vocab.tgt['<pad>'])                   # fill upcoming ouput with <PAD>
        generated[0].fill_(self.vocab.tgt['<s>'])                # we use <EOS> for <BOS> everywhere

        # generated hypotheses
        generated_hyps = [BeamHypotheses(beam_size, max_decoding_time_step, length_penalty=1.0, early_stopping=False) for _ in range(bs)]



        # scores for each sentence in the beam
        beam_scores = src_encodings.new(bs, beam_size).fill_(0)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)

        # current position
        cur_len = 1

        # cache compute states
        # cache = {'slen': 0}

        # done sentences
        done = [False for _ in range(bs)]

        """ rule for alpha write """
        forbidden_ids = []
        for _ in range(bs):
            forbidden = []
            for index in range(beam_size):
                forbidden.append(self.vocab.tgt["<unk>"] + index * n_words)
            forbidden_ids.append(forbidden)

        while cur_len < max_decoding_time_step:
            """
            tensor, attn_dist, p_gens, coverages = self.forward(
                'fwd',
                x=generated[:cur_len],
                lengths=src_len.new(bs * beam_size).fill_(cur_len),
                positions=positions[:cur_len],
                langs=langs[:cur_len],
                causal=True,
                src_enc=src_enc,
                src_len=src_len,
                cache=cache
            )
            """
            y_tm1 = torch.tensor([hyp.item() if hyp.item() < len(self.vocab.tgt) else self.vocab.tgt["<unk>"] for hyp in generated[cur_len-1, :]], dtype=torch.long, device=self.device)
            y_tm1_embed = self.tgt_embed(y_tm1)  # 所有的，假设的sent的最后一个词，的list
             
            if self.input_feed:  # attention计算
                x = torch.cat([y_tm1_embed, att_tm1], dim=-1)
            else:
                x = y_tm1_embed

            (h_t, cell_t), ctx_t, alpha_t, att_t = self.step(x, h_tm1, src_encodings,src_encodings_att_linear, src_sent_masks, None)
            
            if self.copy:
                p_gen = F.softmax(self.readout(att_t), dim=-1)  # 生成概率 att_t_src -> att_t_skv
                p_gen.data.masked_fill_(tgt_vocab_masks.byte(), float(0))
                if max_oov_num > 0:
                    oov_zeros = torch.zeros(p_gen.size(0), max_oov_num, device=self.device)
                    p_gen = torch.cat([p_gen, oov_zeros], dim=-1)

                p = torch.cat([h_t, ctx_t, y_tm1_embed], dim=-1)
                g = torch.sigmoid(self.p_linear(p))
                #                 epsilon_mask = torch.ones_like(g) * 1e-6
                #                 g = g + epsilon_mask
                #                 g[g > 1] = 1
                """ 加偏移避免loss nan """
                w = torch.clone(g)
                w[w == 0] = 1e-6
                w[w == 1] = 1 - 1e-6
                g = w
                
                sents_var_complete_src_expanded = src_complete_sents_var.permute(1, 0).expand(alpha_t.size(0), -1)
                p_t = (g * p_gen).scatter_add(1, sents_var_complete_src_expanded, (1 - g) * alpha_t)

                log_p_t = torch.log(p_t)

            else:
                log_p_t = F.log_softmax(self.readout(att_t), dim=-1)


            scores = log_p_t

            assert scores.size() == (bs * beam_size, n_words)

            # select next words with scores
            _scores = scores + beam_scores[:, None].expand_as(scores)  # (bs * beam_size, n_words)
            _scores = _scores.view(bs, beam_size * n_words)            # (bs, beam_size * n_words)
            """ forbid """
            assert (len(forbidden_ids) == bs)
            
            for sent_id in range(bs):
                if not done[sent_id]:
                    _scores[sent_id][forbidden_ids[sent_id]] = -float('inf')
            
            forbidden_ids = []
            next_scores, next_words = torch.topk(_scores, 2 * beam_size, dim=1, largest=True, sorted=True)
            assert next_scores.size() == next_words.size() == (bs, 2 * beam_size)

            # next batch beam content
            # list of (bs * beam_size) tuple(next hypothesis score, next word, current position in the batch)
            next_batch_beam = []

            # for each sentence
            for sent_id in range(bs):
                # if we are done with this sentence
                done[sent_id] = done[sent_id] or generated_hyps[sent_id].is_done(next_scores[sent_id].max().item())
                if done[sent_id]:
                    next_batch_beam.extend([(0, self.vocab.tgt['<pad>'], 0)] * beam_size)  # pad the batch
                    forbidden_ids.append([])
                    continue

                # next sentence beam content
                next_sent_beam = []
                forbidden_id_list = [self.vocab.tgt["<unk>"] + i * n_words for i in range(beam_size)]  # UNK
                # next words for this sentence
                for idx, value in zip(next_words[sent_id], next_scores[sent_id]):

                    # get beam and word IDs
                    beam_id = idx // n_words
                    word_id = idx % n_words

                    """ for alpha_w """
                     
                    base_id = len(next_sent_beam) * n_words
                    if self.copy and word_id >= len(self.vocab.tgt):
                        word_value = word_oovs[sent_id][word_id.item() - len(self.vocab.tgt)]
                    else:
                        word_value = self.vocab.tgt.id2word[word_id.item()]
                    new_hyp_sent = [self.vocab.tgt.id2word[prefix_word_id.item()] if prefix_word_id.item() < len(self.vocab.tgt) else word_oovs[sent_id][prefix_word_id.item() - len(self.vocab.tgt)] for prefix_word_id in generated[1: cur_len, sent_id * beam_size + beam_id: sent_id * beam_size + beam_id + 1]] + [word_value]  # 跳过<s>
                    
                    # end of sentence, or next word
                    if word_id == eos_id or cur_len + 1 == max_decoding_time_step:
                        generated_hyps[sent_id].add(generated[:cur_len, sent_id * beam_size + beam_id].clone(), value.item())
                    else:
                        next_sent_beam.append((value, word_id, sent_id * beam_size + beam_id))
                        
                        if self.copy:
                            forbid_duplicate(self, base_id, new_hyp_sent, forbidden_id_list, word_oovs=word_oovs[sent_id], first_uni2id=first_uni2id, first_bi2id=first_bi2id, tri2id=tri2id)
                        else:
                            forbid_duplicate(self, base_id, new_hyp_sent, forbidden_id_list, word_oovs=None, first_uni2id=first_uni2id, first_bi2id=first_bi2id, tri2id=tri2id)

                    # the beam for next step is full
                    if len(next_sent_beam) == beam_size:
                        break
                
                forbidden_ids.append(forbidden_id_list)

                # update next beam content
                assert len(next_sent_beam) == 0 if cur_len + 1 == max_decoding_time_step else beam_size
                if len(next_sent_beam) == 0:
                    next_sent_beam = [(0, self.vocab.tgt.word2id["<pad>"], 0)] * beam_size  # pad the batch
                next_batch_beam.extend(next_sent_beam)
                assert len(next_batch_beam) == beam_size * (sent_id + 1)

            # sanity check / prepare next batch
            assert len(next_batch_beam) == bs * beam_size
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_words = generated.new([x[1] for x in next_batch_beam])
            beam_idx = src_sents_len.new([x[2] for x in next_batch_beam])
            h_tm1 = (h_t[beam_idx], cell_t[beam_idx]) 
            att_tm1 = att_t[beam_idx]


            # re-order batch and internal states
            generated = generated[:, beam_idx]
            generated[cur_len] = beam_words

            # update current length
            cur_len = cur_len + 1

            # stop when we are done with each sentence
            if all(done):
                break

        # select the best hypotheses
        tgt_len = src_sents_len.new(bs)
        best = []

        for i, hypotheses in enumerate(generated_hyps):
            best_hyp = max(hypotheses.hyp, key=lambda x: x[0])[1]
            tgt_len[i] = len(best_hyp) + 1  # +1 for the <EOS> symbol
            best.append(best_hyp)

        # generate target batch
        decoded = src_sents_len.new(tgt_len.max().item(), bs).fill_(self.vocab.tgt["<pad>"])
        for i, hypo in enumerate(best):
            decoded[:tgt_len[i] - 1, i] = hypo
            decoded[tgt_len[i] - 1, i] = eos_id

        # sanity check
        assert (decoded == eos_id).sum() == 1 * bs

        completed_hypotheses = []
        for j in range(decoded.size(1)):
            sent = decoded[:, j]
            target = " ".join([self.vocab.tgt.id2word[sent[k].item()] if sent[k].item() < len(self.vocab.tgt) else word_oovs[j][sent[k].item() - len(self.vocab.tgt)] for k in range(1, len(sent))])
            completed_hypotheses.append(target)
        
        completed_hypotheses_unsort = []
        for index in idx_unsort_dev:
            completed_hypotheses_unsort.append(completed_hypotheses[index].split('</s>')[0].strip())
        return completed_hypotheses_unsort

    @staticmethod
    def reload(argss):
        model_path = argss['--load-model']
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = NMT(vocab=params['vocab'],
                    copy=str_to_bool(argss['--copy']),
                    coverage=str_to_bool(argss['--coverage']),
                    **args)
        model.load_state_dict(params['state_dict'])

        return model

    @staticmethod
    # def load(model_path):
    def load(argss):
        model_path = argss['MODEL_PATH']
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = NMT(vocab=params['vocab'],
                    copy=str_to_bool(argss['--copy']),
                    coverage=str_to_bool(argss['--coverage']),
                    **args)
        model.load_state_dict(params['state_dict'])

        return model

    def save(self, path):
        print('save model parameters to [%s]' % path, file=sys.stderr)

        params = {
            'args': dict(embed_size=self.embed_size, hidden_size=self.hidden_size, dropout_rate=self.dropout_rate,
                         input_feed=self.input_feed, label_smoothing=self.label_smoothing),
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }

        torch.save(params, path)

class BeamHypotheses(object):

    def __init__(self, n_hyp, max_len, length_penalty, early_stopping):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_len = max_len - 1  # ignoring <BOS>
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.n_hyp = n_hyp
        self.hyp = []
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.hyp)

    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list.
        """
        #score = sum_logprobs / len(hyp) ** self.length_penalty
        score = sum_logprobs
        if len(self) < self.n_hyp or score > self.worst_score:
            self.hyp.append((score, hyp))
            if len(self) > self.n_hyp:
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.hyp)])
                del self.hyp[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs):
        """
        If there are enough hypotheses and that none of the hypotheses being generated
        can become better than the worst one in the heap, then we are done with this sentence.
        """
        if len(self) < self.n_hyp:
            return False
        elif self.early_stopping:
            return True
        else:
            return self.worst_score >= best_sum_logprobs / self.max_len ** self.length_penalty


def str_to_bool(str):
    return True if str.lower() == 'true' or str == '1' else False


def evaluate_ppl(model, dev_data, coverage, batch_size=32, tgt_vocab_masks=None):
    """
    Evaluate perplexity on dev sentences

    Args:
        dev_data: a list of dev sentences
        batch_size: batch size

    Returns:
        ppl: the perplexity on dev sentences
    """

    # was_training = model.training
    # model.eval()

    cum_loss = 0.
    cum_tgt_words = 0.

    # you may want to wrap the following code using a context manager provided
    # by the NN library to signal the backend to not to keep gradient information
    # e.g., `torch.no_grad()`

    with torch.no_grad():
        dev_data_unfold = []
        for src, tgts in dev_data:
            for tgt in tgts:
                dev_data_unfold.append([src, tgt])

        for src_sents, tgt_sents in batch_iter(dev_data_unfold, batch_size):
            if coverage:
                loss, coverage_loss = model(src_sents, tgt_sents, tgt_vocab_masks)
                loss = -loss.sum()
            else:
                loss = -model(src_sents, tgt_sents, tgt_vocab_masks).sum()

            cum_loss += loss.item()
            tgt_word_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`
            cum_tgt_words += tgt_word_num_to_predict

        ppl = np.exp(cum_loss / cum_tgt_words)

    # if was_training:
    #    model.train()

    return ppl


def compute_corpus_level_bleu_score(references, hypotheses):
    """
    Given decoding results and reference sentences, compute corpus-level BLEU score

    Args:
        references: a list of gold-standard reference target sentences
        hypotheses: a list of hypotheses, one for each reference

    Returns:
        bleu_score: corpus-level BLEU score
    """

    if references[0][0] == '<s>':
        references = [ref[1:-1] for ref in references]

    bleu_score = corpus_bleu([[ref] for ref in references],
                             [hyp.value for hyp in hypotheses])

    return bleu_score


def train(args):
    train_data_src = read_corpus(args['--train-src'], source='src')
    train_data_tgt = read_corpus(args['--train-tgt'], source='tgt')

    """ 读取停用词 词表文件 """
    stopwords_set = read_stopwords(args['--stop-words-file'])

    dev_data_src, dev_data_tgts, _, _ = read_corpus_json(args['--dev'])

    train_data = list(zip(train_data_src, train_data_tgt))
    dev_data = list(zip(dev_data_src, dev_data_tgts))

    train_batch_size = int(args['--batch-size'])
    clip_grad = float(args['--clip-grad'])
    valid_niter = int(args['--valid-niter'])
    log_every = int(args['--log-every'])
    model_save_path = args['--save-to']
    coverage = str_to_bool(args['--coverage'])

    if args['--load-model'] != 'False':
        model_load_path = args['--load-model']
        print('loading model from %s ...' % model_load_path, file=sys.stderr)
        model = NMT.reload(args)

        hist_valid_ppl_scores = []
        hist_valid_ppl_scores.append(float(args['--best-ppl']))
        hist_valid_rouge_scores = []
        hist_valid_rouge_scores.append(float(args['--best-r2']))
        model.train()
        device = torch.device("cuda:0" if args['--cuda'] else "cpu")
        print('use device: %s' % device, file=sys.stderr)
        model = model.to(device)
        print('restore parameters of the optimizers', file=sys.stderr)
        num_trial = int(args['--num-trial'])

        optimizer = torch.optim.Adam(model.parameters(), lr=float(args['--lr']))
        optimizer.load_state_dict(torch.load(model_load_path + '.optim'))

        lr_cur = float(args['--lr']) * float(args['--lr-decay']) ** num_trial
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_cur

        valid_rouge_score_cur = max(hist_valid_rouge_scores)
        valid_ppl_score_cur = max(hist_valid_ppl_scores)
        print('current lr is: %f' % lr_cur, file=sys.stderr)
        # print('current best rouge2 f1 score is: %f' % valid_score_cur, file=sys.stderr)
        model_save_path_cur = model_load_path

    else:
        vocab = Vocab.load(args['--vocab'])
        model = NMT(embed_size=int(args['--embed-size']),
                    hidden_size=int(args['--hidden-size']),
                    dropout_rate=float(args['--dropout']),
                    input_feed=args['--input-feed'],
                    label_smoothing=float(args['--label-smoothing']),
                    vocab=vocab,
                    copy=str_to_bool(args['--copy']),
                    coverage=coverage)

        num_trial = 0
        hist_valid_rouge_scores = []
        hist_valid_ppl_scores = []
        uniform_init = float(args['--uniform-init'])
        if np.abs(uniform_init) > 0.:
            print('uniformly initialize parameters [-%f, +%f]' % (uniform_init, uniform_init), file=sys.stderr)
            for p in model.parameters():
                p.data.uniform_(-uniform_init, uniform_init)
        model.train()
        device = torch.device("cuda:0" if args['--cuda'] else "cpu")
        print('use device: %s' % device, file=sys.stderr)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(args['--lr']))

    print("make only-copy mask")
    exclusive_words_set = read_stopwords(args['--exclusive-words-file'])
    tgt_vocab_masks = torch.zeros(1, len(model.vocab.tgt), dtype=torch.float)
    for idx, word in model.vocab.tgt.id2word.items():
        if word in exclusive_words_set:
            tgt_vocab_masks[0, idx] = 1
    
    tgt_vocab_masks = tgt_vocab_masks.to(model.device)

    train_iter = patience = cum_loss = report_loss = cum_tgt_words = report_tgt_words = 0
    cum_examples = report_examples = epoch = valid_num = 0
    train_time = begin_time = time.time()
    print('begin Maximum Likelihood training')

    while True:
        epoch += 1

        for src_sents, tgt_sents in batch_iter(train_data, batch_size=train_batch_size, shuffle=True):
            train_iter += 1

            optimizer.zero_grad()

            batch_size = len(src_sents)

            # (batch_size)
            if coverage:
                example_losses, coverage_losses = model(src_sents, tgt_sents, tgt_vocab_masks)
                example_losses = -example_losses
            else:
                example_losses = -model(src_sents, tgt_sents, tgt_vocab_masks)
            batch_loss = example_losses.sum()
            loss = batch_loss / batch_size

            if coverage:
                coverage_loss = coverage_losses.sum() / batch_size
                loss += coverage_loss

            loss.backward()

            # clip gradient
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

            optimizer.step()

            batch_losses_val = batch_loss.item()
            report_loss += batch_losses_val
            cum_loss += batch_losses_val

            tgt_words_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`
            report_tgt_words += tgt_words_num_to_predict
            cum_tgt_words += tgt_words_num_to_predict
            report_examples += batch_size
            cum_examples += batch_size

            if train_iter % log_every == 0:
                print('epoch %d, iter %d, avg. loss %.2f, avg. ppl %.2f ' \
                      'cum. examples %d, speed %.2f words/sec, time elapsed %.2f sec' % (epoch, train_iter,
                                                                                         report_loss / report_examples,
                                                                                         math.exp(
                                                                                             report_loss / report_tgt_words),
                                                                                         cum_examples,
                                                                                         report_tgt_words / (
                                                                                                 time.time() - train_time),
                                                                                         time.time() - begin_time),
                      file=sys.stderr)

                train_time = time.time()
                report_loss = report_tgt_words = report_examples = 0.

            # perform validation
            if train_iter % valid_niter == 0:
                print('epoch %d, iter %d, cum. loss %.2f, cum. ppl %.2f cum. examples %d' % (epoch, train_iter,
                                                                                             cum_loss / cum_examples,
                                                                                             np.exp(
                                                                                                 cum_loss / cum_tgt_words),
                                                                                             cum_examples),
                      file=sys.stderr)

                cum_loss = cum_examples = cum_tgt_words = 0.
                valid_num += 1

                print('begin validation ...', file=sys.stderr)
                model.eval()

                # compute dev. ppl and bleu
                # dev batch size can be a bit larger
                dev_ppl = evaluate_ppl(model, dev_data, coverage, batch_size=10, tgt_vocab_masks=tgt_vocab_masks)
                # valid_metric = -dev_ppl

                # ROUGE evaluation begin

                dev_hyps = beam_search_dev(model, dev_data_src, beam_size=10,
                                           max_decoding_time_step=int(args['--max-decoding-time-step']),
                                           stopwords_set=stopwords_set, args=args, tgt_vocab_masks=tgt_vocab_masks)
                #dev_hyps = [hyps[0].value for hyps in dev_hyps]
                dev_rouge2 = get_rouge2f([tgts for src, tgts in dev_data], dev_hyps)

                print('validation: iter %d, dev. ppl %f, dev. ROUGE2 %f' % (train_iter,
                                                                            dev_ppl, dev_rouge2), file=sys.stderr)

                model.train()
                is_better = len(hist_valid_ppl_scores) == 0 or dev_ppl < min(hist_valid_ppl_scores) or dev_rouge2 > max(
                    hist_valid_rouge_scores)
                hist_valid_ppl_scores.append(dev_ppl)
                hist_valid_rouge_scores.append(dev_rouge2)

                if is_better:
                    best_model_iter = train_iter
                    patience = 0
                    model_save_path_cur = model_save_path + ".iter" + str(best_model_iter)
                    print('save currently the best model to [%s]' % model_save_path_cur, file=sys.stderr)
                    model.save(model_save_path_cur)

                    # also save the optimizers' state
                    torch.save(optimizer.state_dict(), model_save_path_cur + '.optim')
                elif patience < int(args['--patience']):
                    patience += 1
                    print('hit patience %d' % patience, file=sys.stderr)

                    if patience == int(args['--patience']):
                        num_trial += 1
                        print('hit #%d trial' % num_trial, file=sys.stderr)
                        if num_trial == int(args['--max-num-trial']):
                            print('early stop!', file=sys.stderr)
                            print('the best model is from iteration [%d]' % best_model_iter,
                                  file=sys.stderr)
                        print('hit #%d trial' % num_trial, file=sys.stderr)
                        if num_trial == int(args['--max-num-trial']):
                            print('early stop!', file=sys.stderr)
                            print('the best model is from iteration [%d]' % best_model_iter,
                                  file=sys.stderr)
                            exit(0)

                        # decay lr, and restore from previously best checkpoint
                        lr = optimizer.param_groups[0]['lr'] * float(args['--lr-decay'])
                        print('load previously best model %s and decay learning rate to %f' % (model_save_path_cur, lr),
                              file=sys.stderr)

                        # load model
                        params = torch.load(model_save_path_cur, map_location=lambda storage, loc: storage)
                        model.load_state_dict(params['state_dict'])
                        model = model.to(device)

                        print('restore parameters of the optimizers', file=sys.stderr)
                        optimizer.load_state_dict(torch.load(model_save_path_cur + '.optim'))

                        # set new lr
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr

                        # reset patience
                        patience = 0

                if epoch == int(args['--max-epoch']):
                    print('reached maximum number of epochs!', file=sys.stderr)
                    exit(0)


def get_rouge2f(references, hypotheses):
    references = [[[char for char in "".join(ref)] for ref in refs] for refs in references]
#    hypotheses = [hyp.split() for hyp in hypotheses]
    hypotheses = [[char for char in "".join(hyp.split())] for hyp in hypotheses]
    # compute ROUGE-2 F1-SCORE
    rouge2f_score = rouge_2_corpus_multiple_target(references, hypotheses)
    return rouge2f_score


def beam_search_dev(model, test_data_src, beam_size, max_decoding_time_step, stopwords_set, args, tgt_vocab_masks):
    was_training = model.training
    model.eval()
    dev_batch_size = int(args['--dev-batch-size'])
    first_uni2id = {}
    first_bi2id = {}
    tri2id = {}
    for word, idx in model.vocab.tgt.word2id.items():
        first_uni = word[0]
        if first_uni not in first_uni2id:
            first_uni2id[first_uni] = []
        first_uni2id[first_uni].append(idx)

        if len(word) > 1:
            first_bi = word[:2]
            if first_bi not in first_bi2id:
                first_bi2id[first_bi] = []
            first_bi2id[first_bi].append(idx)

        if len(word) > 2:
            for start_idx in range(len(word) - 2):
                tri = word[start_idx: start_idx + 3]
                if tri not in tri2id:
                    tri2id[tri] = []
                tri2id[tri].append(idx)
    for uni, idx in first_uni2id.items():
        first_uni2id[uni] = list(set(idx))
    for bi, idx in first_bi2id.items():
        first_bi2id[bi] = list(set(idx))
    for tri, idx in tri2id.items():
        tri2id[tri] = list(set(idx))

    hypotheses = []
    begin_time = time.time()
    with torch.no_grad():
        for src_sent, idx_unsort in batch_iter_dev(test_data_src, batch_size=dev_batch_size, shuffle=False):
            example_hyps = model.beam_search(idx_unsort, src_sent, first_uni2id, first_bi2id, tri2id,
                                             beam_size=beam_size, max_decoding_time_step=max_decoding_time_step,
                                             stopwords_set=stopwords_set, args=args, tgt_vocab_masks=tgt_vocab_masks)

            hypotheses += example_hyps

    elapsed = time.time() - begin_time
    print('decoded %d examples, took %d s' % (len(test_data_src), elapsed), file=sys.stderr)

    if was_training:
        model.train(was_training)

    return hypotheses


def decode(args):
    """
    performs decoding on a test set, and save the best-scoring decoding results.
    If the target gold-standard sentences are given, the function also computes
    corpus-level BLEU score.
    """

    """ read stop words """
    stopwords_set = read_stopwords(args['--stop-words-file'])

    print("load test source sentences from %s" % args['TEST_FILE'], file=sys.stderr)
    test_data_src = read_corpus(args['TEST_FILE'], source='src')

    print("load model from %s " % args['MODEL_PATH'], file=sys.stderr)
    model = NMT.load(args)

    if args['--cuda']:
        model = model.to(torch.device("cuda:0"))

    """ read only-copy words """
    exclusive_words_set = read_stopwords(args['--exclusive-words-file'])
    
    tgt_vocab_masks = torch.zeros(1, len(model.vocab.tgt), dtype=torch.float)
    for idx, word in model.vocab.tgt.id2word.items():
        if word in exclusive_words_set:
            tgt_vocab_masks[0, idx] = 1
    tgt_vocab_masks = tgt_vocab_masks.to(model.device)

    hypotheses = beam_search_dev(model,
                                 test_data_src,
                                 beam_size=int(args['--beam-size']),
                                 max_decoding_time_step=int(args['--max-decoding-time-step']),
                                 stopwords_set=stopwords_set,
                                 args=args,
                                 tgt_vocab_masks=tgt_vocab_masks)

    with open(args['OUTPUT_FILE'], 'w') as f:
        for hyps in hypotheses:
            '''
            result_line = ""
            for hyp in hyps:
                result_line += " ".join(hyp.value) + '\t'
            f.write(result_line.strip().encode('utf-8') + '\n')
            '''
            #top_hyp = hyps[0]
            #hyp_sent = ' '.join(top_hyp.value)
            f.write(hyps + '\n')


def main():
    args = docopt(__doc__)
    print('Current Time: ' + str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))

    print(args, file=sys.stderr)
    # seed the random number generators
    seed = int(args['--seed'])
    torch.manual_seed(seed)
    if args['--cuda']:
        torch.cuda.manual_seed(seed)
    np.random.seed(seed * 13 // 7)

    if args['--mode'] == 'train':
        train(args)
    elif args['--mode'] == 'decode':
        decode(args)
    else:
        raise RuntimeError('invalid run mode')


if __name__ == '__main__':
    main()
