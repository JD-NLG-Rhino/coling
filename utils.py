# coding=utf-8
import math
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import json
import re
import os
import sys

def forbid_duplicate(model, base_id, new_hyp_sent, forbidden_id_list, word_oovs, first_uni2id, first_bi2id, tri2id):
    """
    forbid for duplicated bi-char, tri-char
    """

    if len(new_hyp_sent) > 1:   # forbid duplicated bi-word [AB ... AB]
        forbidden_ngrams = []
        for wid in range(len(new_hyp_sent) - 1):
            forbidden_ngrams.append([new_hyp_sent[wid], new_hyp_sent[wid + 1]])
        for forbidden_ngram in forbidden_ngrams:
            if forbidden_ngram[0] == new_hyp_sent[-1]:
                if forbidden_ngram[1] in model.vocab.tgt.word2id:
                    forbidden_id_list.append(model.vocab.tgt.word2id[forbidden_ngram[1]] + base_id)
                else:
                    forbidden_id_list.append(word_oovs.index(forbidden_ngram[1]) + len(model.vocab.tgt) + base_id)

    if len("".join(new_hyp_sent)) > 2:  # forbid duplicated tri-char
        forbidden_tri_chars = []

        for cid in range(len("".join(new_hyp_sent)) - 2):
            forbidden_tri = "".join(new_hyp_sent)[cid: cid + 3]
            forbidden_tri_chars.append(forbidden_tri)
            if forbidden_tri in tri2id:
                forbidden_id_list += [cur_id + base_id for cur_id in tri2id[forbidden_tri]]
            if word_oovs is not None:
                for word_oov in word_oovs:
                    if forbidden_tri in word_oov:
                        forbidden_id_list.append(word_oovs.index(word_oov) + len(model.vocab.tgt) + base_id)

        for forbidden_tri_char in forbidden_tri_chars:
            if forbidden_tri_char[0] == "".join(new_hyp_sent[0:])[-1]:
                forbidden_bi = forbidden_tri_char[1] + forbidden_tri_char[2]
                if forbidden_bi in first_bi2id:
                    forbidden_id_list += [cur_id + base_id for cur_id in first_bi2id[forbidden_bi]]
                if word_oovs is not None:
                    for word_oov in word_oovs:
                        if word_oov and forbidden_bi == word_oov[:2]:
                            forbidden_id_list.append(word_oovs.index(word_oov) + len(model.vocab.tgt) + base_id)

        for forbidden_tri_char in forbidden_tri_chars:
            if forbidden_tri_char[0] + forbidden_tri_char[1] == "".join(new_hyp_sent[0:])[-2] + "".join(new_hyp_sent[0:])[-1]:
                forbidden_uni = forbidden_tri_char[2]
                if forbidden_uni in first_uni2id:
                    forbidden_id_list += [cur_id + base_id for cur_id in first_uni2id[forbidden_uni]]
                if word_oovs is not None:
                    for word_oov in word_oovs:
                        if word_oov and forbidden_uni == word_oov[0]:
                            forbidden_id_list.append(word_oovs.index(word_oov) + len(model.vocab.tgt) + base_id)

def read_kb_table(file_path):
    """
    read attribute (k-v pair list)
    """

    input_lines = open(file_path).readlines()
    key_list = []
    value_list = []
    
    for line in input_lines:
        segs = line.strip().split('\t')
        assert (len(segs) & 1) == 0
        key_sent = []
        value_sent = []
        for i in range(0, len(segs), 2):
            key_words = [word for word in segs[i].strip().split()]
            value_words = [word for word in segs[i + 1].strip().split()]
            assert len(key_words) > 0
            assert len(value_words) > 0

            key_sent.append(key_words)
            value_sent.append(value_words)

        key_list.append(key_sent)
        value_list.append(value_sent)

    assert len(key_list) > 0
    assert len(value_list) > 0

    return key_list, value_list


def read_stopwords(file_path):
    """
    read stop words
    """

    input_lines = open(file_path).readlines()
    stopwords_set = set()

    for line in input_lines:
        stopwords_set.add(line.strip())

    return stopwords_set


def replace_digit(word):
    """
    replace int 、float ==> 'DD'
    """
    word = re.sub(r'\d+\.\d+', 'DD', word)
    word = re.sub(r'\d+', 'DD', word)
    return word


def input_transpose(sents, pad_token):
    max_len = max(len(s) for s in sents)
    batch_size = len(sents)

    sents_t = []
    for i in range(max_len):
        sents_t.append([sents[k][i] if len(sents[k]) > i else pad_token for k in range(batch_size)])

    return sents_t


def read_corpus(file_path, source):
    data = []
    for line in open(file_path).readlines():
        sent = [word for word in line.strip().split()]
        # only append <s> and </s> to the target sentence
        if source == 'tgt':
            sent = ['<s>'] + sent + ['</s>']
        data.append(sent)

    return data

def read_corpus_json(file_path):
    res = json.load(open(file_path))
    data_src = []
    data_tgt = []

    data_table_key = []   # key-value list
    data_table_value = []

    for sku, txt in res.items():
        src = txt["source"]
        tgts = txt["targets"]
        table = txt["table"]

        src = [word for word in src.strip().split()]
        tgts = [[word for word in tgt.strip().split()] for tgt in tgts]
        data_src.append(src)
        data_tgt.append(tgts)

        table_segs = table.strip().split('\t')
        assert (len(table_segs) & 1) == 0
        key_sent = []
        value_sent = []
        for i in range(0, len(table_segs), 2):
            key_words = [word for word in table_segs[i].strip().split()]
            value_words = [word for word in table_segs[i + 1].strip().split()]
            assert len(key_words) > 0
            assert len(value_words) > 0
            key_sent.append(key_words)
            value_sent.append(value_words)

        data_table_key.append(key_sent)
        data_table_value.append(value_sent)

        assert len(data_table_key) > 0
        assert len(data_table_value) > 0
    return data_src, data_tgt, data_table_key, data_table_value


def batch_iter(data, batch_size, shuffle=False):
    batch_num = int(math.ceil(len(data)*1.0 / batch_size))
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)
    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]

        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        src_sents = [e[0] for e in examples]
        tgt_sents = [e[1] for e in examples]

        if len(examples[0]) > 2:
            table_keys = [e[2] for e in examples]
            table_values = [e[3] for e in examples]

            yield src_sents, tgt_sents, table_keys, table_values
        else:
            yield src_sents, tgt_sents


def batch_iter_dev(data, batch_size, shuffle=False):
    batch_num = int(math.ceil(len(data)*1.0 / batch_size))
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)
    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]

        if not isinstance(examples[0], tuple):
            results = sorted(enumerate(examples), key=lambda e: len(e[1]), reverse=True)
        else:
            results = sorted(enumerate(examples), key=lambda e: len(e[1][0]), reverse=True)

        idx_sort = [index for index, value in results]
        idx_unsort = np.argsort(idx_sort)
        
        examples = [value for index, value in results]

        src_sents = [e[0] for e in examples]
       
        if not isinstance(examples[0], tuple):
            src_sents = [e for e in examples]
            yield src_sents, idx_unsort

        elif len(examples[0]) > 2:
            src_sents = [e[0] for e in examples]
            table_keys = [e[1] for e in examples]
            table_values = [e[2] for e in examples]

            yield src_sents, table_keys, table_values, idx_unsort
        elif len(examples[0]) > 1:
            src_sents = [e[0] for e in examples]
            tgt_sents = [e[1] for e in examples]

            yield src_sents, tgt_sents, idx_unsort


class LabelSmoothingLoss(nn.Module):
    """
    label smoothing

    Code adapted from OpenNMT-py
    """
    def __init__(self, label_smoothing, tgt_vocab_size, padding_idx=0):
        assert 0.0 < label_smoothing <= 1.0
        self.padding_idx = padding_idx
        super(LabelSmoothingLoss, self).__init__()

        smoothing_value = label_smoothing / (tgt_vocab_size - 2)  # -1 for pad, -1 for gold-standard word
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.padding_idx] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))

        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target, max_oov_num=0, copy=False):
        """
        output (FloatTensor): batch_size x tgt_vocab_size
        target (LongTensor): batch_size
        """
        # (batch_size, tgt_vocab_size)
        true_dist = self.one_hot.repeat(target.size(0), 1)

        if copy:
            oov_zeros = torch.zeros(true_dist.size(0), max_oov_num, device=torch.device("cuda:0"))
            true_dist = torch.cat([true_dist, oov_zeros], dim=-1)

        # fill in gold-standard word position with confidence value
        true_dist.scatter_(1, target.unsqueeze(-1), self.confidence)

        # fill padded entries with zeros
        true_dist.masked_fill_((target == self.padding_idx).unsqueeze(-1), 0.)


        loss = -F.kl_div(output, true_dist, reduction='none').sum(-1)

        return loss

