import copy
import functools
import math
import re
import string
from collections import namedtuple
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from hashlib import md5
from typing import Any, Dict, List, Set, Tuple, Union

import numpy as np
import scipy
import torch
from tokenizers import Tokenizer, Encoding
from transformers.utils import logging

logger = logging.get_logger(__name__)

def construct_padded_tensors(max_length: int, encs: List[Encoding]):
    attention_mask = np.zeros((len(encs), max_length), dtype=bool)
    input_ids = np.zeros((len(encs), max_length), dtype=int)
    
    for i, enc in enumerate(encs):
        assert len(enc.attention_mask) == len(enc.ids)

        attention_mask[i, : len(enc.attention_mask)] = enc.attention_mask
        input_ids[i, : len(enc.ids)] = enc.ids

    return input_ids, attention_mask

def prepare_batch(
    batch: list[tuple],
    condition2encoding: Dict[str, Dict[str, torch.tensor]],
    tokenizer: Tokenizer,
    rng: Union[int, np.random.Generator],
    tokens_per_batch: int = None,
    device: Union[torch.device, str] = "cpu", #not specified usually
) -> List[Dict]:

    #this could be faster but dataloading does not seem to be the bottleneck
    seqs = []
    ECs = []
    taxes = []
    stablities = []
    texts = []
    for seq, ec, tax, stablity, text in batch:
        seqs.append(seq)
        ECs.append(ec)
        taxes.append(tax)
        stablities.append(stablity)
        texts.append(text)

    processed_seqs: List[str] = []
    reverse_booleans = rng.random(len(seqs)) < 0.5

    for seq, reverse in zip(seqs, reverse_booleans):
        #randomly reverse seq based on rval probability
        seq = '1' + seq + '2'
        if reverse:
            seq = seq[::-1]
        processed_seqs.append(seq)
    
    seq_encodings = tokenizer.encode_batch(processed_seqs)
    if 'ec' in condition2encoding:
        ec2encoding = condition2encoding['ec']
        EC_encodings = [ec2encoding[ec] for ec in ECs]
    if 'tax' in condition2encoding:
        tax2encoding = condition2encoding['tax']
        tax_encodings = [tax2encoding[tax_id] for tax_id in taxes]
    if 'text' in condition2encoding:
        text2encoding = condition2encoding['text']
        text_encodings = [text2encoding[text] for text in texts]

    # If we want a specific number of tokens per sub-batch, sort the sequences by length, and
    # combine them into sub-batches that have the appropriate number of total tokens (including padding)
    if tokens_per_batch is None:
        batch_idxs = [list(range(len(seq_encodings)))]
    else:
        lengths = [len(enc.attention_mask) for enc in seq_encodings]
        batch_idxs, sub_batch_idxs, max_length = [], [], 0
        for i in np.argsort(lengths):
            max_length = max(lengths[i], max_length)
            if len(sub_batch_idxs) > 0 and (len(sub_batch_idxs) + 1) * max_length > tokens_per_batch:
                batch_idxs.append(sub_batch_idxs)
                sub_batch_idxs, max_length = [], 0
            sub_batch_idxs.append(i)
        if len(sub_batch_idxs) > 0:
            batch_idxs.append(sub_batch_idxs)

    # Construct the sub-batches
    batches = []
    for k in rng.permutation(len(batch_idxs)):
        idxs = batch_idxs[k]
        encs = [seq_encodings[i] for i in idxs]
        selected_seqs = [seqs[i] for i in idxs]

        max_length = len(encs[-1].ids)

        #Construct padded attention masks, position ids and sequence ids
        input_ids, attention_mask = construct_padded_tensors(max_length, encs)

        if 'ec' in condition2encoding:
            EC_encodings_selected = [EC_encodings[i] for i in idxs]
            #stack the encodings
            EC_encodings_selected = torch.stack(EC_encodings_selected)
            #repeat EC_encodings to have second dimension as max_length
            EC_encodings_selected = EC_encodings_selected.repeat(max_length, 1, 1)
            #flip the first two dimensions
            EC_encodings_selected = torch.transpose(EC_encodings_selected, 0, 1)
        if 'tax' in condition2encoding:
            tax_encodings_selected = [tax_encodings[i] for i in idxs]
            tax_encodings_selected = torch.stack(tax_encodings_selected)
            tax_encodings_selected = tax_encodings_selected.repeat(max_length, 1, 1)
            tax_encodings_selected = torch.transpose(tax_encodings_selected, 0, 1)
        if 'text' in condition2encoding:
            text_encodings_selected = [text_encodings[i] for i in idxs]
            text_encodings_selected = torch.stack(text_encodings_selected)
            text_encodings_selected = text_encodings_selected.repeat(max_length, 1, 1)
            text_encodings_selected = torch.transpose(text_encodings_selected, 0, 1)

        batch = dict(
            input_ids=torch.tensor(input_ids, device=device),
            labels=torch.tensor(input_ids, device=device),
            attention_mask=torch.tensor(attention_mask, device=device, dtype=torch.bool),
            sequences=selected_seqs, #doesn't seem to increase memory
        )

        adapter_input = {}
        if 'ec' in condition2encoding:
            adapter_input['ec'] = EC_encodings_selected.to(device)
        if 'tax' in condition2encoding:
            adapter_input['tax'] = tax_encodings_selected.to(device)
        if 'text' in condition2encoding:
            adapter_input['text'] = text_encodings_selected.to(device)
        batch['adapter_input'] = adapter_input

        batches.append(batch)
    return batches