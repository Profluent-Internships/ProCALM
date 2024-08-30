import math
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchmetrics
import tqdm
from torchmetrics.utilities.distributed import gather_all_tensors

from ..model.model import CausalLMOutputWithPast

_skip_metric_updates_key = "skip_metric_updates"

class PerTokenLoss(torchmetrics.Metric, ABC):
    def __init__(self, pad_token: int = 0):
        super().__init__()
        self.pad_token = pad_token
        self.add_state("sum_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_tokens", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def compute(self):
        return self.sum_loss / self.num_tokens if self.num_tokens > 0 else torch.tensor(0.0)

class ARLMPerplexity(PerTokenLoss):
    def compute(self):
        return torch.exp(self.sum_loss / self.num_tokens) if self.num_tokens > 0 else torch.tensor(0.0)
    
    def update(self, batch: dict, outputs: CausalLMOutputWithPast):
        if not batch.get(_skip_metric_updates_key, False):
            n = (batch["labels"] != self.pad_token).sum()
            self.num_tokens += n
            self.sum_loss += outputs.loss * n

class PerTokenAccuracy(PerTokenLoss):
    def update(self, batch: dict, outputs: CausalLMOutputWithPast):
        if not batch.get(_skip_metric_updates_key, False):
            logits = outputs.logits.argmax(dim=-1)[..., :-1]
            labels = batch["labels"][..., 1:]
            mask = labels != self.pad_token

            correct = (logits == labels).float() * mask.float()
            self.sum_loss += correct.sum()
            self.num_tokens += mask.sum()