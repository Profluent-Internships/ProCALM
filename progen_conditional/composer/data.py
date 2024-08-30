import functools
import math
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import pandas as pd
import composer
import numpy as np
import streaming
import torch
from composer.utils import dist
from tokenizers import Tokenizer
from torch.utils.data import Dataset, DataLoader

from ..data import (
    PAD_TOKEN_ID,
    get_tokenizer,
    prepare_batch,
)
from .metrics import _skip_metric_updates_key


def trim_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Trims all tensors in a batch to the maximum observed sequence length."""
    new_batch = dict()
    for k, v in batch.items():
        if not isinstance(v, (torch.Tensor, np.ndarray)) or v.ndim < 2:
            new_batch[k] = v
    maxlen = (batch["input_ids"] != PAD_TOKEN_ID).sum(dim=-1).max().item()
    mask = batch["attention_mask"]  # (bsz, qlen, kvlen) or (bsz, qlen)
    if mask.ndim == 3:
        new_batch["attention_mask"] = mask[:, :maxlen, :maxlen]
    new_batch.update({k: v[:, :maxlen] for k, v in batch.items() if k not in new_batch})
    return new_batch


def get_num_samples_in_batch(batch: Dict[str, torch.Tensor]) -> int:
    return batch["input_ids"].shape[0]


def get_num_tokens_in_batch(batch: Dict[str, torch.Tensor]) -> int:
    return (batch["input_ids"] != PAD_TOKEN_ID).sum().item()


def _subselect(batch: dict, idxs: Union[int, List[int]], ignore: bool):
    sub_batch = {k: v[idxs] if isinstance(v, (np.ndarray, torch.Tensor)) else v for k, v in batch.items()}
    if ignore:
        sub_batch[_skip_metric_updates_key] = True
    return trim_batch(sub_batch)
    
class BatchSplitter:
    def __init__(self, batch_size: int):
        self.batch_size = batch_size

    def __call__(self, batch: Dict[str, torch.Tensor], microbatch_size: int) -> List[Dict[str, torch.Tensor]]:
        """Split a big batch into ``self.batch_size / microbatch_size`` sub-batches."""
        n = len(batch["input_ids"])
        n_microbatches = math.ceil(self.batch_size / microbatch_size)

        # Case 1: we don't have enough sequences to split into the desired number of microbatches.
        # Each microbatch has 1 sequence. There are repeats, but we ignore the repeats for gradient updates.
        if n < n_microbatches:
            return [_subselect(batch, [i % n], ignore=i >= n) for i in range(n)]

        # Case 2: we have enough sequences. Split them as evenly as possible.
        k = 0
        sub_batches = []
        for i in range(n_microbatches):
            n_in_mb = n // n_microbatches + int(i % n_microbatches < n % n_microbatches)
            sub_batches.append(_subselect(batch, list(range(k, k + n_in_mb)), ignore=False))
            k += n_in_mb

        return sub_batches

class TokenDataSpec(composer.DataSpec):
    def __init__(self, dataloader: DataLoader, batch_tokens, num_samples=None, num_tokens=None):
        super().__init__(
            dataloader,
            num_samples=num_samples,
            num_tokens=num_tokens,
            split_batch=BatchSplitter(batch_tokens),
            get_num_samples_in_batch=get_num_samples_in_batch,
            get_num_tokens_in_batch=get_num_tokens_in_batch,
        )

class StreamingDataset(streaming.StreamingDataset):
    def __init__(
        self,
        *args,
        sequence_key: str = "sequence",
        ec_condition_key: str= "ec",
        tax_condition_key: str = "tax",
        stability_condition_key: str = "stability",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.sequence_key = sequence_key
        self.ec_condition_key = ec_condition_key
        self.tax_condition_key = tax_condition_key
        self.stability_condition_key = stability_condition_key

    def __getitem__(self, at) -> Union[str, Tuple[str, Dict[str, Any]]]:
        ret = super().__getitem__(at)
        sequence = ret.get(self.sequence_key, None)
        ec = ret.get(self.ec_condition_key, None)
        tax = ret.get(self.tax_condition_key, None)
        stability = ret.get(self.stability_condition_key, None)
        return (sequence, ec, tax, stability)

class StreamingDataLoader(streaming.StreamingDataLoader):
    def __init__(
        self,
        *args,
        condition2encoding: Dict[str, Dict[str, torch.tensor]] = None,
        tokens_per_batch: int = None,
        tokenizer: Tokenizer = None,
        rng: np.random.Generator = None,
        task: str = "train",
        device: str = "cpu", #not specified originally
        **kwargs,
    ):
        task2prepare = {"train": prepare_batch} #train and eval share the same batch preparation
        assert task in task2prepare, f"task must be one of {list(task2prepare.keys())}, but got {task}"
        self.condition2encoding = condition2encoding

        self.tokens_per_batch = tokens_per_batch
        self.tokenizer = tokenizer
        self.rng = rng if not isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        collate_fn = functools.partial(
            task2prepare[task],
            condition2encoding=self.condition2encoding,
            tokenizer=self.tokenizer,
            tokens_per_batch=self.tokens_per_batch,
            rng=self.rng,
            device=device,
            #device=dist.get_local_rank(), #added this for ddp but this causes issues
        )
        super().__init__(*args, collate_fn=collate_fn, **kwargs)

    def state_dict(self) -> Optional[Dict[str, Any]]:
        if isinstance(self.dataset, streaming.StreamingDataset):
            # When resuming training, we will skip data points rather than repeating them
            local_rank = dist.get_local_rank()
            n = torch.tensor([self.num_samples_yielded], device=local_rank) * dist.get_world_size()
            dist.all_reduce(n, reduce_operation="MAX")
            return self.dataset.state_dict(n.item(), False)
        return None

    def __len__(self):
        return None if self.tokens_per_batch is not None else super().__len__()

    def __iter__(self) -> Iterator[Any]:
        self.num_samples_yielded = 0
        done_reduce = "MAX" if self.drop_last else "MIN"
        epoch_done = torch.tensor([0], device=dist.get_local_rank())

        #this part ocasionaly causes issues if reloading from a certain checkpoint
        for batches in DataLoader.__iter__(self):
            for batch in batches:
                #print(batch['input_ids'].shape)
                dist.all_reduce(epoch_done, reduce_operation=done_reduce)
                if epoch_done.item():
                    break
                self.num_samples_yielded += get_num_samples_in_batch(batch)
                yield batch
            if epoch_done.item():
                break

        # If we don't drop any examples, make sure the excess batches skip metric updates
        batch[_skip_metric_updates_key] = True
        while not epoch_done.item():
            epoch_done |= 1
            dist.all_reduce(epoch_done, reduce_operation=done_reduce)
            if epoch_done.item():
                break
            yield batch
