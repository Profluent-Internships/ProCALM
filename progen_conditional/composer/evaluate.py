import copy
import itertools
import logging
import math
import os
import tempfile
from typing import Any, Dict, Iterator, List
import streaming
import torch
import composer
import numpy as np
from composer.utils import dist

from .data import StreamingDataLoader, StreamingDataset, TokenDataSpec, get_tokenizer

logger = logging.getLogger(__name__)


def get_evaluator(
    source: dict,
    batch_size: int,
    batch_tokens: str,
    rng: np.random.RandomState,
    evaluator_name: str,
    condition2encoding: Dict[str, Dict[str, torch.tensor]],
    metric_names: List[str],
    eval_interval: str = "1dur",
    subset_num_batches: int = None,
    **kwargs,
) -> composer.Evaluator:

    streams = [streaming.Stream(**s) for s in source]
    dataset = StreamingDataset(
        streams=streams,
        batch_size=batch_size,
        # shuffle=False,
        download_timeout=60,
    )
    dataloader = StreamingDataLoader(
        dataset,
        pin_memory=False,
        drop_last=False,
        batch_size=dataset.batch_size,
        tokens_per_batch=batch_tokens,
        condition2encoding=condition2encoding,
        rng=rng,
        **kwargs,
    )

    if subset_num_batches is None:
        subset_num_batches = len(
            dataset)  # overestimate, just to get completion
    return composer.Evaluator(
        dataloader=TokenDataSpec(dataloader, batch_tokens=batch_tokens),
        eval_interval=eval_interval,
        metric_names=metric_names,
        subset_num_batches=subset_num_batches,
        label=evaluator_name,
        device_eval_microbatch_size=batch_tokens,
    )

def get_test_evaluators(
    config: list,
    batch_size: int,
    batch_tokens: int,
    condition2encoding: Dict[str, Dict[str, torch.tensor]],
    rng: np.random.RandomState,
    eval_interval: str = "1dur",
    subset_num_batches: int = None,
    **kwargs,
) -> Iterator[composer.Evaluator]:

    # for split_config in config:
    for source_dict in config:
        split_name = list(source_dict.keys())[0]
        source = source_dict[split_name]

        if "remote" in source and "local" not in source:
            tmpdir = [tempfile.mkdtemp()]
            dist.broadcast_object_list(tmpdir, src=0)
            os.makedirs(tmpdir[0], exist_ok=True)  # To handle multi-node cases
            source["local"] = tmpdir[0]

        yield get_evaluator(
            source=source,
            batch_size=batch_size,
            batch_tokens=batch_tokens,
            rng=rng,
            condition2encoding=condition2encoding,
            evaluator_name=f"{split_name}",
            metric_names=["loss/ar_lm_perplexity"],
            eval_interval=eval_interval,
            subset_num_batches=subset_num_batches,
            **kwargs,
        )

def get_evaluators(
    config: Dict[str, Any],
    condition2encoding: Dict[str, Dict[str, torch.tensor]],
    debug: bool = False,
    skip_evals: List[str] = None,
) -> Iterator[composer.Evaluator]:
    # Get the model, tokenizer, and basic configs
    config = copy.deepcopy(config)
    data_config = copy.deepcopy(config["data"])
    train_config = copy.deepcopy(config["train"])
    double_max_duration = str(
        2 * int(train_config["max_duration"])) + "tok"

    def get_rng():
        return np.random.default_rng(data_config.get("seed", 9176))

    num_workers = data_config.get("num_workers", math.ceil(os.cpu_count() / dist.get_local_world_size()) - 1)
    # Set up common keyword arguments for all the evaluation datasets
    common_kwargs = dict(
        tokenizer=get_tokenizer(),
        num_workers=0 if debug else max(1, min(4, num_workers)),
        prefetch_factor=None if debug else 1,
        persistent_workers=False,
        batch_size=data_config.get("prefetch_batch_size", 2000),
        batch_tokens=data_config.get("eval_tokens_per_batch", 1 << 16),
        subset_num_batches=data_config.get("eval_subset_num_batches", None),
        eval_interval=data_config.get("eval_interval", double_max_duration),
    )

    skip_evals = skip_evals or []

    if "val_sources" in data_config:
        yield from get_test_evaluators(
            data_config["val_sources"],
            condition2encoding=condition2encoding,
            rng=get_rng(),
            **common_kwargs,
        )
