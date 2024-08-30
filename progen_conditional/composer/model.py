from typing import Any, Dict, List, Optional

import torchmetrics
from composer import ComposerModel as _ComposerModel

from ..model.config import ProGenConditionalConfig
from ..model.model import ProgenConditional, CausalLMOutputWithPast
from .metrics import (
    ARLMPerplexity,
    _skip_metric_updates_key,
)

class ComposerModel(_ComposerModel):
    def __init__(self, *args, **kwargs):
        super().__init__()


class Model(ComposerModel):
    def __init__(
        self,
        config: ProGenConditionalConfig,
    ):
        super().__init__(config)
        pad = config.pad_token_id
        self.model = ProgenConditional(config=config)
        self.config = self.model.config
        self.train_ar_perplexity = ARLMPerplexity(pad_token=pad)

    def forward(self, *args, **kwargs) -> CausalLMOutputWithPast:
        if len(args) == 1 and isinstance(args[0], dict):
            batch = args[0]
        elif "batch" in kwargs:
            batch = kwargs["batch"]
        else:
            batch = None
        if batch is None:
            return self.model.forward(*args, **kwargs)

        include = {"input_ids", "attention_mask", "labels", "adapter_input"}
        processed_batch = {k: v for k, v in batch.items() if k in include}
        #print(processed_batch)
        return self.model.forward(**processed_batch)

    def loss(self, outputs: CausalLMOutputWithPast, batch: Dict[str, Any]):
        return outputs.loss * float(not batch.get(_skip_metric_updates_key, False))

    def get_metrics(self, is_train=True) -> Dict[str, torchmetrics.Metric]:
        loss_dict = {
            "loss/ar_lm_perplexity": self.train_ar_perplexity,
        }
        return loss_dict

    def update_metric(
        self,
        batch: Dict[str, Any],
        outputs: CausalLMOutputWithPast,
        metric: torchmetrics.Metric,
    ):
        metric.update(batch=batch, outputs=outputs)
