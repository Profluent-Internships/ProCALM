# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import contextlib
import copy
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from composer.core import Callback, Event, State, Time, TimeUnit
from composer.core.state import fsdp_state_dict_type_context
from composer.loggers import Logger
from composer.utils import (
    dist,
    format_name_with_dist_and_time,
    maybe_create_remote_uploader_downloader_from_uri,
    parse_uri,
)
from composer.utils.misc import create_interval_scheduler
from packaging import version
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP #currently does not support fsdp but left it in
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from ..utils import init_empty_weights

log = logging.getLogger(__name__)

__all__ = ["HuggingFaceCheckpointer"]


class HuggingFaceCheckpointer(Callback):
    """Save a huggingface formatted checkpoint during training.

    Args:
        save_folder (str): Top level folder to save checkpoints to (can be a
            URI). It is likely that this would be the same as your save_folder.
        save_interval: Union[str, int, Time]: The interval describing how often
            checkpoints should be saved. If an integer, it will be assumed to be
            in :attr:`.TimeUnit.EPOCH`. Otherwise, the unit must be either
            :attr:`.TimeUnit.EPOCH`, :attr:`.TimeUnit.BATCH`,
            :attr:`.TimeUnit.TOKEN`, or :attr:`.TimeUnit.SAMPLE`.
        huggingface_folder_name (str): Folder to save each checkpoint under (can
            be a format string). Default is ``ep{epoch}``.
        precision: The precision to save the model in. Default is ``float32``.
            Options are ``bfloat16``, ``float16``, or ``float32``.
        overwrite (bool): Whether to overwrite previous checkpoints.
    """

    def __init__(
        self,
        save_folder: str,
        save_interval: Union[str, int, Time],
        huggingface_folder_name: str = "ba{batch}", #str = "latest" #"ba{batch}"ep{epoch} #this will save each checkpoint, instead just override it to save space
        precision: str = "float32",
        overwrite: bool = True,
    ):
        _, _, self.save_dir_format_str = parse_uri(save_folder)
        self.overwrite = overwrite
        self.precision = precision
        self.dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[precision]

        self.huggingface_folder_name_fstr = os.path.join(
            "huggingface",
            huggingface_folder_name,
        )

        self.save_interval: Time = Time.from_input(
            save_interval,
            TimeUnit.EPOCH,
        )
        self.check_interval = create_interval_scheduler(
            self.save_interval,
            include_end_of_training=True,
        )
        self.remote_ud = maybe_create_remote_uploader_downloader_from_uri(
            save_folder,
            loggers=[],
        )
        if self.remote_ud is not None:
            self.remote_ud._num_concurrent_uploads = 4

        self.last_checkpoint_batch: Optional[Time] = None

        #supress torch warnings for now?
        #getting one about meta parameters and passing "assign=True"

    def run_event(self, event: Event, state: State, logger: Logger) -> None:
        # The interval scheduler handles only returning True for the appropriate events
        if (
            state.get_elapsed_duration() is not None
            and self.check_interval(
                state,
                event,
            )
            and self.last_checkpoint_batch != state.timestamp.batch
        ):
            self._save_checkpoint(state, logger)
        elif event == Event.INIT:
            if not isinstance(state.model.model, PreTrainedModel):
                raise ValueError(
                    "`HuggingFaceCheckpointer` is only compatible with `HuggingFaceModel`s. "
                    + f"Got {type(state.model.model)} instead.",
                )
            if self.remote_ud is not None:
                self.remote_ud.init(state, logger)
                state.callbacks.append(self.remote_ud)

    def transform_model_and_tokenizer(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        """Transform the model and tokenizer before saving.

        This allows a subclass to modify the model and tokenizer before saving. The base class implementation will
        make no modifications.

        Args:
            model (PreTrainedModel): The model to be transformed.
            tokenizer (PreTrainedTokenizerBase): The tokenizer to be transformed.

        Returns:
            Tuple[PreTrainedModel, PreTrainedTokenizerBase]: The transformed model and tokenizer.
        """
        return model, tokenizer

    def _save_checkpoint(self, state: State, logger: Logger):
        del logger  # unused

        self.last_checkpoint_batch = state.timestamp.batch

        log.info("Saving HuggingFace formatted checkpoint")

        # from transformers.models.auto.configuration_auto import CONFIG_MAPPING
        # CONFIG_MAPPING._extra_content['mpt'] = MPTConfig
        # MPTConfig.register_for_auto_class()
        # MPTForCausalLM.register_for_auto_class('AutoModelForCausalLM')

        save_dir = format_name_with_dist_and_time(
            str(
                Path(self.save_dir_format_str) / self.huggingface_folder_name_fstr,
            ),
            state.run_name,
            state.timestamp,
        )

        # Use a temporary directory if save_dir is remote.
        use_temp_dir = self.remote_ud is not None
        temp_save_dir = tempfile.mkdtemp() if use_temp_dir else save_dir

        log.info("Gathering state dict")

        #the use of module is different when using ddp and when not
        original_tokenizer = None
        if state.is_model_ddp:
            composer_model = state.model
            original_model: PreTrainedModel = state.model.module.model
            state_dict_model = state.model.module.model
            # original_tokenizer = state.model.model.module.tokenizer

        #not sure if this works
        elif isinstance(state.model.model, FSDP):
            composer_model = state.model
            original_model: PreTrainedModel = state.model.module.model
            state_dict_model = state.model.module.model
            # original_tokenizer = state.model.model.tokenizer
        else:
            composer_model = state.model
            original_model: PreTrainedModel = state.model.model #put module before model?
            state_dict_model = state.model.model #put module before model?
            # original_tokenizer = state.model.model.tokenizer

        if version.parse(torch.__version__) > version.parse("2.2.9"):
            from torch.distributed._tensor import DTensor
            from torch.distributed.checkpoint.state_dict import (
                StateDictOptions,
                get_model_state_dict,
            )

            cpu_offload = True

            # Add a dtensor->cpu tensor hook to avoid CUDA OOM
            def dtensor_to_tensor_hook(
                module: nn.Module,
                state_dict: Dict[str, Any],
                prefix: str,
                *args: Any,
            ) -> Dict[str, Any]:
                dtensor_fqns = []
                for fqn in state_dict.keys():
                    tensor = state_dict[fqn]
                    if isinstance(tensor, DTensor):
                        dtensor_fqns.append(fqn)
                        tensor = tensor.full_tensor()  # type: ignore
                        if dist.get_global_rank() == 0:
                            if cpu_offload:
                                tensor = tensor.cpu()
                            state_dict[fqn] = tensor
                if dist.get_global_rank() != 0:
                    for fqn in dtensor_fqns:
                        del state_dict[fqn]
                return state_dict

            hooks = []
            for _, module in state_dict_model.named_modules():
                if isinstance(module, FSDP):
                    hooks.append(
                        module._register_state_dict_hook(dtensor_to_tensor_hook),
                    )

            state_dict = get_model_state_dict(
                state_dict_model,
                options=StateDictOptions(
                    full_state_dict=True,
                    cpu_offload=cpu_offload,
                ),
            )
            for hook in hooks:
                hook.remove()
        else:
            state_dict_context = (
                fsdp_state_dict_type_context(
                    original_model,
                    state_dict_type="full",
                )
                if ((not state.is_model_ddp) and isinstance(state_dict_model, FSDP))
                else contextlib.nullcontext()
            )
            with state_dict_context:
                state_dict = state_dict_model.state_dict()

        # Convert the state dict to the requested precis
        for k, v in state_dict.items():
            if isinstance(v, torch.Tensor):
                state_dict[k] = v.to(dtype=self.dtype)

        new_model_instance = None  # Need this for pyright because variable could be unbound

        if dist.get_global_rank() == 0:
            log.info("Saving Hugging Face checkpoint in global rank 0")

            # Edit HF config before building 2nd model copy
            copied_config = copy.deepcopy(original_model.config)
            log.info("Creating new model instance")

            if hasattr(composer_model, "using_peft") and composer_model.using_peft:
                # We don't use meta here because the state dict does not contain the full
                # model, only the adapter weights.
                #this path may not work anymore
                active_adapter = original_model.active_adapter
                base_model = original_model.get_base_model()
                new_base_model_instance = type(base_model)(copied_config)

                new_model_instance = type(original_model)(
                    new_base_model_instance,
                    original_model.peft_config[active_adapter],
                )
                new_model_instance.to(dtype=self.dtype)
            else:
                # First create the model instance on meta device to avoid the
                # initialization cost.
                with init_empty_weights():
                    new_model_instance = type(original_model)(copied_config)
                    if hasattr(original_model, "generation_config") and original_model.generation_config is not None:
                        new_model_instance.generation_config.update(
                            **original_model.generation_config.to_dict(),
                        )

            # Then load the state dict in with "assign" so that the state dict
            # is loaded properly even though the model is initially on meta device.
            new_model_instance.load_state_dict(state_dict, assign=True)  #this is throwing a bunch of warnings
            del state_dict

            # Transform the model and tokenizer before saving
            new_model_instance, original_tokenizer = self.transform_model_and_tokenizer(
                new_model_instance,
                original_tokenizer,
            )

            log.info("Saving Hugging Face checkpoint to disk")
            new_model_instance.save_pretrained(temp_save_dir)
            if original_tokenizer is not None:
                assert isinstance(original_tokenizer, PreTrainedTokenizerBase)
                original_tokenizer.save_pretrained(temp_save_dir)

            if self.remote_ud is not None:
                for filename in os.listdir(temp_save_dir):
                    remote_file_name = os.path.join(save_dir, filename)
                    remote_file_uri = self.remote_ud.remote_backend.get_uri(
                        remote_file_name,
                    )
                    log.info(
                        f"Uploading HuggingFace formatted checkpoint to {remote_file_uri}",
                    )
                    self.remote_ud.upload_file(
                        state=state,
                        remote_file_name=remote_file_name,
                        file_path=Path(os.path.join(temp_save_dir, filename)),
                        overwrite=self.overwrite,
                    )

        dist.barrier()

        if dist.get_global_rank() == 0 and use_temp_dir:
            shutil.rmtree(temp_save_dir)
        dist.barrier()