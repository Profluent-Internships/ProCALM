import copy
import logging
import math
import os
import tempfile
from typing import Any, Dict
import json
from ruamel.yaml import YAML

import composer
import composer.optim
import numpy as np
import streaming
import torch
from composer.algorithms import GradientClipping
from composer.loggers import FileLogger, WandBLogger
from composer.utils import dist, reproducibility
from composer import Trainer

from ..data.tokenizer import get_tokenizer
from ..defaults import get_config
from .checkpoint import HuggingFaceCheckpointer
from .data import StreamingDataLoader, StreamingDataset, TokenDataSpec
from .model import Model
from .model import ProGenConditionalConfig
from .evaluate import get_evaluators

logger = logging.getLogger(__name__)

def get_trainer(
    config_file: str,
    force_new_run: bool = False,
    disable_logging: bool = False,
    debug: bool = False,
):  
    with open(config_file) as f:
        config = YAML(typ="safe").load(f)
        if "run_name" not in config:
            config["run_name"] = os.path.basename(config_file).split(".")[0]

    # Initialize the model and optimizer
    config = copy.deepcopy(config)
    data_config = copy.deepcopy(config["data"])
    train_config = copy.deepcopy(config["train"])
    model_config = copy.deepcopy(config["model"])

    # Initialize the model
    with reproducibility.seed_context(train_config.get("seed", 42)):
        #load pretrained model and update the config
        kwargs, condition2encoding = get_config(model_config=model_config, data_config=data_config)
        print(kwargs)
            
        progenconditional_config = ProGenConditionalConfig(**kwargs)
        composer_model = Model(config=progenconditional_config) 

        #set weights to be trained
        if model_config.get("full_finetuning", False):
            train_parameters = list(composer_model.model.parameters())
            for p in train_parameters:
                p.requires_grad = True
        else:
            adapter_parameters = list(composer_model.model.adapter_layers.parameters())
            projection_parameters = list(composer_model.model.projection_mlps.parameters())

            train_parameters = adapter_parameters + projection_parameters
            for p in train_parameters:
                p.requires_grad = True

    composer_model = composer_model.to(device=dist.get_local_rank()) #addded this for ddp

    num_parameters = sum(p.numel() for p in composer_model.model.parameters())
    print("Total parameters: {:e}".format(num_parameters)) #print in scientific notation
    
    num_trainable_parameters = sum(p.numel() for p in train_parameters)
    print("Trainable parameters: {:e}".format(num_trainable_parameters))

    # Set up the optimizer and LR scheduler
    optimizer = composer.optim.DecoupledAdamW(
        train_parameters,
        lr=train_config.get("lr", 5e-4),
        weight_decay=train_config.get("weight_decay", 5e-6),
    )

    max_duration = str(train_config.get("max_duration", "1e9")) + "tok" #cannot be set as epochs because the dataloader has no length
    warmup_steps = int(train_config.get("warmup_steps", 1000))
    tokens_per_batch = int(data_config.get("train_tokens_per_batch", 1 << 20))
    warmup_tokens =  str(int(warmup_steps * tokens_per_batch)) + "tok"

    scheduler = composer.optim.CosineAnnealingWithWarmupScheduler(
        t_warmup=warmup_tokens,
        t_max=max_duration,
        #to reproduce original model results, use below
        #t_warmup=str(warmup_steps) + "ba",
        #t_max = str(100000) + "ba", #filler value as the approximate maximum number of batches
        alpha_f=train_config.get("total_lr_decay_factor", 0.1),
    )
   
    algorithms = []
    clipping_threshold = train_config.get("gradient_clipping_threshold", 1.0)
    if clipping_threshold is not None:
        gc = GradientClipping(
            clipping_type="norm",
            clipping_threshold=float(clipping_threshold),
        )
        algorithms.append(gc)
    
    experiment_name = config.get("run_name", "test")
    save_folder = os.path.join(config["save_folder"], experiment_name)
    save_dir = save_folder
    file_logger = FileLogger(filename=os.path.join(save_dir, "logs", "logs-rank{rank}.txt"), flush_interval=30)
    os.makedirs(os.path.join(save_dir), exist_ok=True)
    #copy config file to the save_folder
    with open(os.path.join(save_dir, os.path.basename(config_file)), "w") as f:
        YAML().dump(config, f)

    # Set up the training data
    num_workers = data_config.get("num_workers", math.ceil(os.cpu_count() / dist.get_local_world_size()) - 1)
    train_data_kwargs = dict(
        rng=np.random.default_rng(data_config.get("seed", 9176)),
        tokenizer=get_tokenizer(model=composer_model),
        prefetch_factor=None if debug else 1,  # Pre-fetches based on batch_size, not device_batch_tokens
        num_workers=0 if debug else max(1, min(4, num_workers)),
    )

    for source in data_config["train_sources"]:
        if "remote" in source and "local" not in source:
            tmpdir = [tempfile.mkdtemp()]
            dist.broadcast_object_list(tmpdir, src=0)
            os.makedirs(tmpdir[0], exist_ok=True)  # To handle multi-node cases
            source["local"] = tmpdir[0]

    #use streaming dataset
    train_dataset = StreamingDataset(
    streams=[streaming.Stream(**s) for s in data_config["train_sources"]],
    batch_size=data_config.get("prefetch_batch_size", 2000),
    shuffle=data_config.get("shuffle", True), #might want to change this for validation set
    shuffle_seed=data_config.get("seed", 9176),
    download_timeout=600)

    device_batch_tokens = math.ceil(tokens_per_batch / dist.get_world_size())

    train_loader = StreamingDataLoader(
        train_dataset,
        pin_memory=True,
        drop_last=True, #dropping the last batch is fine for training
        persistent_workers=False,
        batch_size = train_dataset.batch_size,
        condition2encoding=condition2encoding,
        tokens_per_batch=device_batch_tokens,
        **train_data_kwargs,
    )

    #save a huggingface compatible model at each checkpoint
    save_interval = config.get("save_interval", "1000ba")
    callbacks = [
        HuggingFaceCheckpointer(
            save_folder=save_folder,
            save_interval= save_interval,
            precision="bfloat16" if torch.cuda.is_bf16_supported() else "float16",
            # huggingface_folder_name="",  #saves to f"{save_folder}/huggingface"
        )]

    evaluators = list(get_evaluators(config, condition2encoding, debug=debug))

    wandb_logger = WandBLogger(
    project="ProCALM",  # Replace with your W&B project name
    #entity="your_wandb_entity",   # Optional: your W&B entity name
    log_artifacts=True            # Set to True if you want to save artifacts
)

    # Set up the actual trainer
    half = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    trainer = Trainer(
        model=composer_model,
        run_name=experiment_name,
        train_dataloader= TokenDataSpec(train_loader, batch_tokens=device_batch_tokens),
        eval_dataloader=evaluators,
        accumulate_train_batch_on_tokens=True,
        device_train_microbatch_size=data_config.get("device_train_microbatch_tokens", device_batch_tokens), #overide the automatic microbatching, each batch is one microbatch. Could alternatively set this to be smaller than the total batch size
        spin_dataloaders=False,
        precision=f"amp_{half}",
        seed=train_config.get("seed", None),
        loggers=[wandb_logger, file_logger] if not disable_logging else None, #added the wandb logger
        callbacks=callbacks,
        # Optimization
        max_duration=max_duration,
        optimizers=optimizer,
        schedulers=scheduler,
        algorithms=algorithms,
        step_schedulers_every_batch=True,
        # Save/load
        autoresume=True,
        load_path=train_config.get("composer_checkpoint", None),  # To initialize continual pre-training
        load_weights_only=train_config.get("composer_load_weights_only", True),  # Only used for continual pre-training
        save_folder=save_folder,
        save_filename="ep{epoch}-ba{batch}-rank{rank}.pt",
        save_latest_filename="latest.pt",
        save_interval=save_interval,
        progress_bar=dist.get_local_rank() == 0,
    )

    return trainer
