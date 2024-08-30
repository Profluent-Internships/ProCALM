from progen_conditional.model import ProgenConditional
from progen_conditional.data.tokenizer import get_tokenizer
from progen_conditional.composer.data import StreamingDataset, StreamingDataLoader, TokenDataSpec
from composer.utils import dist
import tempfile
from tqdm import tqdm

import streaming
import torch
import numpy as np
import pandas as pd
from tokenizers import Tokenizer, Encoding
from typing import Any, Dict, List, Set, Tuple, Union
import os
import json

device = 'cuda'
os.chdir('../')

for model_name, checkpoint in zip(['ec-onehot-swissprot'],['ba21000']): 

    ckpt_file = f'results/{model_name}/huggingface/{checkpoint}'

    if os.path.exists(os.path.join(ckpt_file, "model.safetensors")):
        #if there is a local safetensors file to load
        model = ProgenConditional.from_pretrained(ckpt_file)
        tokenizer = get_tokenizer()
    else:
        #download the model from the huggingface model hub and cache it locally
        model = ProgenConditional.from_pretrained("jsunn-y/ProCALM", subfolder="{}/{}".format(model_name, checkpoint_name), cache_dir=ckpt_file)
        tokenizer = Tokenizer.from_pretrained("jsunn-y/ProCALM")

    model.to(device)
    model.eval()

    train_data_kwargs = dict(
        rng=np.random.default_rng(9176),
        tokenizer=tokenizer,
        prefetch_factor= 1,  # Pre-fetches based on batch_size, not device_batch_tokens
        num_workers=4
    )

    sources = {'val70': 'data/sharded_datasets/CARE_resampled50cluster_medium_withTax/val70',
            'val90': 'data/sharded_datasets/CARE_resampled50cluster_medium_withTax/val90',
            'test': 'data/sharded_datasets/CARE_resampled50cluster_medium_withTax/test',
            'train_common': 'data/sharded_datasets/CARE_resampled50cluster_medium_withTax/train_common',
            'train_rare': 'data/sharded_datasets/CARE_resampled50cluster_medium_withTax/train_rare'}

    condition2encoding = {}
    for condition_name, encoding_file in model.config.encoding_files.items():
        condition2encoding[condition_name] = torch.load(encoding_file)

    df = pd.DataFrame(columns=['sequence', 'perplexity', 'split'])

    tokens_per_batch = 24000 #fits on a 40GB A100
    tqdm_iter = tqdm(range(len(sources)), desc="Calculating perplexity")

    for split, source in sources.items():
        tmpdir = [tempfile.mkdtemp() if dist.get_global_rank() == 0 else None]
        dist.broadcast_object_list(tmpdir)
        tmpdir = tmpdir[0]
        os.makedirs(tmpdir, exist_ok=True) # To handle multi-node cases

        #use the streaming dataset if it has been built and the source is linked
        dataset = StreamingDataset(
        remote=source,
        local=tmpdir,
        batch_size=2000, #number of sequences to query
        shuffle=False, 
        download_timeout=600)

        loader = StreamingDataLoader(
            dataset,
            pin_memory=False,
            drop_last=False, #need to include this so the last partial batch is included
            persistent_workers=False,
            batch_size = dataset.batch_size,
            condition2encoding=condition2encoding,
            tokens_per_batch=tokens_per_batch,
            **train_data_kwargs,
        )

        all_sequences = []
        all_perplexities = []

        with torch.no_grad():
            for batch in loader:
                sequences = batch['sequences']
                include = {"input_ids", "attention_mask", "labels", "adapter_input"}
                processed_batch = {k: v for k, v in batch.items() if k in include}

                for key in {"input_ids", "attention_mask", "labels"}:
                    processed_batch[key] = processed_batch[key].to(device)

                for key in processed_batch["adapter_input"]:
                    processed_batch["adapter_input"][key] = processed_batch["adapter_input"][key].to(device)
                
                batch_size = processed_batch["input_ids"].shape[0]

                outputs = model.forward(**processed_batch)
                losses = outputs.all_losses
                all_sequences.extend(sequences)

                losses = losses.reshape(batch_size, -1).cpu().numpy()
                #take the mean along each row, but ignore zeros (these are the pad tokens)
                losses = list(np.exp(losses.sum(axis=1) / (losses != 0).sum(axis=1)))
                all_perplexities.extend(losses)
                
        update_df = pd.DataFrame({'sequence': all_sequences, 'perplexity': all_perplexities, 'split': split})
        df = pd.concat([df, update_df])
        df.to_csv(f'results/{model_name}/perplexity_{checkpoint}.csv', index=False)
        tqdm_iter.update(1)
#
