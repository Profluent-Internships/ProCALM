import logging
import re
import os
import json
import tarfile
import urllib.request
from pathlib import Path
import torch

logger = logging.getLogger(__name__)

def get_parameters(param_dir, model_name='progen2-base'):
    url = "https://storage.googleapis.com/sfr-progen-research/checkpoints/{}.tar.gz"

    model_dir = os.path.join(param_dir, model_name)
    model_url = url.format(model_name)
    
    if os.path.exists(model_dir + ".success"):
        print(f"Skipping {model_name} because it already exists.")
    else:
        print(f"Downloading {model_name} from {model_url} to {model_dir}")
        os.makedirs(model_dir, exist_ok=True)
        urllib.request.urlretrieve(model_url, model_dir + ".tar.gz")  # nosec
        with tarfile.open(model_dir + ".tar.gz") as f:
            f.extractall(model_dir)  # nosec
        Path(model_dir + ".success").touch()
    
    return model_dir

def get_config(model_config, data_config) -> dict:
    condition2encoding = {}

    pretrained_model_dir = get_parameters('data/pretrained_models', model_name=model_config["pretrained_model"])
    kwargs = json.load(open(os.path.join(pretrained_model_dir, 'config.json')))
    kwargs["gradient_checkpointing"] = model_config.get("gradient_checkpointing", True)

    kwargs["full_finetuning"] = model_config.get("full_finetuning", False)
    kwargs["pretrained_model_name"] = model_config["pretrained_model"]
    kwargs["pretrained_model_dir"] = pretrained_model_dir
    kwargs["adapter_weight_init"] = model_config.get("adapter_weight_init", 1e-5)
    kwargs["adapter_nlayers"] = model_config.get("adapter_nlayers", 2)
    kwargs["adapter_dropout"] = model_config.get("adapter_dropout", 0.1)
    kwargs["adapter_c_s"] = model_config.get("adapter_c_s", 128)
    kwargs["adapter_c_hidden"] = model_config.get("adapter_c_hidden", 16)

    kwargs["adapter_projection_nlayers"] = model_config.get("adapter_projection_nlayers", None)
    kwargs["adapter_summation"] = model_config.get("adapter_summation", False)
    if kwargs["adapter_summation"]:
        #override c_hidden with c_s
        print("Summation in the adapter is enabled. Overriding c_hidden with c_s.")
        kwargs["adapter_c_hidden"] = kwargs["adapter_c_s"]
    kwargs["adapter_shared_projection"]= model_config.get("adapter_shared_projection", True)
    
    #this only applies to joint conditioning with multiple conditions
    kwargs["conditions_shared_adapter"] = model_config.get("conditions_shared_adapter", True)
    
    kwargs["encoding_files"] = {}
    kwargs["encoding_dimensions"] = {}

    for condition_name in ["ec", "tax", "stability"]:
        encoding_file = data_config.get(f"{condition_name}_encoding", None)
        if encoding_file is not None:
            kwargs["encoding_files"][condition_name] = encoding_file
            condition2encoding[condition_name] = torch.load(encoding_file)
            kwargs["encoding_dimensions"][condition_name] = list(condition2encoding[condition_name].values())[0].shape[0]
            print(f"Training with {condition_name} conditioning.")
        else:
            print(f"Skipping {condition_name} conditioning.")
    
    return kwargs, condition2encoding