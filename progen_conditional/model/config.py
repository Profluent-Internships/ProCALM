# Copyright 2021 The EleutherAI and HuggingFace Teams. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modified configuration implementation based on https://github.com/huggingface/transformers/blob/main/src/transformers/models/gptj/configuration_gptj.py

from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

class ProGenConfig(PretrainedConfig):
    model_type = "progen"

    def __init__(
        self,
        vocab_size=50400,
        n_positions=2048,
        n_ctx=2048,
        n_embd=4096,
        n_layer=28, #but it's actually 27
        n_head=16,
        rotary_dim=64,
        n_inner=None,
        activation_function="gelu_new",
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        scale_attn_weights=True,
        gradient_checkpointing=True,
        use_cache=False,
        bos_token_id=50256,
        eos_token_id=50256,
        #Attention implementation & rotary positional embeddings
        #this is added in from the foundation model since there's better data processing here
        inter_context_sequence_attention=True,
        inter_sequence_attention=True,
        msa_style_attention=True,
        unified_active_context_pos_ids=True,
        max_num_sequences=512,
        #mixture of experts
        output_router_logits=False,
        router_aux_loss_coef=0.05,
        pad_token_id=0,
        **kwargs
    ):
        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.n_ctx = n_ctx
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner
        self.rotary_dim = rotary_dim
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.gradient_checkpointing = gradient_checkpointing
        self.scale_attn_weights = scale_attn_weights
        self.use_cache = use_cache
        self.msa_style_attention = msa_style_attention
        self.unified_active_context_pos_ids = unified_active_context_pos_ids
        self.inter_context_sequence_attention = inter_context_sequence_attention
        self.inter_sequence_attention = inter_sequence_attention
        self.max_num_sequences = max_num_sequences
        if router_aux_loss_coef > 0:
            output_router_logits = True
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.pad_token_id = pad_token_id

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

    @property
    def max_position_embeddings(self):
        return self.n_positions

    @property
    def hidden_size(self):
        return self.n_embd

    @property
    def num_attention_heads(self):
        return self.n_head

    @property
    def num_hidden_layers(self):
        return self.n_layer

class ProGenConditionalConfig(ProGenConfig):
    model_type = "progen-conditional"

    def __init__(
        self,
        pretrained_model_name="progen2-base",
        pretrained_model_dir="data/pretrained_models/progen2-base",
        full_fineturning = False, #whether to finetune the full model or just the adapter (parameter efficient)
        encoding_files: Dict[str, str] = None, #dictionary mapping each type of condition to the file containing the respective encodings
        encoding_dimensions: Dict[str, int] = None, #dictionary mapping each type of condition to the dimension of the respective encodings
        adapter_c_s=128, #adapter condition dimension after linear layer
        adapter_c_hidden=16, #low-rank dimenion of LM emebeddings (to be concaeated with adapter_c_s)
        adapter_dropout = 0.1,
        adapter_weight_init = 1e-5, #weight to initialize up-projection of hidden embedding update
        adapter_nlayers = 2, #number of layers in the adapter MLP
        adapter_projection_nlayers: int = None, #number of layers in the adapter projection (conditioning encoder). None corresponds to linear projection
        adapter_summation: bool = False, #sum conditon and hidden state instead of concatenating
        adapter_shared_projection: bool = True, #use a shared projection layer (conditioning encoder) for each of the adapter layers
        conditions_shared_adapter: bool = True, #use a shared adapter for all conditions. if False, a separate adapter is used for each condition
        **kwargs
    ):
        super().__init__(**kwargs)
        self.full_fineturning = full_fineturning
        self.pretrained_model_name = pretrained_model_name
        self.pretrained_model_dir = pretrained_model_dir
        self.encoding_files = encoding_files
        self.encoding_dimensions = encoding_dimensions
        self.adapter_c_s = adapter_c_s
        self.adapter_c_hidden = adapter_c_hidden
        self.adapter_dropout = adapter_dropout
        self.adapter_weight_init = adapter_weight_init
        self.adapter_nlayers = adapter_nlayers
        self.adapter_projection_nlayers = adapter_projection_nlayers
        self.adapter_summation = adapter_summation
        self.adapter_shared_projection = adapter_shared_projection
        self.conditions_shared_adapter = conditions_shared_adapter
