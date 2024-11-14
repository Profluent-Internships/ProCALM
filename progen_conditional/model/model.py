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

# Modified forward-pass implementation based on https://github.com/huggingface/transformers/blob/main/src/transformers/models/gptj/modeling_gptj.py
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import (
    BaseModelOutputWithPast as _BaseModelOutputWithPast,
)
from transformers.modeling_outputs import (
    CausalLMOutputWithPast as _CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map

from .adapter import ParallelAdapterLayer, ProjectionMLP
from .config import ProGenConfig, ProGenConditionalConfig
from ..utils import exists

logger = logging.get_logger(__name__)

@dataclass
class BaseModelOutputWithPast(_BaseModelOutputWithPast):
    inputs: Optional[Union[torch.LongTensor, torch.FloatTensor]] = None


@dataclass
class CausalLMOutputWithPast(_CausalLMOutputWithPast):
    all_losses: Optional[torch.FloatTensor] = None
    inputs: Optional[Union[torch.LongTensor, torch.FloatTensor]] = None

def fixed_pos_embedding(x, seq_dim=1, seq_len=None):
    dim = x.shape[-1]
    if seq_len is None:
        seq_len = x.shape[seq_dim]
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2) / dim))
    sinusoid_inp = (
        torch.einsum("i , j -> i j", torch.arange(seq_len), inv_freq).to(x.device).float()
    )
    return torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)


def rotate_every_two(x):
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = torch.stack((-x2, x1), axis=-1)
    return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')


def apply_rotary_pos_emb(x, sincos, offset=0):
    sin, cos = map(
        lambda t: t[None, offset : x.shape[1] + offset, None, :].repeat_interleave(2, 3), sincos
    )
    # einsum notation for lambda t: repeat(t[offset:x.shape[1]+offset,:], "n d -> () n () (d j)", j=2)
    return (x * cos) + (rotate_every_two(x) * sin)


class ProGenAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
        )
        self.register_buffer("masked_bias", torch.tensor(-1e9))

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.attn_pdrop = config.attn_pdrop
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        self.embed_dim = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_attention_heads
        if self.head_dim * self.num_attention_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_attention_heads (got `embed_dim`: {self.embed_dim} and `num_attention_heads`: {self.num_attention_heads})."
            )
        self.scale_attn = math.sqrt(self.head_dim)
        self.qkv_proj = nn.Linear(self.embed_dim, self.embed_dim * 3, bias=False)

        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.rotary_dim = None
        if config.rotary_dim is not None:
            self.rotary_dim = config.rotary_dim

    def _split_heads(self, x, n_head, dim_head, mp_num):
        reshaped = x.reshape(x.shape[:-1] + (n_head // mp_num, dim_head))
        reshaped = reshaped.reshape(x.shape[:-2] + (-1,) + reshaped.shape[-1:])
        return reshaped

    def _naive_attn(
        self,
        query,
        key,
        value,
        attention_mask=None,
    ):
        # compute causal mask from causal mask buffer
        batch_size, query_length, key_length = query.size(0), query.size(-2), key.size(-2)
        causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]

        attn_weights = torch.matmul(query, key.transpose(-1, -2)) / self.scale_attn
        attn_weights = torch.where(
            causal_mask, attn_weights, self.masked_bias.to(attn_weights.dtype)
        )

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_weights = self.attn_dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, value)

        expected_size = (batch_size, self.num_attention_heads, query_length, self.head_dim)
        if attn_output.size() != expected_size:
            raise ValueError(
                f"`attn_output` should be of size {expected_size}, but is  {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, query_length, self.embed_dim)
        return attn_output, attn_weights

    def _sdpa_attn(
        self,
        query,
        key,
        value,
        attention_mask=None,
    ):
        bsz, q_len = query.shape[0], query.shape[2]
        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query.device.type == "cuda" and attention_mask is not None:
            query = query.contiguous()
            key = key.contiguous()
            value = value.contiguous()

        attn_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=self.attn_pdrop if self.training else 0.0,
            # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
            is_causal=q_len > 1,
            scale=1 / self.scale_attn,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.embed_dim)
        return attn_output, None

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        layer_past=None,
        use_cache=False,
        output_attentions=False,
    ):
        qkv = self.qkv_proj(hidden_states)
        # TODO(enijkamp): factor out number of logical TPU-v3/v4 cores or make forward pass agnostic
        # mp_num = 4
        mp_num = 8
        qkv_split = qkv.reshape(qkv.shape[:-1] + (mp_num, -1))

        local_dim = self.head_dim * self.num_attention_heads // mp_num
        query, value, key = torch.split(qkv_split, local_dim, dim=-1)
        query = self._split_heads(query, self.num_attention_heads, self.head_dim, mp_num=mp_num)
        key = self._split_heads(key, self.num_attention_heads, self.head_dim, mp_num=mp_num)

        value = self._split_heads(value, self.num_attention_heads, self.head_dim, mp_num=mp_num)
        value = value.permute(0, 2, 1, 3)

        seq_len = key.shape[1]
        offset = 0

        if layer_past is not None:
            offset = layer_past[0].shape[-2]
            seq_len += offset

        if self.rotary_dim is not None:
            k_rot = key[:, :, :, : self.rotary_dim]
            k_pass = key[:, :, :, self.rotary_dim :]

            q_rot = query[:, :, :, : self.rotary_dim]
            q_pass = query[:, :, :, self.rotary_dim :]

            sincos = fixed_pos_embedding(k_rot, 1, seq_len=seq_len)
            k_rot = apply_rotary_pos_emb(k_rot, sincos, offset=offset)
            q_rot = apply_rotary_pos_emb(q_rot, sincos, offset=offset)

            key = torch.cat([k_rot, k_pass], dim=-1)
            query = torch.cat([q_rot, q_pass], dim=-1)
        else:
            sincos = fixed_pos_embedding(key, 1, seq_len=seq_len)
            key = apply_rotary_pos_emb(key, sincos, offset=offset)
            query = apply_rotary_pos_emb(query, sincos, offset=offset)

        key = key.permute(0, 2, 1, 3)
        query = query.permute(0, 2, 1, 3)

        if layer_past is not None:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in float16 just to be sure everything works as expected.

        input_dtype = query.dtype
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.qkv_proj.weight.dtype #this is giving an issue, but it usually isn't called

        if input_dtype != target_dtype:
            logger.warning_once(
                f"The input hidden states seems to be silently casted in {input_dtype}. "
                f"This might be because you have upcasted embedding or layer norm layers "
                f"in {input_dtype}. We will cast back the input in {target_dtype}."
            )
            query = query.to(target_dtype)
            key = key.to(target_dtype)
            value = value.to(target_dtype)

        # compute self-attention: V x Softmax(QK^T)
        if output_attentions:
            attn_output, attn_weights = self._naive_attn(query, key, value, attention_mask)
        else:
            attn_output, attn_weights = self._sdpa_attn(query, key, value, None)
        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class ProGenMLP(nn.Module):
    def __init__(self, intermediate_size, config):  # in MLP: intermediate_size= 4 * embed_dim
        super().__init__()
        embed_dim = config.n_embd

        self.fc_in = nn.Linear(embed_dim, intermediate_size)
        self.fc_out = nn.Linear(intermediate_size, embed_dim)

        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states):
        hidden_states = self.fc_in(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc_out(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class ProGenBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        inner_dim = config.n_inner if config.n_inner is not None else 4 * config.n_embd
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = ProGenAttention(config)
        self.mlp = ProGenMLP(inner_dim, config)

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        adapter_layer=None,
        adapter_dropout=None,
        adapter_input=None,
        use_cache=False,
        output_attentions=False,
    ):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states) 
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0] 
        outputs = attn_outputs[1:]

        feed_forward_hidden_states = self.mlp(hidden_states)
        
        ### addition of adapter layer ###
        if exists(adapter_layer) and exists(adapter_dropout) and exists(
                adapter_input):
            
            hidden_states_update = attn_output + feed_forward_hidden_states
            adapter_out = adapter_layer(hidden_states_update, adapter_input)
            adapter_out = adapter_dropout(adapter_out)
            hidden_states_update = hidden_states_update + adapter_out

            hidden_states = hidden_states_update + residual
        else:
            hidden_states = attn_output + feed_forward_hidden_states + residual
        ### end of addition of adapter layer ###

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]
        
        return outputs 


class ProGenPreTrainedModel(PreTrainedModel):
    """An abstract class to handle weights initialization and a simple interface for downloading
    and loading pretrained models."""

    config_class = ProGenConfig
    base_model_prefix = "transformer"
    is_parallelizable = True
    _no_split_modules = ["ProGenBlock"]

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear,)):
            # Slightly different from Mesh Transformer JAX which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

class ModularProGenModel(ProGenPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)

        self.embed_dim = config.n_embd
        self.vocab_size = config.vocab_size
        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList(
            [ProGenBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)
        self.rotary_dim = min(config.rotary_dim,
                              config.n_ctx // config.num_attention_heads)
        self.init_weights()

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def forward_prep(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states
                                if output_hidden_states is not None else
                                self.config.output_hidden_states)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        
        if getattr(self.config, "gradient_checkpointing",
                   False) and self.training:
            #print('using gradient checkpointing')
            if use_cache:
                use_cache = False
            
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])

        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)

        if position_ids is None:
            position_ids = torch.arange(past_length,
                                        input_shape[-1] + past_length,
                                        dtype=torch.long,
                                        device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # Attention mask.
        if attention_mask is not None:
            assert batch_size > 0, "batch_size has to be defined and > 0"
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(
                dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x num_attention_heads x N x N
        # head_mask has shape n_layer x batch x num_attention_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        return input_ids, attention_mask, head_mask, position_ids, token_type_ids, inputs_embeds, past_key_values, use_cache, output_attentions, output_hidden_states, return_dict

    def forward_embed(
        self,
        input_ids=None,
        token_type_ids=None,
        inputs_embeds=None,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)

        hidden_states = inputs_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        return hidden_states

    def forward_layer(
        self,
        hidden_states,
        layer_i,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        adapter_layer=None,
        adapter_dropout=None,
        adapter_input=None,
        use_cache=None,
        output_attentions=None,
    ):
        if getattr(self.config, "gradient_checkpointing",
                   False) and self.training:
            if use_cache:
                logger.warning(
                    "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                    "`use_cache=False`...")
                use_cache = False

            def create_custom_forward(module):

                def custom_forward(*inputs):
                    # None for past_key_value
                    return module(*inputs, use_cache, output_attentions)

                return custom_forward

            outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.h[layer_i]),
                hidden_states,
                None,
                attention_mask,
                head_mask[layer_i],
                adapter_layer, 
                adapter_dropout,
                adapter_input,
            )
        else:
            outputs = self.h[layer_i](
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask[layer_i],
                adapter_layer=adapter_layer,
                adapter_dropout=adapter_dropout,
                adapter_input=adapter_input,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

        hidden_states = outputs[0]

        if use_cache:
            presents = (outputs[1], )
        else:
            presents = None

        if output_attentions:
            self_attentions = outputs[2 if use_cache else 1]
        else:
            self_attentions = None

        return hidden_states, presents, self_attentions

    def forward_layers(
        self,
        hidden_states,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        all_presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for i in range(self.config.n_layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states, )

            hidden_states, presents, self_attentions = self.forward_layer(
                hidden_states,
                i,
                layer_past=past_key_values[i]
                if past_key_values is not None else None,
                attention_mask=attention_mask,
                head_mask=head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

            if use_cache is True:
                all_presents = all_presents + presents
            if output_attentions:
                all_self_attentions = all_self_attentions + (self_attentions, )

        return hidden_states, all_presents, all_self_attentions, all_hidden_states

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        input_shape = input_ids.size()
        input_ids, attention_mask, head_mask, position_ids, token_type_ids, inputs_embeds, past_key_values, use_cache, output_attentions, output_hidden_states, return_dict = self.forward_prep(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = self.forward_embed(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )

        hidden_states, all_presents, all_self_attentions, all_hidden_states = self.forward_layers(
            hidden_states=hidden_states,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        hidden_states = self(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1), )
        hidden_states = hidden_states.view(*output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states, )

        if not return_dict:
            return tuple(v for v in [
                hidden_states, all_presents, all_hidden_states,
                all_self_attentions
            ] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=all_presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

class ModularProGenForCausalLM(ProGenPreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"h\.\d+\.attn\.masked_bias", r"h\.\d+\.attn\.bias", r"lm_head\.weight"
    ]

    def __init__(self, config):
        super().__init__(config)

        self.transformer = ModularProGenModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        self.init_weights()

    def get_output_embeddings(self):
        return None

    def set_output_embeddings(self, new_embeddings):
        return

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            ``labels = input_ids`` Indices are selected in ``[-100, 0, ..., config.vocab_size]`` All labels set to
            ``-100`` are ignored (masked), the loss is only computed for labels in ``[0, ..., config.vocab_size]``
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # make sure sampling in fp16 works correctly and
        # compute loss in fp32 to match with mesh-tf version
        # https://github.com/EleutherAI/gpt-neo/blob/89ce74164da2fb16179106f54e2269b5da8db333/models/gpt2/gpt2.py#L179
        lm_logits = self.lm_head(hidden_states).to(torch.float32)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss() 
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1))

            loss = loss.to(hidden_states.dtype)

        if not return_dict:
            output = (lm_logits, ) + transformer_outputs[1:]
            return ((loss, ) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    @staticmethod
    def _reorder_cache(past: Tuple[Tuple[torch.Tensor]],
                       beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the :obj:`past_key_values` cache if
        :meth:`~transformers.PretrainedModel.beam_search` or :meth:`~transformers.PretrainedModel.beam_sample` is
        called. This is required to match :obj:`past_key_values` with the correct beam_idx at every generation step.
        """
        return tuple(
            tuple(
                past_state.index_select(0, beam_idx.to(past_state.device))
                for past_state in layer_past) for layer_past in past)

    
class ProgenConditional(ProGenPreTrainedModel): #nn.Module
    def __init__(self, config: ProGenConditionalConfig):
        super().__init__(config)

        #self.model = ModularProGenForCausalLM.from_pretrained(pretrained_model_name_or_path=config.pretrained_model_dir, config=config) #use this if you already have the pretrained model loaded to the data directory
        self.model = ModularProGenForCausalLM.from_pretrained("jsunn-y/ProCALM", subfolder=config.pretrained_model_name, config=config, cache_dir=config.pretrained_model_dir)
        self.model.requires_grad_(False) #freeze the pretrained model by default

        self.config = config

        self.projection_mlps = torch.nn.ModuleDict() #conditioning encoders
        if config.adapter_shared_projection == True:
            n_projection_mlps = 1 #sharing a projector
        else:
            n_projection_mlps = len(self.model.transformer.h) #having a projector for every layer

        for key, input_dim in config.encoding_dimensions.items():
            adapter_projection_layers = nn.ModuleList()
            for i in range(n_projection_mlps):
                if config.adapter_projection_nlayers == None:
                    projection_mlp = torch.nn.Linear(input_dim, config.adapter_c_s)
                else:
                    projection_mlp = ProjectionMLP(input_dim=input_dim, c_s=config.adapter_c_s, num_layers=config.adapter_projection_nlayers)
                adapter_projection_layers.append(projection_mlp)

            self.projection_mlps[key] = adapter_projection_layers
        
        #if using a shared adapter, append an extra MLP to process the summed input
        #not necessary if you have a separate adapter for each layer
        #this one is always nonlinear and uses two layers
        if (config.conditions_shared_adapter == True) and (len(config.encoding_dimensions.values()) >=2):
            adapter_projection_layers = nn.ModuleList()
            for i in range(n_projection_mlps):
                projection_mlp = ProjectionMLP(input_dim=config.adapter_c_s, c_s=config.adapter_c_s, num_layers=2)
                adapter_projection_layers.append(projection_mlp)

            self.projection_mlps["combination"] = adapter_projection_layers

        #initialize the adapter layers
        self.adapter_layers = torch.nn.ModuleList()
        if config.conditions_shared_adapter == False:
            keys = config.encoding_dimensions.keys()
        else:
            keys = ["joint"]
        n_parallel = len(keys)
        
        for i in range(len(self.model.transformer.h)):
            parallel_adapter_layer = ParallelAdapterLayer(
            n_parallel=n_parallel,
            c_s=config.adapter_c_s, 
            c_h=config.n_embd,
            adapter_summation=config.adapter_summation,
            weight_init=config.adapter_weight_init,
            adapter_nlayers=config.adapter_nlayers,
            )
            adapter_dropout = torch.nn.Dropout(config.adapter_dropout)
            self.adapter_layers.append(nn.ModuleList([parallel_adapter_layer, adapter_dropout]))

    def prepare_inputs_for_generation(self, input_ids, condition_encodings: Dict[str, torch.tensor] = None, past=None, **kwargs):
        """
        Overides the prepare inputs for generation function (HF compatible) to allow for the addition of adapter input.
        """
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        past = kwargs.get("past_key_values", past)
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        
        adapter_input = {}
        for key, condition_encoding in condition_encodings.items():
            if condition_encoding is not None:
                single_adapter_input = condition_encoding.repeat(input_ids.shape[0], input_ids.shape[1], 1)
            else:
                single_adapter_input = None
            adapter_input[key] = single_adapter_input

        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "adapter_input": adapter_input,
        }

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        if isinstance(past_key_values, Cache):
            return past_key_values.reorder_cache(beam_idx)

        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return DynamicCache.from_legacy_cache(reordered_past)
    
    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        adapter_input=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_shape = input_ids.size()

        input_ids, attention_mask, head_mask, position_ids, token_type_ids, inputs_embeds, past_key_values, use_cache, output_attentions, output_hidden_states, return_dict = self.model.transformer.forward_prep(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = self.model.transformer.forward_embed(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )

        all_presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        #project the condition to the dimension of the adapter
        #if sharing a single projection layer
        #else do nothing until we get into the loop
        if self.config.adapter_shared_projection == True:
            encoded_adapter_input = ()
            #if you're sharing an adapter and doing joint conditioning
            if len(adapter_input.keys()) >= 2  and self.config.conditions_shared_adapter == True:
                summed_adapter_input = torch.zeros(input_shape[0], input_shape[1], self.config.adapter_c_s).to(input_ids.device)
                for key, single_adapter_input in adapter_input.items():
                    projected_adapter_input = self.projection_mlps[key][0](single_adapter_input)
                    summed_adapter_input += projected_adapter_input
                
                #combine the inputs and pass through one
                key = "combination"
                summed_adapter_input = self.projection_mlps[key][0](summed_adapter_input)
                encoded_adapter_input = (summed_adapter_input, )
            
            #if you're not sharing an adapter (with or without multiple conditions)
            else:
                for key, value in adapter_input.items():
                    summed_adapter_input = self.projection_mlps[key][0](value)
                    encoded_adapter_input = encoded_adapter_input + (summed_adapter_input, )                
            encoded_adapter_input = torch.stack(encoded_adapter_input, dim=0)

        for i in range(len(self.model.transformer.h)):
            #if not sharing a projection layer
            if self.config.adapter_shared_projection == False:
                encoded_adapter_input = ()
                #if you're sharing an adapter and doing joint conditioning
                if len(adapter_input.keys()) >= 2 and self.config.conditions_shared_adapter == True:
                    summed_adapter_input = torch.zeros(input_shape[0], input_shape[1], self.config.adapter_c_s).to(input_ids.device)
                    for key, single_adapter_input in adapter_input.items():
                        projected_adapter_input = self.projection_mlps[key][i](single_adapter_input)
                        encoded_adapter_input += projected_adapter_input
                    
                    #combine the inputs and pass through one more mlp
                    key = "combination"
                    summed_adapter_input = self.projection_mlps[key][i](summed_adapter_input)
                    encoded_adapter_input = (summed_adapter_input, )

                #if you're not sharing an adapter (with or without multiple conditions)
                else:
                    for key, value in adapter_input.items():
                        summed_adapter_input = self.projection_mlps[key][i](value)
                        encoded_adapter_input = encoded_adapter_input + (summed_adapter_input, )
                encoded_adapter_input = torch.stack(encoded_adapter_input, dim=0)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states, )

            hidden_states, presents, self_attentions = self.model.transformer.forward_layer(
                hidden_states=hidden_states,
                layer_i=i,
                layer_past=past_key_values[i] if past_key_values[i] is not None else None,
                attention_mask=attention_mask,
                head_mask=head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                adapter_layer=self.adapter_layers[i][0],
                adapter_dropout=self.adapter_layers[i][1],
                adapter_input=encoded_adapter_input,
            )

            if use_cache is True:
                all_presents = all_presents + presents
            if output_attentions:
                all_self_attentions = all_self_attentions + (self_attentions, )
            
        hidden_states = self.model.transformer.ln_f(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1), )
        hidden_states = hidden_states.view(*output_shape)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states, )

        if not return_dict:
            return tuple(v for v in [
                hidden_states, all_presents, all_hidden_states,
                all_self_attentions
            ] if v is not None)

        transformer_outputs = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=all_presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
        
        hidden_states = transformer_outputs[0]

        # make sure sampling in fp16 works correctly and
        # compute loss in fp32 to match with mesh-tf version
        # https://github.com/EleutherAI/gpt-neo/blob/89ce74164da2fb16179106f54e2269b5da8db333/models/gpt2/gpt2.py#L179
        lm_logits = self.model.lm_head(hidden_states).to(torch.float32)

        loss = None
        all_losses = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            #added this so that the loss of each sample is outputted
            loss_fct = CrossEntropyLoss(ignore_index=0, reduction='none')
            all_losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1))
            all_losses = all_losses.to(hidden_states.dtype)
            
            #still output the mean reduced loss
            loss_fct = CrossEntropyLoss(ignore_index=0)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits, ) + transformer_outputs[1:]
            return ((loss, ) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            all_losses=all_losses,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )