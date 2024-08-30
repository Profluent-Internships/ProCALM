import torch
import torch.nn as nn

from .primitives import Linear, LayerNorm
from ..utils import exists

class ProjectionMLP(nn.Module):
    """
    Small MLP for projecting the conditioning vectors to the adapter dimension. Also called the conditioning encoder.
    """

    def __init__(
        self, 
        input_dim: int, #encoding dimension
        c_h: int=128, #hidden layer size
        c_s: int=128, #adapter input dimension
        num_layers: int=2, #number of layers in the MLP
        bias: bool=True,
    ) -> None:
        super().__init__()

        # Linear layers for MLP
        self.W_in = nn.Linear(input_dim, c_h, bias=bias)
        self.W_inter = nn.ModuleList([
            nn.Linear(c_h, c_h, bias=bias)
            for _ in range(num_layers - 2)
        ])
        self.W_out = nn.Linear(c_h, c_s, bias=bias)

        # Activation function
        self.act = nn.ReLU()

    def forward(self, x):

        # Embed inputs with input layer
        x = self.act(self.W_in(x))

        # Pass through intermediate layers
        for layer in self.W_inter:
            x = self.act(layer(x))

        # Get output from output layer
        x = self.W_out(x)

        return x
    
class MLP(nn.Module):
    """
    MLP to be used in the adapter layers.
    """

    def __init__(
        self, 
        c_in, #input dimmension
        c_hidden, #hidden layer dimension
        c_out, #output dimension
        num_layers, #number of layers
        bias=True):
        super().__init__()

        # Linear layers for MLP
        self.W_in = nn.Linear(c_in, c_hidden, bias=bias)
        self.W_inter = nn.ModuleList([
            nn.Linear(c_hidden, c_hidden, bias=bias)
            for _ in range(num_layers - 2)
        ])
        self.W_out = nn.Linear(c_hidden, c_out, bias=bias)

        # Activation function
        self.act = nn.ReLU()

    def forward(self, x):

        # Embed inputs with input layer
        x = self.act(self.W_in(x))

        # Pass through intermediate layers
        for layer in self.W_inter:
            x = self.act(layer(x))

        # Get output from output layer
        x = self.W_out(x)

        return x
                
class AdapterLayer(nn.Module):
    """
    A single adapter layer.
    """

    def __init__(
        self,
        c_s: int=128, #dimensionality of the adapter input (projected conditioning vectors)
        c_h: int=1536, #hidden dimenionsion of the language model (1536 in progen)
        c_hidden: int = 16, #low-rank dimenion of the LM embeddings
        low_rank_cond: bool = True,
        low_rank_mlp: bool = True,
        adapter_summation: bool = False, #sum conditon and hidden state instead of concatenating
        s_reduction: int = None, #not used anymore
        dropout_rate: float = 0.1,
        weight_init: float = 1e-5,
        adapter_nlayers: int = 2,
    ) -> None:
        super().__init__()
    
        self.low_rank_cond = low_rank_cond
        self.low_rank_mlp = low_rank_mlp
        self.adapter_summation = adapter_summation
        if adapter_summation:
            c_hidden = c_s #override c_hidden with c_s

        self.h_dropout = nn.Dropout(dropout_rate)
        self.h_ln = LayerNorm(c_h)

        self.s_dropout = nn.Dropout(dropout_rate)
        self.s_ln = LayerNorm(c_s)

        if exists(s_reduction):
            self.s_down = Linear(c_s, c_s // s_reduction, init="default")
            c_s = c_s // s_reduction
        else:
            self.s_down = None

        if low_rank_cond:
            c_down_in = c_h
            c_down_out = c_hidden
            if not adapter_summation:
                c_hidden = c_hidden + c_s
        else:
            c_down_in = c_h + c_s
            c_down_out = c_hidden

        self.linear_down = Linear(c_down_in, c_down_out, init="default")
        self.linear_up = Linear(c_hidden, c_h, init="final")

        #init last layer with gaussian noise near 0
        self.linear_up.weight.data.normal_(mean=0.0, std=weight_init)
        self.linear_up.bias.data.zero_()

        self.act = nn.ReLU()

        if low_rank_mlp:
            self.mlp = MLP(
                c_hidden,
                c_hidden*2,
                c_hidden,
                num_layers=adapter_nlayers, 
            )

    def forward(self, h: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        h = self.h_dropout(self.h_ln(h.float()))
        s = self.s_dropout(self.s_ln(s))

        if exists(self.s_down):
            s = self.s_down(s)

        if not self.low_rank_cond:
            x = torch.cat([h, s], dim=-1)
        else:
            x = h

        x = self.linear_down(x)

        if self.low_rank_cond:
            if self.adapter_summation:
                x = x + s
            else:
                x = torch.cat([s, x], dim=-1)

        if self.low_rank_mlp:
            x = self.mlp(x)

        x = self.act(x)
        x = self.linear_up(x)

        return x

class ParallelAdapterLayer(nn.Module):
    """
    Wrapper around AdapterLayer that supports one or more conditions being passed through the adapter in parallel.
    """
    def __init__(
        self,
        n_parallel: int=1, #number of separate conditions to pass through in parallel
        c_s: int=128, #dimensionality of the adapter input (projected conditioning vectors)
        c_h: int=1536, #hidden dimenionsion of the language model (1536 in progen)
        c_hidden: int = 16, #low-rank dimenion of the LM embeddings
        low_rank_cond: bool = True,
        low_rank_mlp: bool = True,
        adapter_summation: bool = False, #sum conditon and hidden state instead of concatenating
        s_reduction: int = None, #not used anymore
        dropout_rate: float = 0.1,
        weight_init: float = 1e-5,
        adapter_nlayers: int = 2,
    ) -> None:
        super().__init__()
        
        self.parallel_adapter_layer = nn.ModuleList()
        for i in range(n_parallel):
            adapter_layer = AdapterLayer(
                c_s=c_s,
                c_h=c_h,
                c_hidden=c_hidden,
                low_rank_cond=low_rank_cond,
                low_rank_mlp=low_rank_mlp,
                adapter_summation=adapter_summation,
                s_reduction=s_reduction,
                dropout_rate=dropout_rate,
                weight_init=weight_init,
                adapter_nlayers=adapter_nlayers,
            )
            self.parallel_adapter_layer.append(adapter_layer)
    
    def forward(self, h: torch.Tensor, s_parallel: torch.Tensor) -> torch.Tensor:
        """
        h: hidden state of the language model
        s_parallel: stacked tensors, each representing a separate condition as the adapter input
        """
        updates = ()
        for adapter_layer, s in zip(self.parallel_adapter_layer, s_parallel):
            update = adapter_layer(h, s)
            updates = updates + (update,)

        output = torch.stack(updates, dim=0).sum(dim=0)
        
        return output