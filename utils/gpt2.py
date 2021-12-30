from typing import Optional, Tuple
import torch
from torch.functional import Tensor
import torch.nn as nn
from math import sqrt
import torch.nn.functional as F


# class ScaledDotProductAttention(nn.Module):
#     def __init__(self, temperature: int, prop: float = 0.1) -> None:
#         super().__init__()

#         self.temperature = temperature
#         self.dropout = nn.Dropout(p=prop)

#     def forward(self, q, k, v, mask=None) -> Dict[Tensor, Tensor]:
#         scores = torch.bmm(q, k.transpose(0, 2, 1))/sqrt(self.temperature)

#         if mask is not None:
#             scores = scores.masked_fill(mask == 0, float('-inf'))
#         weights = F.softmax(scores, dim=-1)

#         return torch.bmm(weights, v), weights

class GPT2Config(object):
    """
    This is the configuration class of GPT-2 model

    Args:
        vocab_size (`int`, defaults to 32000):
            Vocabulary size of the rinna GPT-2 model. Defines the number of different tokens that can be represented by the `input_ids`.
        n_positions (`int`, defaults to 512):
            Aka.max_position_embeddings: The maximum sequence length that this model can process. (e.g., 512, 1024 or 2048)
        n_embed (`int`, defaults to 768):
            Aka. hidden_size: the demensionality of the embediddings and hidden states.
        n_layer (`int`, defaults to 12):
            Number of hidden layers in the Transformer encoder
        n_head (`int`, defaults to 8):
            Aka. num_heads: number of attention heads for each attention layer in the Transformer encoder.
        n_inner (`int`, defaults to 3072):
            Aka. intermediate_size: Dimensionality of the inner feed-forward layers. Default will set it to 4 times `n_embed`. 
        activation_func (`str`, defaults to `"gelu"`):
            Activation function, to be selected in the list `["relu", "silu", "gelu", "tanh", "gelu_new"]`.
        resid_pdrop (`float`, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        embed_pdrop (`float`, defaults to 0.1):
            The dropout ratio for the embeddings.
        attn_pdrop: (`float`, defaults to 0.1):
            The dropout ratio for the attention.
        layer_norm_epsilon (`float`, defaults to 1e-5):
            The epsilon to use in the layer normalization layers.
    """

    def __init__(self,
                 vocab_size: int = 32000,
                 n_positions: int = 512,
                 n_embed: int = 768,
                 n_layer: int = 12,
                 n_head: int = 8,
                 n_inner: int = 3072,
                 activation_func: str = 'gelu',
                 resid_pdrop: float = 0.1,
                 embed_pdrop: float = 0.1,
                 attn_pdrop: float = 0.1,
                 layer_norm_epsilon: float = 1e-5) -> None:

        if activation_func not in ["relu", "silu", "gelu", "tanh", "gelu_new"]:
            raise ValueError(f"Activation name {activation_func} not in: ['relu', 'silu', 'gelu', 'tanh', 'gelu_new']")
        self.vocab_size = vocab_size
        self.n_embed = n_embed
        self.n_head = n_head
        self.n_positions = n_positions
        self.n_layer = n_layer
        self.n_inner = n_inner
        self.activation_func = activation_func
        self.resid_pdrop = resid_pdrop
        self.embed_pdrop = embed_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon


    def __repr__(self):
        return """(config):
        vocab_size: %d
        n_embed: %d
        n_head: %d
        n_positions: %d
        n_layer: %d
        n_inner: %d
        activation_func: %s
        resid_pdrop: %.1f
        embed_pdrop: %.1f
        attn_pdrop: %.1f
        layer_norm_epsilon: %.5f""" % (self.vocab_size,
                                    self.n_embed,
                                    self.n_head,
                                    self.n_positions,
                                    self.n_layer,
                                    self.n_inner,
                                    self.activation_func,
                                    self.resid_pdrop,
                                    self.embed_pdrop,
                                    self.attn_pdrop,
                                    self.layer_norm_epsilon)


def scaled_dot_product(query: Tensor,
                       key: Tensor,
                       value: Tensor,
                       heads: int,
                       mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
    dim_k = query.size(-1)

    scores = torch.bmm(query, key.transpose(1, 2))/sqrt(dim_k)
    if mask is not None:
        mask_repeat_size = (heads,) + tuple(1 for _ in range(mask.dim()-1))
        mask = mask.repeat(mask_repeat_size)
        scores = scores.masked_fill(mask == 0, float('-inf'))
    weights = F.softmax(scores, dim=-1)

    return torch.bmm(weights, value), weights


class MultiheadAttention(nn.Module):
    def __init__(self, config) -> None:
        super(MultiheadAttention, self).__init__()

        self.n_embed = config.n_embed
        self.n_head = config.n_head
        self.head_dim = self.n_embed//self.n_head
        self.q = nn.Linear(self.n_embed, self.n_head*self.head_dim)
        self.k = nn.Linear(self.n_embed, self.n_head*self.head_dim)
        self.v = nn.Linear(self.n_embed, self.n_head*self.head_dim)

        self.fc = nn.Linear(self.n_embed, self.n_embed)

    def forward(self,
                queries: Tensor,
                keys: Tensor,
                values: Tensor,
                mask=None) -> Tuple[Tensor, Tensor]:

        heads = self.n_head
        head_dim = self.head_dim
        batch_size, seq_len, embed_dim = keys.size()
        assert embed_dim == self.n_embed, f"Input embedding dim ({embed_dim}) must match layer embedding dim {self.n_embed}"

        queries, keys, values = self._split_head(
            queries, keys, values, batch_size, heads, seq_len, head_dim)

        # compute scaled dot-product attention
        queries = queries.transpose(1, 2).contiguous().view(
            batch_size*heads, seq_len, head_dim)
        keys = keys.transpose(1, 2).contiguous().view(
            batch_size*heads, seq_len, head_dim)
        values = values.transpose(1, 2).contiguous().view(
            batch_size*heads, seq_len, head_dim)

        # attention size of: [batch_size*heads, seq_len, head_dim]
        # weights size of: [batch_size*heads, seq_len, seq_len]
        attn, weights = scaled_dot_product(queries, keys, values, heads, mask)

        attn, weights = self._merge_head(
            attn, weights, batch_size, heads, seq_len, head_dim)

        return self.fc(attn), weights

    def _split_head(self,
                    queries: Tensor,
                    keys: Tensor,
                    values: Tensor,
                    batch_size: int,
                    heads: int,
                    seq_len: int,
                    head_dim: int) -> Tuple[Tensor, Tensor, Tensor]:

        queries = queries.view(batch_size, seq_len, heads, head_dim)
        keys = keys.view(batch_size, seq_len, heads, head_dim)
        values = values.view(batch_size, seq_len, heads, head_dim)

        return queries, keys, values

    def _merge_head(self,
                    attn: Tensor,
                    weights: Tensor,
                    batch_size: int,
                    heads: int,
                    seq_len: int,
                    head_dim: int) -> Tuple[Tensor, Tensor]:

        attn = attn.view(batch_size, heads, seq_len, head_dim).transpose(1, 2)
        attn = attn.contiguous().view(batch_size, seq_len, heads*head_dim)

        weights = weights.view(batch_size, heads, seq_len, seq_len)

        return attn, weights


class FeedForward(nn.Module):
    def __init__(self, config) -> None:
        super(FeedForward, self).__init__()

        if config.activation_func=='gelu':
            atv_func = nn.GELU()
        elif config.activation_func=='silu':
            atv_func = nn.SiLU()
        elif config.activation_func=='relu':
            atv_func =nn.ReLU()
        else:
            atv_func=nn.Tanh()
        
        self.fc = nn.Sequential(
            nn.Linear(config.n_embed, config.n_inner),
            atv_func,
            nn.Linear(config.n_inner, config.n_embed)
        )

        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.fc(hidden_states)

        return self.dropout(hidden_states)


class DecoderLayer(nn.Module):
    def __init__(self, config) -> None:
        super(DecoderLayer, self).__init__()

        self.mask_mha = MultiheadAttention(config)
        self.fc = FeedForward(config)
        self.ln_1 = nn.LayerNorm(config.n_embed, eps=config.layer_norm_epsilon)
        self.ln_2 = nn.LayerNorm(config.n_embed, eps=config.layer_norm_epsilon)

    def forward(self,
                hidden_state: Tensor,
                mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        attns, weights = self.mask_mha(hidden_state, hidden_state, hidden_state, mask)
        hidden_state = hidden_state + attns
        hidden_state = self.ln_1(hidden_state)

        hidden_state = hidden_state + self.fc(hidden_state)
        return self.ln_2(hidden_state), weights


class Embedding(nn.Module):
    def __init__(self, config) -> None:
        super(Embedding, self).__init__()

        self.token_embeddings = nn.Embedding(
            config.vocab_size, config.n_embed)
        self.position_embeddings = nn.Embedding(
            config.n_positions, config.n_embed)
        self.dropout = nn.Dropout(config.embed_pdrop)
        self.ln = nn.LayerNorm(config.n_embed, config.layer_norm_epsilon)

    def forward(self, x: Tensor) -> Tensor:
        pos_id = torch.arange(x.size(1), dtype=torch.long, device=torch.device(x.device)).unsqueeze(0)
        token_embed = self.token_embeddings(x)
        pos_embed = self.position_embeddings(pos_id)
        embed = token_embed + pos_embed
        embed = self.ln(embed)

        return self.dropout(embed)


class Decoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        self.embeddings = Embedding(config)
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.n_layer)])

    def forward(self,
                x: Tensor,
                mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        
        h_state = self.embeddings(x)
        for layer in self.layers:
            h_state, weights = layer(h_state, mask)

        return h_state, weights


class GPT2Model(nn.Module):
    def __init__(self, config) -> None:
        super(GPT2Model, self).__init__()

        self.decoder = Decoder(config)
        self.fc = nn.Linear(config.n_embed, config.vocab_size)

    def _mask_pad_idx(self,
                      input_ids: Tensor,
                      pad_idx: int) -> Tensor:

        return (input_ids != pad_idx).unsqueeze(1)

    def forward(self,
                input_ids: Tensor,
                pad_idx: int) -> Tuple[Tensor, Tensor]:
        
        mask_pad = self._mask_pad_idx(input_ids, pad_idx)

        seq_len = input_ids.size(-1)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=torch.device(input_ids.device))).view(1, seq_len, seq_len).bool()
        mask = mask & mask_pad
        h_state, weights = self.decoder(input_ids, mask)

        logits = self.fc(h_state)

        return logits, weights