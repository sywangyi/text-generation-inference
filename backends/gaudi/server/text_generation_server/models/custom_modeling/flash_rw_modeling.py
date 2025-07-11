from typing import List, Optional, Tuple

import torch
import torch.distributed
from torch import nn
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from text_generation_server.layers import (
    SpeculativeHead,
    TensorParallelColumnLinear,
    TensorParallelEmbedding,
    TensorParallelRowLinear,
    get_linear,
)
from text_generation_server.layers.attention.kv_cache import get_kv_scales
from text_generation_server.layers.layernorm import FastLayerNorm
from text_generation_server.layers.rotary import PositionRotaryEmbedding
from text_generation_server.layers.attention import (
    attention,
    paged_attention,
    set_block_mapping,
    Seqlen,
    HPUPagedAttentionMetadata,
)
import habana_frameworks.torch as htorch


def load_row(config, prefix: str, weights, bias: bool):
    weight = weights.get_weights_row(prefix)

    if bias and weights.process_group.rank() == 0:
        # Rank is only on the first rank process
        bias = weights.get_tensor(f"{prefix}.bias")
    else:
        bias = None

    linear = get_linear(weight, bias)
    if config.parallel_attn:
        return linear
    else:
        return TensorParallelRowLinear(linear, process_group=weights.process_group)


class RWConfig(PretrainedConfig):
    attribute_map = {
        "num_hidden_layers": "n_layer",
        "num_attention_heads": "n_head",
        "num_key_value_heads": "n_head_kv",
    }

    def __init__(
        self,
        model_type="RefinedWeb",
        vocab_size=250880,
        hidden_size=64,
        num_hidden_layers=None,
        num_attention_heads=None,
        num_ln_in_prallel_attention=None,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        use_cache=True,
        bos_token_id=1,
        eos_token_id=2,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        num_kv_heads=None,
        multi_query=False,
        alibi=False,
        new_decoder_architecture=None,
        bias=False,
        parallel_attn=False,
        rope_theta=10_000.0,
        **kwargs,
    ):
        if alibi:
            raise NotImplementedError(
                "alibi is not supported by this version of the model"
            )

        self.model_type = model_type
        self.alibi = False
        self.rotary = True
        self.rope_theta = rope_theta
        self.max_position_embeddings = 2048

        self.vocab_size = vocab_size
        # Backward compatibility with n_embed kwarg
        n_embed = kwargs.pop("n_embed", None)
        self.hidden_size = hidden_size if n_embed is None else n_embed
        self.n_layer = (
            num_hidden_layers
            if num_hidden_layers is not None
            else kwargs.pop("n_layer", 2)
        )
        self.n_head = (
            num_attention_heads
            if num_attention_heads is not None
            else kwargs.pop("n_head", 8)
        )
        self.layer_norm_epsilon = layer_norm_epsilon
        self.num_ln_in_parallel_attn = num_ln_in_prallel_attention
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.bias = bias
        self.parallel_attn = parallel_attn

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        if num_kv_heads is not None:
            self.n_head_kv = num_kv_heads
        else:
            old_n_head_kv = kwargs.pop("n_head_kv", None)
            if old_n_head_kv is not None:
                self.n_head_kv = old_n_head_kv
            else:
                self.n_head_kv = 1 if multi_query else self.n_head

        if new_decoder_architecture is not None:
            self.new_decoder_architecture = new_decoder_architecture
        elif model_type == "RefinedWeb":
            self.new_decoder_architecture = True
        else:
            self.new_decoder_architecture = False

        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)


class FlashRWAttention(torch.nn.Module):
    def __init__(
        self,
        config,
        prefix: str,
        weights,
        rotary_emb,
    ):
        super().__init__()
        self.num_heads = config.n_head
        self.num_heads_kv = config.n_head_kv
        self.hidden_size = config.hidden_size
        self.head_size = self.hidden_size // self.num_heads
        self.rope_theta = config.rope_theta
        self.rotary_emb = rotary_emb

        self.softmax_scale = self.head_size ** (-0.5)

        if self.num_heads % weights.process_group.size() != 0:
            raise ValueError(
                f"`num_heads` must be divisible by `num_shards` (got `num_heads`: {self.num_heads} "
                f"and `num_shards`: {weights.process_group.size()}"
            )
        self.num_heads = self.num_heads // weights.process_group.size()

        self.query_key_value = TensorParallelColumnLinear.load(
            config,
            prefix=f"{prefix}.query_key_value",
            weights=weights,
            bias=config.bias,
        )
        self.kv_scales = get_kv_scales(weights, f"{prefix}")
        self.dense = load_row(
            config, prefix=f"{prefix}.dense", weights=weights, bias=config.bias
        )

        if self.num_heads_kv == 1:
            self.kv_head_mapping = torch.zeros(
                self.num_heads, dtype=torch.int32, device=weights.device
            )
        else:
            self.kv_head_mapping = torch.arange(
                0, self.num_heads, dtype=torch.int32, device=weights.device
            )

    def forward(
        self,
        hidden_states,
        cos,
        sin,
        cu_seqlen_prefill,
        kv_cache,
        slots,
        seqlen,
        hpu_attention_meta,
    ):
        qkv = self.query_key_value(hidden_states)

        # Split query from key_value
        query, kv = qkv.split(
            [self.head_size * self.num_heads, 2 * self.head_size * self.num_heads_kv],
            dim=1,
        )

        # Prepare query and key_value for indexing
        query = query.view(-1, self.num_heads, self.head_size)
        kv = kv.view(-1, 2, self.num_heads_kv, self.head_size)

        # Inplace rotary
        self.rotary_emb(query, torch.select(kv, dim=1, index=0), cos, sin)

        kv_cache.store(
            key=kv[:, 0],
            value=kv[:, 1],
            slots=slots,
            kv_scales=self.kv_scales,
        )

        # Prefill
        if cu_seqlen_prefill is not None:
            # sdpa
            attn_output = attention(
                query=query,
                key=kv[:, 0],
                value=kv[:, 1],
                kv_cache=kv_cache,
                kv_scales=self.kv_scales,
                seqlen=seqlen,
                softmax_scale=self.softmax_scale,
            )
        # Decode
        else:
            attn_output = paged_attention(
                query,
                kv_cache,
                self.kv_head_mapping,
                self.softmax_scale,
                seqlen,
                kv_scales=self.kv_scales,
                hpu_attention_meta=hpu_attention_meta,
            )

        return self.dense(attn_output.view(-1, self.num_heads * self.head_size))


class FlashRWLargeAttention(torch.nn.Module):
    def __init__(
        self,
        config,
        prefix: str,
        weights,
        rotary_emb,
    ):
        super().__init__()

        hidden_size = config.hidden_size
        num_heads = config.n_head
        # num_heads_kv = config.n_head_kv
        num_groups = config.n_head_kv

        self.hidden_size = hidden_size
        self.head_size = hidden_size // num_heads
        self.num_groups = num_groups
        self.rope_theta = config.rope_theta
        self.rotary_emb = rotary_emb

        self.softmax_scale = self.head_size ** (-0.5)

        # self.num_groups = num_heads // (num_heads_kv * 2)
        self.num_heads = num_heads // self.num_groups
        # self.num_heads_kv = num_heads_kv // self.num_groups
        process_group = weights.process_group

        if process_group.size() > self.num_groups:
            raise NotImplementedError(
                "Tensor Parallelism is not implemented for world_size > n groups"
            )
        if self.num_groups % process_group.size() != 0:
            raise NotImplementedError(
                f"Tensor Parallelism is not implemented for {self.num_groups} not divisible by {process_group.size()}"
            )

        self.num_groups = self.num_groups // process_group.size()

        self.query_key_value = TensorParallelColumnLinear.load(
            config,
            prefix=f"{prefix}.query_key_value",
            weights=weights,
            bias=config.bias,
        )
        self.kv_scales = get_kv_scales(weights, f"{prefix}")
        self.dense = load_row(
            config, prefix=f"{prefix}.dense", weights=weights, bias=config.bias
        )

        self.kv_head_mapping = torch.arange(
            0, self.num_groups, dtype=torch.int32, device=weights.device
        ).repeat_interleave(self.num_heads)

    def forward(
        self,
        hidden_states,
        cos,
        sin,
        cu_seqlen_prefill,
        kv_cache,
        slots,
        seqlen,
        hpu_attention_meta,
    ):
        qkv = self.query_key_value(hidden_states)
        qkv = qkv.view(-1, self.num_groups, self.num_heads + 2, self.head_size)

        # Split on group dimension
        query, kv = qkv.split(
            [self.num_heads, 2],
            dim=2,
        )
        # Merge groups and heads
        query = query.reshape(-1, self.num_groups * self.num_heads, self.head_size)

        # Inplace rotary
        self.rotary_emb(query, torch.select(kv, dim=2, index=0), cos, sin)

        kv_cache.store(
            key=kv[:, :, 0].contiguous(),
            value=kv[:, :, 1].contiguous(),
            slots=slots,
            kv_scales=self.kv_scales,
        )

        # Prefill
        if cu_seqlen_prefill is not None:
            # flash attention
            attn_output = attention(
                query=query,
                key=kv[:, :, 0],
                value=kv[:, :, 1],
                kv_cache=kv_cache,
                kv_scales=self.kv_scales,
                seqlen=seqlen,
                softmax_scale=self.softmax_scale,
            )
        # Decode
        else:
            attn_output = paged_attention(
                query,
                kv_cache,
                self.kv_head_mapping,
                self.softmax_scale,
                seqlen,
                kv_scales=self.kv_scales,
                hpu_attention_meta=hpu_attention_meta,
            )

        return self.dense(
            attn_output.view(-1, self.num_groups * self.num_heads * self.head_size)
        )


class FlashMLP(nn.Module):
    def __init__(self, config, prefix: str, weights):
        super().__init__()
        self.act = torch.nn.functional.gelu

        self.dense_h_to_4h = TensorParallelColumnLinear.load(
            config, prefix=f"{prefix}.dense_h_to_4h", weights=weights, bias=config.bias
        )
        self.dense_4h_to_h = load_row(
            config, prefix=f"{prefix}.dense_4h_to_h", weights=weights, bias=config.bias
        )

    def forward(self, hidden_states):
        hidden_states = self.dense_h_to_4h(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dense_4h_to_h(hidden_states)
        return hidden_states


class FlashRWLayer(nn.Module):
    def __init__(
        self,
        layer_id,
        prefix: str,
        config,
        weights,
        rotary_emb,
    ):
        super().__init__()

        parallel_attn = config.parallel_attn
        self.parallel_attn = parallel_attn

        prefix = f"{prefix}.h.{layer_id}"

        # NOTE: Falcon 180B uses the ln_attn prefix
        ln_prefix = "input_layernorm"
        if config.num_hidden_layers == 80:
            ln_prefix = "ln_attn"

        self.input_layernorm = FastLayerNorm.load(
            prefix=f"{prefix}.{ln_prefix}",
            weights=weights,
            eps=config.layer_norm_epsilon,
        )
        self.self_attention = FlashRWAttention(
            config,
            prefix=f"{prefix}.self_attention",
            weights=weights,
            rotary_emb=rotary_emb,
        )
        self.post_attention_layernorm = (
            FastLayerNorm.load(
                prefix=f"{prefix}.post_attention_layernorm",
                weights=weights,
                eps=config.layer_norm_epsilon,
            )
            if not parallel_attn
            else None
        )

        self.mlp = FlashMLP(
            config,
            prefix=f"{prefix}.mlp",
            weights=weights,
        )

        self.process_group = weights.process_group

    def forward(
        self,
        hidden_states,
        residual,
        cos,
        sin,
        cu_seqlen_prefill,
        kv_cache,
        slots,
        seqlen,
        hpu_attention_meta,
    ):
        if self.parallel_attn:
            ln_hidden_states, residual = self.input_layernorm(hidden_states, residual)

            attn_output = self.self_attention(
                ln_hidden_states,
                cos,
                sin,
                cu_seqlen_prefill,
                kv_cache,
                slots,
                seqlen,
                hpu_attention_meta,
            )

            mlp_output = self.mlp(ln_hidden_states)
            intermediate = mlp_output + attn_output

            if self.process_group.size() > 1:
                torch.distributed.all_reduce(intermediate, group=self.process_group)

            return intermediate, residual
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

            hidden_states = self.self_attention(
                hidden_states,
                cos,
                sin,
                cu_seqlen_prefill,
                kv_cache,
                slots,
                seqlen,
                hpu_attention_meta,
            )

            if self.post_attention_layernorm is not None:
                hidden_states, residual = self.post_attention_layernorm(
                    hidden_states, residual
                )

            mlp_output = self.mlp(hidden_states)

            return mlp_output, residual


class FlashRWLayerNorm(nn.Module):
    def __init__(self, config, prefix: str, weights):
        super().__init__()
        # Falcon2 includes the number of layer norms in the config
        # in the case no number of layer norms is provided, we default to 1
        self.num_ln = getattr(config, "num_ln_in_parallel_attn", 1)

        # Falcon 180B uses the ln_attn prefix and has 2 layer norms
        if config.num_hidden_layers == 80:
            self.num_ln = 2

        if self.num_ln == 1:
            self.input_ln = FastLayerNorm.load(
                prefix=f"{prefix}.input_layernorm",
                weights=weights,
                eps=config.layer_norm_epsilon,
            )
        elif self.num_ln == 2:
            self.ln_attn = FastLayerNorm.load(
                prefix=f"{prefix}.ln_attn",
                weights=weights,
                eps=config.layer_norm_epsilon,
            )
            self.ln_mlp = FastLayerNorm.load(
                prefix=f"{prefix}.ln_mlp",
                weights=weights,
                eps=config.layer_norm_epsilon,
            )
        else:
            raise ValueError("Number of layer norms can either be 1 or 2.")

    def forward(
        self,
        hidden_states,
        residual,
    ):
        if self.num_ln == 1:
            ln_hidden_states, residual = self.input_ln(hidden_states, residual)
            return ln_hidden_states, ln_hidden_states, residual
        elif self.num_ln == 2:
            ln_attn, residual = self.ln_attn(hidden_states, residual)
            ln_mlp, _ = self.ln_mlp(residual)
            return ln_attn, ln_mlp, residual


class FlashRWLargeLayer(nn.Module):
    def __init__(self, layer_id, prefix: str, config, weights, rotary_emb):
        super().__init__()
        prefix = f"{prefix}.h.{layer_id}"

        self.ln_layer = FlashRWLayerNorm(config, prefix, weights)

        self.self_attention = FlashRWLargeAttention(
            config,
            prefix=f"{prefix}.self_attention",
            weights=weights,
            rotary_emb=rotary_emb,
        )
        assert config.parallel_attn, "This version doesn't support non parallel_attn"

        self.mlp = FlashMLP(config, prefix=f"{prefix}.mlp", weights=weights)

        self.process_group = weights.process_group

    def forward(
        self,
        hidden_states,
        residual,
        cos,
        sin,
        cu_seqlen_prefill,
        kv_cache,
        slots,
        seqlen,
        hpu_attention_meta,
    ):
        # Layer norm.
        ln_attn, ln_mlp, residual = self.ln_layer(hidden_states, residual)

        # Self attention.
        attn_output = self.self_attention(
            ln_attn,
            cos,
            sin,
            cu_seqlen_prefill,
            kv_cache,
            slots,
            seqlen,
            hpu_attention_meta,
        )

        # MLP.
        mlp_output = self.mlp(ln_mlp)

        intermediate = attn_output + mlp_output

        if self.process_group.size() > 1:
            torch.distributed.all_reduce(intermediate, group=self.process_group)

        return intermediate, residual


class FlashRWPreTrainedModel(PreTrainedModel):
    config_class = RWConfig


class FlashRWModel(FlashRWPreTrainedModel):
    def __init__(self, prefix: str, config, weights):
        super().__init__(config)
        self.config = config

        self.word_embeddings = TensorParallelEmbedding(
            prefix=f"{prefix}.word_embeddings", weights=weights
        )
        rotary_emb = PositionRotaryEmbedding.static(
            config=config,
            dim=config.hidden_size // config.n_head,
            base=config.rope_theta,
            device=weights.device,
        )

        if config.new_decoder_architecture:
            self.h = nn.ModuleList(
                [
                    FlashRWLargeLayer(layer_id, prefix, config, weights, rotary_emb)
                    for layer_id in range(config.num_hidden_layers)
                ]
            )
            self.cache_size = self.h[0].self_attention.num_groups
        else:
            self.h = nn.ModuleList(
                [
                    FlashRWLayer(layer_id, prefix, config, weights, rotary_emb)
                    for layer_id in range(config.num_hidden_layers)
                ]
            )
            self.cache_size = self.h[0].self_attention.num_heads_kv

        self.ln_f = FastLayerNorm.load(
            prefix=f"{prefix}.ln_f",
            weights=weights,
            eps=config.layer_norm_epsilon,
        )

        self.head_size = self.h[0].self_attention.head_size

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cu_seqlen_prefill: Optional[torch.Tensor],
        kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
        slots: torch.Tensor,
        seqlen: Seqlen,
        hpu_attention_meta: Optional[HPUPagedAttentionMetadata],
    ) -> torch.Tensor:
        if hpu_attention_meta is not None:
            hpu_attention_meta = set_block_mapping(
                hpu_attention_meta, input_ids.shape[0]
            )
        hidden_states = self.word_embeddings(input_ids)

        # Get rotary cos and sin for this forward
        # Avoid to index in each layer
        cos, sin = self.h[0].self_attention.rotary_emb.get_cos_sin(position_ids)

        residual = None
        lazy_mode = htorch.utils.internal.is_lazy()
        if lazy_mode:
            htorch.core.mark_step()
        for i, layer in enumerate(self.h):
            hidden_states, residual = layer(
                hidden_states,
                residual,
                cos,
                sin,
                cu_seqlen_prefill,
                kv_cache[i],
                slots,
                seqlen,
                hpu_attention_meta,
            )
            if lazy_mode:
                htorch.core.mark_step()

        hidden_states, _ = self.ln_f(hidden_states, residual)

        return hidden_states


class FlashRWForCausalLM(FlashRWPreTrainedModel):
    def __init__(self, prefix: str, config, weights):
        super().__init__(config)

        if not prefix:
            prefix = "transformer"
        else:
            prefix = f"{prefix}.transformer"

        self.transformer = FlashRWModel(prefix, config, weights)

        self.lm_head = SpeculativeHead.load(config, prefix="lm_head", weights=weights)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cu_seqlen_prefill: Optional[torch.Tensor],
        kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
        slots: torch.Tensor,
        seqlen: Seqlen,
        hpu_attention_meta: Optional[HPUPagedAttentionMetadata],
        lm_head_indices: Optional[torch.Tensor] = None,
        adapter_data: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.transformer(
            input_ids,
            position_ids,
            cu_seqlen_prefill,
            kv_cache,
            slots,
            seqlen,
            hpu_attention_meta,
        )
        if lm_head_indices is not None:
            hidden_states = hidden_states[lm_head_indices]
        logits = self.lm_head(hidden_states)
        return logits
