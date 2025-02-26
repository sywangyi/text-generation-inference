import torch
import itertools
from text_generation_server.layers.attention import Seqlen
from typing import Optional, List
from text_generation_server.layers.attention.kv_cache import KVCache, KVScales
from vllm_hpu_extension import ops
from vllm_hpu_extension.utils import Matmul
from habana_frameworks.torch.hpex.kernels import FusedSDPA

SUPPORTS_WINDOWING = False


def fetch_from_cache(cache, blocks):
    return cache.index_select(0, blocks)


def attention(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: KVCache,
    kv_scales: KVScales,
    seqlen: Seqlen,
    block_tables: torch.Tensor,
    softmax_scale: float,
    window_size_left: int = -1,
    causal: bool = True,
    softcap: Optional[float] = None,
):
    query = query.unsqueeze(0).transpose(1, 2)
    key = key.unsqueeze(0).transpose(1, 2)
    value = value.unsqueeze(0).transpose(1, 2)
    attn_output = FusedSDPA.apply(query, key, value, None, 0.0, causal, None)
    attn_output = attn_output.transpose(1, 2).squeeze(0).contiguous()
    return attn_output


def paged_attention(
    query: torch.Tensor,
    kv_cache: KVCache,
    kv_head_mapping: torch.Tensor,
    softmax_scale: float,
    block_tables: torch.Tensor,
    seqlen: Seqlen,
    max_s: int,
    *,
    kv_scales: KVScales,
    softcap: Optional[float] = None,
):
    batch_size, seq_len, hidden_size = query.shape
    blocks_used = [len(bt) for bt in block_tables if bt]
    block_list = []
    block_scales = []
    for i, bt in enumerate(block_tables):
        block_list.extend(bt)
        blocks_in_group = len(bt)
        if blocks_in_group > 0:
            scale = 1.0 / blocks_in_group
            block_scales.extend([scale] * blocks_in_group)

    block_mapping_nested: List[List[int]] = [
        [i] * b_u for i, b_u in enumerate(blocks_used)
    ]
    block_mapping: List[int] = list(itertools.chain.from_iterable(block_mapping_nested))
    block_list = torch.tensor(block_list, dtype=torch.int, device="hpu")
    block_mapping = torch.tensor(block_mapping, dtype=torch.long, device="hpu")
    block_scales = torch.tensor(block_scales, dtype=torch.bfloat16, device="hpu")
    output = ops.flat_pa(
        query=query,
        key_cache=kv_cache.key,
        value_cache=kv_cache.value,
        block_list=block_list,
        block_mapping=block_mapping,
        block_bias=None,
        block_scales=block_scales,
        block_groups=None,
        scale=softmax_scale,
        matmul_qk_op=Matmul(),
        matmul_av_op=Matmul(),
        batch2block_matmul_op=Matmul(),
        block2batch_matmul_op=Matmul(),
        keys_fetch_func=fetch_from_cache,
        values_fetch_func=fetch_from_cache,
    )
    # Reshape the output tensor.
    return output.view(batch_size, seq_len, hidden_size)


__all__ = [
    "SUPPORTS_WINDOWING",
    "attention",
    "paged_attention",
]
