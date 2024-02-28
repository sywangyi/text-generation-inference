import torch
from loguru import logger
from text_generation_server.utils.import_utils import IS_CUDA_SYSTEM, IS_ROCM_SYSTEM, IS_XPU_SYSTEM
# vllm imports
# logger.info("is cuda: {}".format(IS_CUDA_SYSTEM))
# logger.info("is amd: {}".format(IS_ROCM_SYSTEM))
if IS_CUDA_SYSTEM or IS_ROCM_SYSTEM:
    from vllm import cache_ops
    from vllm import attention_ops

_PARTITION_SIZE = 512



def ref_reshape_and_cache(key, value, key_cache, value_cache, slots):
    x = 16 // torch.tensor([], dtype=key.dtype).element_size()
    num_key_value_heads = key.shape[-2]
    head_size = key.shape[-1]
    reshaped_key = key.reshape(-1, num_key_value_heads, head_size // x, x)
    num_tokens = value.shape[0]
    block_size = value_cache.shape[3]
    for i in range(num_tokens):
        block_idx = torch.div(slots[i], block_size, rounding_mode='floor')
        block_offset = slots[i] % block_size
        key_cache[block_idx, :, :, block_offset, :] = reshaped_key[i]
        value_cache[block_idx, :, :, block_offset] = value[i]


def reshape_and_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slots: torch.Tensor,
):
    if IS_CUDA_SYSTEM or IS_ROCM_SYSTEM:
        cache_ops.reshape_and_cache(key, value, key_cache, value_cache, slots)
    elif IS_XPU_SYSTEM:
        import intel_extension_for_pytorch as ipex
        ipex.llm.modules.PagedAttention.reshape_and_cache(key, value, key_cache, value_cache, slots)
        # torch.xpu.reshape_and_cache(key, value, key_cache, value_cache, slots)
    else:
        raise NotImplementedError("reshape and cache do not support on current system")


def ref_attention(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
) -> None:
    num_heads = value_cache.shape[1]
    head_size = value_cache.shape[2]
    block_size = value_cache.shape[3]

    num_input_tokens = query.shape[0]
    for i in range(num_input_tokens):
        q = query[i].unsqueeze(0)
        block_table = block_tables[i]
        context_len = int(context_lens[i])
        keys = []
        values = []
        for j in range(context_len):
            block_number = int(block_table[j // block_size])
            block_offset = j % block_size

            k = key_cache[block_number, :, :, block_offset, :]
            k = k.reshape(num_heads, head_size)
            keys.append(k)

            v = value_cache[block_number, :, :, block_offset]
            values.append(v)
        keys = torch.stack(keys, dim=0)
        values = torch.stack(values, dim=0)

        scale = 1.0 / (head_size ** 0.5)
        out = torch.nn.functional.scaled_dot_product_attention(q.view(1,-1,num_heads,head_size).transpose(1,2),
                keys.view(1,-1,num_heads,head_size).transpose(1,2),
                values.view(1,-1,num_heads,head_size).transpose(1,2),
                attn_mask=None, dropout_p=0.0, is_causal=False)
        out = out.view(num_heads, head_size)
        output[i].copy_(out, non_blocking=True)

def attention(
    out: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    kv_head_mapping: torch.Tensor,
    softmax_scale: float,
    block_tables: torch.Tensor,
    input_lengths: torch.Tensor,
    max_s: int,
):
    # Adapted from: https://github.com/vllm-project/vllm/blob/f8a1e39fae05ca610be8d5a78be9d40f5274e5fc/vllm/model_executor/layers/attention.py
    # Copyright 2023 The vLLM team. All rights
    # reserved.
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
    #

    # value_cache => [num_blocks, num_heads, head_size, block_size]
    block_size = value_cache.shape[3]
    num_seqs, num_heads, head_size = query.shape
    max_num_partitions = (max_s + _PARTITION_SIZE - 1) // _PARTITION_SIZE
    # NOTE(woosuk): We use a simple heuristic to decide whether to use
    # PagedAttention V1 or V2. If the number of partitions is 1, we use
    # V1 to avoid the overhead of reduction. Also, if the number of
    # sequences or heads is large, we use V1 since there is enough work
    # to parallelize.
    if IS_XPU_SYSTEM:
        import intel_extension_for_pytorch as ipex
        query = query.contiguous()
        return ipex.llm.modules.PagedAttention.single_query_cached_kv_attention(
            out,
            query,
            key_cache,
            value_cache,
            kv_head_mapping,
            softmax_scale,
            block_tables,
            input_lengths,
            block_size,
            max_s,
            None
        )
        # return torch.xpu.IpexPaged_attention(
        #     out,
        #     query,
        #     key_cache,
        #     value_cache,
        #     kv_head_mapping,
        #     block_tables,
        #     input_lengths,
        #     softmax_scale,
        #     block_size,
        #     max_s,
        #     None
        # )
    use_v1 = max_num_partitions == 1 or num_seqs * num_heads > 512
    if use_v1:
        attention_ops.paged_attention_v1(
            out,
            query,
            key_cache,
            value_cache,
            kv_head_mapping,
            softmax_scale,
            block_tables,
            input_lengths,
            block_size,
            max_s,
            None,
        )
    else:
        # Run PagedAttention V2.
        assert _PARTITION_SIZE % block_size == 0
        tmp_output = torch.empty(
            size=(num_seqs, num_heads, max_num_partitions, head_size),
            dtype=out.dtype,
            device=out.device,
        )
        exp_sums = torch.empty(
            size=(num_seqs, num_heads, max_num_partitions),
            dtype=torch.float32,
            device=out.device,
        )
        max_logits = torch.empty_like(exp_sums)
        attention_ops.paged_attention_v2(
            out,
            exp_sums,
            max_logits,
            tmp_output,
            query,
            key_cache,
            value_cache,
            kv_head_mapping,
            softmax_scale,
            block_tables,
            input_lengths,
            block_size,
            max_s,
            None,
        )
