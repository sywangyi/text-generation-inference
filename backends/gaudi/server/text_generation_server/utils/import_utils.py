import torch
from loguru import logger
import habana_frameworks.torch as htorch
import os
from vllm_hpu_extension.profiler import format_bytes
from text_generation_server.utils.log import log_master


def get_hpu_free_memory(device, memory_fraction):
    graph_reserved_mem = (
        float(os.environ.get("TGI_GRAPH_RESERVED_MEM", "0.1"))
        if htorch.utils.internal.is_lazy()
        else 0
    )
    free_hpu_memory, total_hpu_memory = torch.hpu.mem_get_info()
    mem_reserved = total_hpu_memory * (1 - memory_fraction)
    free_memory_for_kv_cache = int(
        (free_hpu_memory - mem_reserved) * (1 - graph_reserved_mem)
    )
    mem_used_from_graph = int((free_hpu_memory - mem_reserved) * graph_reserved_mem)
    log_master(
        logger.info,
        f"Free memory on device {device}: {format_bytes(free_hpu_memory)}/{format_bytes(total_hpu_memory)}, used_for_kv_cache: {format_bytes(free_memory_for_kv_cache)}, used_for_graph: {format_bytes(mem_used_from_graph)} ratio{graph_reserved_mem} reserved_for_runtime: {format_bytes(mem_reserved)}",
    )
    return free_memory_for_kv_cache, mem_used_from_graph, mem_reserved


def synchronize_hpu(device):
    torch.hpu.synchronize()


def noop(*args, **kwargs):
    pass


empty_cache = noop
synchronize = synchronize_hpu
get_free_memory = get_hpu_free_memory
