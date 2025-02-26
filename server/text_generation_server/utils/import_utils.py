import torch
from loguru import logger
import os


import importlib.util


def is_ipex_available():
    return importlib.util.find_spec("intel_extension_for_pytorch") is not None


def is_hpu_available():
    "Checks if `torch.hpu` is installed and potentially if a HPU is in the environment"
    if importlib.util.find_spec("habana_frameworks") is None:
        return False

    import habana_frameworks.torch.hpu as torch_hpu  # noqa: F401

    if os.environ.get("HABANA_VISIBLE_MODULES", "") != "":
        true_device_count = len(os.environ.get("HABANA_VISIBLE_MODULES").split(","))
        if true_device_count != torch_hpu.device_count():
            logger.warning(
                f"Detected {torch_hpu.device_count()} HPU devices, but HABANA_VISIBLE_MODULES is set to "
                f"{os.environ.get('HABANA_VISIBLE_MODULES')} (i.e. {true_device_count} devices). "
                "Patching torch.hpu.device_count() to return the correct number of devices."
            )
            torch_hpu.device_count = lambda: true_device_count

    return torch_hpu.device_count() > 0


def get_cuda_free_memory(device, memory_fraction):
    total_free_memory, _ = torch.cuda.mem_get_info(device)
    total_gpu_memory = torch.cuda.get_device_properties(device).total_memory
    free_memory = max(0, total_free_memory - (1 - memory_fraction) * total_gpu_memory)
    return free_memory


def get_hpu_free_memory(device, memory_fraction):
    from habana_frameworks.torch.hpu import memory_stats

    device_id = device.index
    mem_stats = memory_stats(device_id)
    logger.info(f"mem_stats: {mem_stats}")
    total_free_memory = mem_stats["Limit"] - mem_stats["MaxInUse"]
    memory_fraction = float(os.getenv("HPU_MEMORY_FRACTION", "0.8"))
    free_memory = max(0, int(total_free_memory * memory_fraction))
    return free_memory


def get_xpu_free_memory(device, memory_fraction):
    total_memory = torch.xpu.get_device_properties(device).total_memory
    device_id = device.index
    memory_fraction = float(os.getenv("XPU_MEMORY_FRACTION", "1.0"))
    free_memory = max(
        0,
        int(
            total_memory * 0.9 * memory_fraction - torch.xpu.memory_reserved(device_id)
        ),
    )

    return free_memory


def get_cpu_free_memory(device, memory_fraction):
    import psutil
    from text_generation_server.utils.dist import WORLD_SIZE

    mem = psutil.virtual_memory()
    free_memory = int(mem.available * 0.95 / WORLD_SIZE)
    return free_memory


def synchronize_hpu(device):
    torch.hpu.synchronize()


def noop(*args, **kwargs):
    pass


SYSTEM = None
if torch.version.hip is not None:
    SYSTEM = "rocm"
    empty_cache = torch.cuda.empty_cache
    synchronize = torch.cuda.synchronize
    get_free_memory = get_cuda_free_memory
elif torch.version.cuda is not None and torch.cuda.is_available():
    SYSTEM = "cuda"
    empty_cache = torch.cuda.empty_cache
    synchronize = torch.cuda.synchronize
    get_free_memory = get_cuda_free_memory
elif is_ipex_available():
    SYSTEM = "ipex"
    import intel_extension_for_pytorch  # noqa: F401

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        empty_cache = torch.xpu.empty_cache
        synchronize = torch.xpu.synchronize
        get_free_memory = get_xpu_free_memory
    else:
        empty_cache = noop
        synchronize = noop
        get_free_memory = get_cpu_free_memory
elif hasattr(torch, "xpu") and torch.xpu.is_available():
    SYSTEM = "xpu"
    empty_cache = torch.xpu.empty_cache
    synchronize = torch.xpu.synchronize
    get_free_memory = get_xpu_free_memory
elif is_hpu_available():
    SYSTEM = "hpu"
    empty_cache = noop
    synchronize = synchronize_hpu
    get_free_memory = get_hpu_free_memory
else:
    SYSTEM = "cpu"

    empty_cache = noop
    synchronize = noop
    get_free_memory = get_cpu_free_memory
logger.info(f"Detected system {SYSTEM}")
