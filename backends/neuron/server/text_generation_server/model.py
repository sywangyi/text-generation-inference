import os
import shutil
import time
from typing import Optional

from huggingface_hub import snapshot_download
from huggingface_hub.constants import HF_HUB_CACHE
from loguru import logger

from optimum.neuron.cache import get_hub_cached_entries
from optimum.neuron.configuration_utils import NeuronConfig


from .tgi_env import check_env_and_neuron_config_compatibility


def get_export_kwargs_from_env():
    batch_size = os.environ.get("MAX_BATCH_SIZE", None)
    if batch_size is not None:
        batch_size = int(batch_size)
    sequence_length = os.environ.get("MAX_TOTAL_TOKENS", None)
    if sequence_length is not None:
        sequence_length = int(sequence_length)
    num_cores = os.environ.get("HF_NUM_CORES", None)
    if num_cores is not None:
        num_cores = int(num_cores)
    auto_cast_type = os.environ.get("HF_AUTO_CAST_TYPE", None)
    return {
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "num_cores": num_cores,
        "auto_cast_type": auto_cast_type,
    }


def is_cached(model_id):
    # Look for cached entries for the specified model
    in_cache = False
    entries = get_hub_cached_entries(model_id)
    # Look for compatible entries
    for entry in entries:
        if check_env_and_neuron_config_compatibility(
            entry, check_compiler_version=True
        ):
            in_cache = True
            break
    return in_cache


def log_cache_size():
    path = HF_HUB_CACHE
    if os.path.exists(path):
        usage = shutil.disk_usage(path)
        gb = 2**30
        logger.info(
            f"Cache disk [{path}]: total = {usage.total / gb:.2f} G, free = {usage.free / gb:.2f} G"
        )
    else:
        raise ValueError(f"The cache directory ({path}) does not exist.")


def fetch_model(
    model_id: str,
    revision: Optional[str] = None,
) -> str:
    """Fetch a neuron model.

    Args:
        model_id (`str`):
            The *model_id* of a model on the HuggingFace hub or the path to a local model.
        revision (`Optional[str]`, defaults to `None`):
            The revision of the model on the HuggingFace hub.

    Returns:
        A string corresponding to the model_id or path.
    """
    if not os.path.isdir("/sys/class/neuron_device/"):
        raise SystemError("No neuron cores detected on the host.")
    if os.path.isdir(model_id) and revision is not None:
        logger.warning(
            "Revision {} ignored for local model at {}".format(revision, model_id)
        )
        revision = None
    # Download the model from the Hub (HUGGING_FACE_HUB_TOKEN must be set for a private or gated model)
    # Note that the model may already be present in the cache.
    try:
        neuron_config = NeuronConfig.from_pretrained(model_id, revision=revision)
    except Exception as e:
        logger.debug(
            "NeuronConfig.from_pretrained failed for model %s, revision %s: %s",
            model_id,
            revision,
            e,
        )
        neuron_config = None
    if neuron_config is not None:
        if os.path.isdir(model_id):
            return model_id
        # Prefetch the neuron model from the Hub
        logger.info(
            f"Fetching revision [{revision}] for neuron model {model_id} under {HF_HUB_CACHE}"
        )
        log_cache_size()
        return snapshot_download(model_id, revision=revision, ignore_patterns="*.bin")
    # Model needs to be exported: look for compatible cached entries on the hub
    if not is_cached(model_id):
        hub_cache_url = "https://huggingface.co/aws-neuron/optimum-neuron-cache"
        neuron_export_url = "https://huggingface.co/docs/optimum-neuron/main/en/guides/export_model#exporting-neuron-models-using-neuronx-tgi"
        error_msg = (
            f"No cached version found for {model_id} with {get_export_kwargs_from_env()}."
            f"You can start a discussion to request it on {hub_cache_url}"
            f"Alternatively, you can export your own neuron model as explained in {neuron_export_url}"
        )
        raise ValueError(error_msg)
    logger.warning(
        f"{model_id} is not a neuron model: it will be exported using cached artifacts."
    )
    if os.path.isdir(model_id):
        return model_id
    # Prefetch weights, tokenizer and generation config so that they are in cache
    log_cache_size()
    start = time.time()
    snapshot_path = snapshot_download(
        model_id, revision=revision, ignore_patterns="*.bin"
    )
    end = time.time()
    logger.info(f"Model weights fetched in {end - start:.2f} s.")
    log_cache_size()
    return snapshot_path
