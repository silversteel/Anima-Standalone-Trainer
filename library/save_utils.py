import json
import os
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from types import MethodType
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedType
from accelerate.utils.other import clean_state_dict_for_safetensors
from safetensors.torch import load_file as safetensors_load_file
from safetensors.torch import save_file as safetensors_save_file


SAVE_INTENT_LORA_STATE = "lora_state"
SAVE_INTENT_FULL_MODEL_EXPORT = "full_model_export"
SAVE_INTENT_RESUME_STATE_MODEL_PAYLOAD = "resume_state_model_payload"


@dataclass(frozen=True)
class StateDictModelSpec:
    model: Any
    filename: str = "model"
    save_intent: str = SAVE_INTENT_FULL_MODEL_EXPORT
    unwrap_model: bool = True
    keep_torch_compile: bool = False
    target_model: Any = None


def is_fsdp_active(accelerator: Optional[Accelerator]) -> bool:
    return accelerator is not None and accelerator.distributed_type == DistributedType.FSDP


def unwrap_model(accelerator: Optional[Accelerator], model: Any, *, unwrap: bool, keep_torch_compile: bool):
    if model is None or accelerator is None or not unwrap:
        return model
    return accelerator.unwrap_model(model, keep_torch_compile=keep_torch_compile)


def get_model_state_dict_for_save(
    accelerator: Optional[Accelerator],
    model: Any,
    save_intent: str,
    *,
    unwrap_model_for_non_fsdp: bool = True,
    keep_torch_compile: bool = False,
):
    if model is None:
        return None

    # LoRA network is never FSDP-wrapped (kept outside accelerator.prepare), so state_dict() is safe directly.
    if save_intent == SAVE_INTENT_LORA_STATE:
        target_model = unwrap_model(
            accelerator,
            model,
            unwrap=unwrap_model_for_non_fsdp,
            keep_torch_compile=keep_torch_compile,
        )
        return target_model.state_dict()

    if is_fsdp_active(accelerator):
        return accelerator.get_state_dict(model)

    target_model = unwrap_model(
        accelerator,
        model,
        unwrap=unwrap_model_for_non_fsdp,
        keep_torch_compile=keep_torch_compile,
    )
    return target_model.state_dict()


@contextmanager
def override_model_state_dict(model: Any, state_dict: Dict[str, torch.Tensor]):
    original_state_dict = model.state_dict
    model.state_dict = MethodType(lambda _self, *args, **kwargs: state_dict, model)
    try:
        yield model
    finally:
        model.state_dict = original_state_dict


@contextmanager
def override_model_state_dicts_for_save(
    accelerator: Optional[Accelerator],
    specs: Sequence[StateDictModelSpec],
):
    with ExitStack() as stack:
        for spec in specs:
            state_dict = get_model_state_dict_for_save(
                accelerator,
                spec.model,
                spec.save_intent,
                unwrap_model_for_non_fsdp=spec.unwrap_model,
                keep_torch_compile=spec.keep_torch_compile,
            )
            target_model = spec.target_model if spec.target_model is not None else spec.model
            if target_model is None or state_dict is None:
                continue
            stack.enter_context(override_model_state_dict(target_model, state_dict))
        yield


def save_state_dict_to_safetensors(
    accelerator: Optional[Accelerator],
    model: Any,
    save_path: str,
    save_intent: str,
    *,
    metadata: Optional[Dict[str, str]] = None,
    unwrap_model_for_non_fsdp: bool = True,
    keep_torch_compile: bool = False,
):
    state_dict = get_model_state_dict_for_save(
        accelerator,
        model,
        save_intent,
        unwrap_model_for_non_fsdp=unwrap_model_for_non_fsdp,
        keep_torch_compile=keep_torch_compile,
    )
    state_dict = clean_state_dict_for_safetensors(state_dict)
    safetensors_save_file(state_dict, save_path, metadata=metadata or {"format": "pt"})


def write_train_state_metadata(output_dir: str, epoch: int, step: int):
    train_state_file = os.path.join(output_dir, "train_state.json")
    with open(train_state_file, "w", encoding="utf-8") as f:
        json.dump({"current_epoch": epoch, "current_step": step}, f)
    return train_state_file


def load_train_state_metadata(input_dir: str) -> Optional[Dict[str, int]]:
    train_state_file = os.path.join(input_dir, "train_state.json")
    if not os.path.exists(train_state_file):
        return None

    with open(train_state_file, "r", encoding="utf-8") as f:
        return json.load(f)


def create_safetensors_state_hooks(
    accelerator: Accelerator,
    specs: Sequence[StateDictModelSpec],
    *,
    get_current_epoch: Callable[[], int],
    get_current_step: Callable[[], int],
    allow_non_main_process_save: bool = False,
    use_accelerate_native_fsdp: bool = False,
):
    state_tracker: Dict[str, Optional[int]] = {"current_step": None}

    def save_model_hook(models, weights, output_dir):
        native_fsdp = use_accelerate_native_fsdp and is_fsdp_active(accelerator)

        if not native_fsdp:
            if accelerator.is_main_process or allow_non_main_process_save:
                for spec in specs:
                    save_path = os.path.join(output_dir, f"{spec.filename}.safetensors")
                    save_state_dict_to_safetensors(
                        accelerator,
                        spec.model,
                        save_path,
                        spec.save_intent,
                        unwrap_model_for_non_fsdp=spec.unwrap_model,
                        keep_torch_compile=spec.keep_torch_compile,
                    )
            weights.clear()

        if accelerator.is_main_process:
            write_train_state_metadata(output_dir, get_current_epoch(), get_current_step())

    def load_model_hook(models, input_dir):
        metadata = load_train_state_metadata(input_dir)
        if metadata is not None:
            state_tracker["current_step"] = metadata["current_step"]

        native_fsdp = use_accelerate_native_fsdp and is_fsdp_active(accelerator)
        if native_fsdp:
            return

        for spec in specs:
            load_path = os.path.join(input_dir, f"{spec.filename}.safetensors")
            if not os.path.exists(load_path):
                continue

            base_model = unwrap_model(
                accelerator,
                spec.model,
                unwrap=spec.unwrap_model,
                keep_torch_compile=spec.keep_torch_compile,
            )
            state_dict = safetensors_load_file(load_path, device="cpu")
            base_model.load_state_dict(state_dict)

        models.clear()

    return save_model_hook, load_model_hook, state_tracker
