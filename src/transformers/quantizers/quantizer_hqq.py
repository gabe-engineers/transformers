# Copyright 2024 The HuggingFace Team. All rights reserved.
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

import os
import tempfile
from typing import TYPE_CHECKING

from ..integrations import prepare_for_hqq_linear
from ..utils import is_hqq_available, is_torch_available, logging
from .base import HfQuantizer
from .quantizers_utils import get_module_from_name


if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel
    from ..utils.quantization_config import HqqConfig


if is_torch_available():
    import torch

if is_hqq_available():
    from hqq.core.quantize import HQQLinear

    # This is a compatibility hack. HQQ-quantized linear layers do not have a `weight` attribute,
    # but some models attempt to access `weight.dtype` during the forward pass. To prevent runtime errors,
    # we patch HQQLinear with a dummy `weight` property that returns an empty tensor with the correct dtype and device.
    @property
    def weight(self):
        return torch.empty(0, dtype=self.compute_dtype, device=self.device)

    HQQLinear.weight = weight

logger = logging.get_logger(__name__)


class HqqHfQuantizer(HfQuantizer):
    """
    HQQ quantizer base HF class.
    nn.Linear modules are first tagged with quant_config in _process_model_before_weight_loading().
    """

    requires_calibration = False
    quantization_config: "HqqConfig"

    def __init__(self, quantization_config, **kwargs):
        if not is_hqq_available():
            raise ImportError(
                "A valid HQQ version (>=0.2.1) is not available. Please follow the instructions to install it: `https://github.com/mobiusml/hqq/`."
            )
        super().__init__(quantization_config, **kwargs)
        self.dtype = None
        self.using_multi_gpu = False
        # Keys that are serialized specifically by hqq
        self.hqq_keys = HQQLinear(None, None).state_dict_keys() - {"bias"}

    def validate_environment(self, *args, **kwargs):
        if self.dtype is None:
            if "dtype" in kwargs:
                self.dtype = kwargs["dtype"]
            else:
                self.dtype = torch.float32
                logger.info("Setting dtype to torch.float32 as the default value since it was not specified.")

        device_map = kwargs.get("device_map")
        if isinstance(device_map, dict):
            if "cpu" in device_map.values() or "disk" in device_map.values():
                raise ValueError(
                    "You are attempting to use an HQQ model with a device_map that contains a CPU or disk device."
                    " This is not supported. Please remove the CPU or disk device from the device_map."
                )
            else:
                self.using_multi_gpu = len(set(device_map.values())) > 1

    # TODO: to remove
    # Kept here in case we see some interest in adding support for it
    # # Adds missing keys for HQQLinear modules that are loaded but the model with initialized with torch.nn.Linear
    # def update_expected_keys(
    #     self, model: "PreTrainedModel", expected_keys: list[str], loaded_keys: list[str]
    # ) -> list[str]:
    #     if not self.pre_quantized:
    #         return expected_keys

    #     # Collects all quantizable (linear) layers
    #     def _find_hqq_quantizable_layers(model, layers):
    #         for name, module in model.named_children():
    #             if isinstance(module, (torch.nn.Linear)):
    #                 layers.add(module.name)
    #             _find_hqq_quantizable_layers(module, layers)

    #     new_keys = set(expected_keys)

    #     # Name modules
    #     for name, module in model.named_modules():
    #         module.name = name

    #     # valid modules are Linear layers that have HQQLinear state_dict. We ignore skip_modules and any layers with Linear state_dict() params
    #     _valid_modules = set()
    #     _find_hqq_quantizable_layers(model, _valid_modules)

    #     # Remove skipped modules
    #     _skipped_modules = set()
    #     for _module in _valid_modules:
    #         for _skip_module in model.config.quantization_config["skip_modules"]:
    #             if _skip_module in _module:
    #                 _skipped_modules.add(_module)
    #     _valid_modules -= _skipped_modules

    #     # Append new expected layers based on _ref_keys
    #     _ref_keys = HQQLinear(
    #         linear_layer=None,
    #         quant_config=None,
    #         compute_dtype=torch.float16,
    #         device="cpu",
    #         del_orig=False,
    #     ).state_dict_keys() - {"bias"}

    #     # Clean-up
    #     _rm_keys = set()
    #     for key in new_keys:
    #         if any(_module in key for _module in _valid_modules):
    #             _rm_keys.add(key)
    #     new_keys -= _rm_keys
    #     # At this point, new_keys contains all the keys of the layers that are NOT HQQLinear or torch.nn.Linear

    #     # Re-populate Linear/HQQLinear
    #     for _module in _valid_modules:
    #         if _module + ".weight" in loaded_keys:
    #             new_keys.add(_module + ".weight")
    #         else:
    #             new_keys.update({_module + "." + _ref_key for _ref_key in _ref_keys})
    #         if _module + ".bias" in loaded_keys:
    #             new_keys.add(_module + ".bias")

    #     return list(new_keys)

    def param_needs_quantization(self, model: "PreTrainedModel", param_name: str, **kwargs) -> bool:
        module, _ = get_module_from_name(model, param_name)
        # Since we do not prepare the modules in advance, we need every param of the Linear layer to go through
        # `create_quantized_param`, even when `self.is_quantized == True`
        return isinstance(module, torch.nn.Linear)

    # TODO: to remove
    # def create_quantized_param(
    #     self,
    #     model: "PreTrainedModel",
    #     param_value: "torch.Tensor",
    #     param_name: str,
    #     target_device: "torch.device",
    #     **kwargs,
    # ):
    #     module, tensor_name = get_module_from_name(model, param_name)
    #     module_name = param_name.rsplit(".", 1)[0]
    #     parent_module, node = get_module_from_name(model, module_name)

    #     quant_config = model.config.quantization_config["quant_config"]
    #     skip_modules = model.config.quantization_config["skip_modules"]

    #     # In this case we do not quantize this layer (it's explicitly skipped) -> simply load param
    #     if any(skip_module in module.name for skip_module in skip_modules):
    #         module.load_state_dict(
    #             {tensor_name: param_value.to(device=target_device, dtype=self.dtype)}, strict=False, assign=True
    #         )
    #         return

    #     # We need this hack as the model is not pre-prepared as an empty skeleton on meta device
    #     if self.pre_quantized:
    #         # Save them for later
    #         if not hasattr(self, "hqq_params"):
    #             self.hqq_params = defaultdict(dict)
    #         self.hqq_params[module_name].update({tensor_name: param_value})
    #         hqq_params = self.hqq_params[module_name]

    #         # If they are all present and saved, make it a HQQLinear layer! (we cannot do it param after param because
    #         # hqq does not support it...)
    #         if all(k in hqq_params for k in self.hqq_keys) and ("bias" in hqq_params or module.bias is None):
    #             hqq_layer = HQQLinear(
    #                 linear_layer=None,
    #                 quant_config=None,
    #                 compute_dtype=self.dtype,
    #                 device=target_device,
    #                 del_orig=False,
    #             )
    #             hqq_layer.load_state_dict(hqq_params)

    #             if hqq_layer.bias is not None and isinstance(hqq_layer.bias, torch.Tensor):
    #                 hqq_layer.bias = torch.nn.Parameter(hqq_layer.bias)
    #             if self.using_multi_gpu:
    #                 hqq_layer = self._patch_layer_for_multigpu(hqq_layer)

    #             setattr(parent_module, node, hqq_layer)
    #             del self.hqq_params[module_name], module
    #         return

    #     # Load param in the module (without caring about device or dtype, it will be changed later)
    #     module.load_state_dict({tensor_name: param_value}, strict=False, assign=True)

    #     # If both the weight and bias have already been loaded, time to quantize!
    #     module_is_ready = module.weight.device.type != "meta" and (
    #         module.bias is None or module.bias.device.type != "meta"
    #     )

    #     if module_is_ready:
    #         module_tag = ".".join(module.name.split(".")[-2:])
    #         if "weight_quant_params" in quant_config:
    #             module_quant_config = quant_config
    #         elif module_tag in quant_config:
    #             module_quant_config = quant_config[module_tag]

    #         hqq_layer = HQQLinear(
    #             module,
    #             quant_config=module_quant_config,
    #             compute_dtype=self.dtype,
    #             device=target_device,
    #             del_orig=True,
    #         )

    #         if hqq_layer.bias is not None and isinstance(hqq_layer.bias, torch.Tensor):
    #             hqq_layer.bias = torch.nn.Parameter(hqq_layer.bias)

    #         if self.using_multi_gpu:
    #             hqq_layer = self._patch_layer_for_multigpu(hqq_layer)

    #         setattr(parent_module, node, hqq_layer)

    def _patch_layer_for_multigpu(self, hqq_layer):
        def forward_with_device(self, x):
            out = torch.matmul(x.to(self.device), self.dequantize().t())
            if self.bias is not None:
                out += self.bias
            return out

        hqq_layer.forward = lambda x: forward_with_device(hqq_layer, x)
        return hqq_layer

    def _load_torch_checkpoint(self, checkpoint_file):
        try:
            return torch.load(checkpoint_file, map_location="cpu", weights_only=True)
        except Exception:
            return torch.load(checkpoint_file, map_location="cpu", weights_only=False)

    @staticmethod
    def _flatten_nested_state_dict(state_dict):
        if not any(isinstance(value, dict) for value in state_dict.values()):
            return state_dict

        flat_state_dict = {}
        for module_name, value in state_dict.items():
            if isinstance(value, dict):
                for attr_name, attr_value in value.items():
                    flat_state_dict[f"{module_name}.{attr_name}"] = attr_value
            else:
                flat_state_dict[module_name] = value
        return flat_state_dict

    def _normalize_nested_checkpoint_files(self, checkpoint_files):
        if not checkpoint_files:
            return checkpoint_files

        normalized_files = []
        for checkpoint_file in checkpoint_files:
            if checkpoint_file.endswith(".safetensors"):
                normalized_files.append(checkpoint_file)
                continue

            state_dict = self._load_torch_checkpoint(checkpoint_file)
            flat_state_dict = self._flatten_nested_state_dict(state_dict)
            if flat_state_dict is state_dict:
                normalized_files.append(checkpoint_file)
                continue

            temp_dir = tempfile.mkdtemp(prefix="transformers-hqq-")
            temp_file = os.path.join(temp_dir, os.path.basename(checkpoint_file))
            torch.save(flat_state_dict, temp_file)
            normalized_files.append(temp_file)

        return normalized_files

    def _setup_prequantized_key_filters(self, model, checkpoint_files):
        import re

        from safetensors import safe_open

        quantized_modules = set()
        for checkpoint_file in checkpoint_files or []:
            if checkpoint_file.endswith(".safetensors"):
                with safe_open(checkpoint_file, framework="pt") as f:
                    keys = f.keys()
                    quantized_modules.update(key[: -len(".W_q")] for key in keys if key.endswith(".W_q"))
            else:
                state_dict = self._flatten_nested_state_dict(self._load_torch_checkpoint(checkpoint_file))
                quantized_modules.update(key[: -len(".W_q")] for key in state_dict if key.endswith(".W_q"))

        if not quantized_modules:
            return

        hqq_state_dict_keys = HQQLinear(None, None).state_dict_keys()
        existing_unexpected = set(model._keys_to_ignore_on_load_unexpected or [])
        model._keys_to_ignore_on_load_unexpected = existing_unexpected | {rf"\.{key}$" for key in hqq_state_dict_keys}

        existing_missing = set(model._keys_to_ignore_on_load_missing or [])
        model._keys_to_ignore_on_load_missing = existing_missing | {
            re.escape(module_name) + r"\.weight" for module_name in quantized_modules
        }

        for module_name in quantized_modules:
            try:
                module = model.get_submodule(module_name)
            except AttributeError:
                continue
            if hasattr(module, "weight") and module.weight is not None:
                module.weight = torch.nn.Parameter(torch.empty(0, device="meta"), requires_grad=False)

    def _process_model_before_weight_loading(
        self,
        model: "PreTrainedModel",
        checkpoint_files=None,
        **kwargs,
    ):
        if self.pre_quantized:
            original_checkpoint_files = checkpoint_files
            normalized_checkpoint_files = self._normalize_nested_checkpoint_files(checkpoint_files)
            if isinstance(original_checkpoint_files, list):
                original_checkpoint_files[:] = normalized_checkpoint_files
            checkpoint_files = normalized_checkpoint_files
            self._checkpoint_files = checkpoint_files
            self._setup_prequantized_key_filters(model, checkpoint_files)
        else:
            # Add the corresponding quant_config to each valid module. This allows us to do the actual nn.Linear -> HQQLinear conversion in create_quantized_param().
            # prepare_for_hqq_linear() also sets the right quantization config inside the model (model.config.quantization_config) and the layers (hqq_layer.quant_config)
            model = prepare_for_hqq_linear(model, quantization_config=self.quantization_config)

    def _process_model_after_weight_loading(self, model: "PreTrainedModel", **kwargs):
        if self.pre_quantized:
            self._load_hqq_from_checkpoint(model)
        setattr(model, "is_hqq_quantized", True)
        setattr(model, "is_hqq_serializable", self.is_serializable())
        return model

    def _load_hqq_from_checkpoint(self, model: "PreTrainedModel"):
        from collections import defaultdict

        from safetensors import safe_open

        from ..integrations.hqq import autoname_modules, name_to_linear_tag

        device_map = getattr(self, "device_map", None)
        if isinstance(device_map, dict):
            devices = [torch.device(device) for device in device_map.values()]
            target_devices = [device for device in devices if device.type != "cpu"]
            target_device = target_devices[0] if target_devices else torch.device("cpu")
        elif isinstance(device_map, str) and device_map not in ("cpu", "auto"):
            target_device = torch.device(device_map)
        else:
            target_device = torch.device("cpu")

        autoname_modules(model)
        skip_modules = self.quantization_config.skip_modules
        hqq_state_dict_keys = HQQLinear(None, None).state_dict_keys()

        quantizable_modules = {}
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                linear_tag = name_to_linear_tag(name)
                if linear_tag not in skip_modules:
                    quantizable_modules[name] = module

        full_state_dict = {}
        for checkpoint_file in self._checkpoint_files:
            if checkpoint_file.endswith(".safetensors"):
                with safe_open(checkpoint_file, framework="pt") as f:
                    for key in f.keys():
                        full_state_dict[key] = f.get_tensor(key)
            else:
                full_state_dict.update(self._flatten_nested_state_dict(self._load_torch_checkpoint(checkpoint_file)))

        module_states = defaultdict(dict)
        for key, value in full_state_dict.items():
            for module_name in quantizable_modules:
                if key.startswith(module_name + "."):
                    param_name = key[len(module_name) + 1 :]
                    if param_name in hqq_state_dict_keys:
                        module_states[module_name][param_name] = value
                    break

        for module_name, state in module_states.items():
            if "W_q" not in state:
                continue

            hqq_layer = HQQLinear(None, None, compute_dtype=self.dtype or torch.float16, device="cpu", initialize=False)
            state["W_q"] = torch.nn.Parameter(state["W_q"], requires_grad=False)
            hqq_layer.load_state_dict(state)

            if target_device.type != "cpu":
                hqq_layer.cuda(target_device)

            if hqq_layer.bias is not None and isinstance(hqq_layer.bias, torch.Tensor):
                hqq_layer.bias = torch.nn.Parameter(hqq_layer.bias)
            if self.using_multi_gpu:
                hqq_layer = self._patch_layer_for_multigpu(hqq_layer)

            parent_name, _, child_name = module_name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child_name, hqq_layer)

        del full_state_dict

    def is_serializable(self):
        return True

    @property
    def is_trainable(self) -> bool:
        return True
