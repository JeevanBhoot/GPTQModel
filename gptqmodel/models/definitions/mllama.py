# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from types import SimpleNamespace
from typing import Any, Dict, List

import torch
from transformers import AutoModelForPreTraining, AutoProcessor

from ...utils.model import get_module, move_to, nested_move_to
from ...utils.offload import offload_to_disk
from .._const import CPU
from ..base import BaseQModel, QuantizationRegion


class MLlamaQModel(BaseQModel):
    # AutoModelForPreTraining return a correct MLlamaForConditionalGeneration for mllama.
    loader = AutoModelForPreTraining

    pre_lm_head_norm_module = "model.language_model.norm"
    layer_modules_strict = False
    # Current Transformers shells use `model.language_model.*`, while the
    # released Mllama checkpoints store those tensors under `language_model.model.*`.
    HF_CONVERSION_MAP_REVERSED = (
        SimpleNamespace(
            source_patterns=[r"model\.language_model"],
            target_patterns=[r"^language_model.model"],
            operations=[],
        ),
        SimpleNamespace(
            source_patterns=[r"lm_head"],
            target_patterns=[r"^language_model.lm_head"],
            operations=[],
        ),
        SimpleNamespace(
            source_patterns=[r"model\.vision_model"],
            target_patterns=[r"^vision_model"],
            operations=[],
        ),
        SimpleNamespace(
            source_patterns=[r"model\.multi_modal_projector"],
            target_patterns=[r"^multi_modal_projector"],
            operations=[],
        ),
    )

    module_tree = [
        [
            "model",
            "language_model",
            "layers",
            "#",
            {
                "input_layernorm": ("input_layernorm:!",),
                "self_attn": ("q_proj:0", "k_proj:0", "v_proj:0", "o_proj:1"),
                "post_attention_layernorm": ("post_attention_layernorm:!",),
                "mlp": ("gate_proj:0", "up_proj:0", "down_proj:1"),
            },
        ],
        [
            "language_model",
            "model",
            "layers",
            "#",
            {
                "input_layernorm": ("input_layernorm:!",),
                "self_attn": ("q_proj:0", "k_proj:0", "v_proj:0", "o_proj:1"),
                "post_attention_layernorm": ("post_attention_layernorm:!",),
                "mlp": ("gate_proj:0", "up_proj:0", "down_proj:1"),
            },
        ],
    ]

    _VISION_LOCAL_TREE = [
        "model",
        "vision_model",
        "transformer",
        "layers",
        "#",
        {
            "self_attn": ("q_proj:0", "k_proj:0", "v_proj:0", "o_proj:1"),
            "attention": ("q_proj:0", "k_proj:0", "v_proj:0", "o_proj:1"),
            "mlp": ("fc1:0", "fc2:1", "gate_proj:0", "up_proj:0", "down_proj:1"),
        },
    ]

    _VISION_GLOBAL_TREE = [
        "model",
        "vision_model",
        "global_transformer",
        "layers",
        "#",
        {
            "self_attn": ("q_proj:0", "k_proj:0", "v_proj:0", "o_proj:1"),
            "attention": ("q_proj:0", "k_proj:0", "v_proj:0", "o_proj:1"),
            "mlp": ("fc1:0", "fc2:1", "gate_proj:0", "up_proj:0", "down_proj:1"),
        },
    ]

    _PROJECTOR_TREE = [
        "model",
        "multi_modal_projector",
        "#",
        {
            "": ("",),
        },
    ]

    _TEXT_TREE = [
        "model",
        "language_model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("q_proj:0", "k_proj:0", "v_proj:0", "o_proj:1"),
            "cross_attn": ("q_proj:0", "k_proj:0", "v_proj:0", "o_proj:1"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp": ("gate_proj:0", "up_proj:0", "down_proj:1"),
        },
    ]

    _MULTIMODAL_REQUIRED_KEYS = {
        "input_ids",
        "attention_mask",
        "pixel_values",
        "aspect_ratio_ids",
        "aspect_ratio_mask",
        "cross_attention_mask",
    }
    _MULTIMODAL_TENSOR_KEYS = _MULTIMODAL_REQUIRED_KEYS.difference({"input_ids", "attention_mask"})

    @classmethod
    def _resolve_language_model(cls, model):
        for prefix in ("model.language_model", "language_model.model"):
            language_model = get_module(model, prefix)
            if language_model is not None:
                return language_model

        raise AttributeError("Unable to resolve an Mllama language model layout.")

    def _core_language_model(self):
        return self._resolve_language_model(self.model)

    @classmethod
    def _is_multimodal_example(cls, example: Dict[str, Any]) -> bool:
        return any(
            key in example
            for key in (
                "pixel_values",
                "aspect_ratio_ids",
                "aspect_ratio_mask",
                "cross_attention_mask",
                "image",
                "images",
            )
        )

    @classmethod
    def _has_multimodal_calibration(cls, processors) -> bool:
        for processor in processors or []:
            dataset = getattr(processor, "calibration_dataset", None)
            if not dataset:
                continue
            for example in dataset:
                if isinstance(example, dict) and cls._is_multimodal_example(example):
                    return True
        return False

    def quantization_regions(self, processors=None) -> List[QuantizationRegion]:
        if not self._has_multimodal_calibration(processors):
            return super().quantization_regions(processors=processors)

        return [
            QuantizationRegion(
                name="mllama_vision_local",
                layers_node="model.vision_model.transformer.layers",
                module_tree=self._VISION_LOCAL_TREE,
            ),
            QuantizationRegion(
                name="mllama_vision_global",
                layers_node="model.vision_model.global_transformer.layers",
                module_tree=self._VISION_GLOBAL_TREE,
            ),
            QuantizationRegion(
                name="mllama_multi_modal_projector",
                layers_node="model.multi_modal_projector",
                module_tree=self._PROJECTOR_TREE,
                layer_modules=[[""]],
                planning_layer_modules=[[""]],
            ),
            QuantizationRegion(
                name="mllama_text",
                layers_node="model.language_model.layers",
                module_tree=self._TEXT_TREE,
                include_lm_head=bool(getattr(self.quantize_config, "lm_head", False)),
            ),
        ]

    def input_capture_base_modules(self, region: QuantizationRegion = None) -> List[str]:
        if region is not None and str(getattr(region, "name", "")).startswith("mllama_"):
            return self.quantization_region_base_modules(region)

        multimodal_roots = (
            "model.vision_model",
            "model.multi_modal_projector",
            "vision_model",
            "multi_modal_projector",
        )
        return [
            module_name
            for module_name in super().input_capture_base_modules(region)
            if not any(self._paths_overlap(module_name, root) for root in multimodal_roots)
        ]

    def _materialize_language_module(self, language_model, attr_name: str):
        module = getattr(language_model, attr_name, None)
        if module is None:
            return

        if "_turtle_lock" not in self.__dict__ and "shell_module_materialize" not in self.__dict__:
            setattr(language_model, attr_name, move_to(module, device=self.quantize_config.device))
            return

        setattr(
            language_model,
            attr_name,
            self.shell_module_materialize(module, self.quantize_config.device),
        )

    @staticmethod
    def _resolve_parent_and_module(root, path: str):
        parent = root
        parts = path.split(".")
        for part in parts[:-1]:
            parent = getattr(parent, part, None)
            if parent is None:
                return None, None, None
        leaf = parts[-1]
        module = getattr(parent, leaf, None)
        return parent, leaf, module

    def _iter_multimodal_module_refs(self):
        seen = set()
        for path in (
            "model.vision_model",
            "model.multi_modal_projector",
            "vision_model",
            "multi_modal_projector",
        ):
            parent, leaf, module = self._resolve_parent_and_module(self.model, path)
            if module is None or id(module) in seen:
                continue
            seen.add(id(module))
            yield path, parent, leaf, module

    def _is_full_multimodal_region_active(self) -> bool:
        region = getattr(self, "_active_quantization_region", None)
        return bool(
            region is not None
            and str(getattr(region, "name", "")).startswith("mllama_")
        )

    def _is_active_region_path(self, path: str) -> bool:
        region = getattr(self, "_active_quantization_region", None)
        if region is None:
            return False
        return any(
            self._paths_overlap(path, active_node)
            for active_node in self._region_layer_nodes(region)
        )

    def pre_quantize_generate_hook_start(self):
        language_model = self._core_language_model()
        self._materialize_language_module(language_model, "embed_tokens")
        self._materialize_language_module(language_model, "rotary_emb")

        if self._is_full_multimodal_region_active():
            for _path, parent, leaf, module in self._iter_multimodal_module_refs():
                if "_turtle_lock" not in self.__dict__ and "shell_module_materialize" not in self.__dict__:
                    setattr(parent, leaf, move_to(module, device=self.quantize_config.device))
                else:
                    setattr(
                        parent,
                        leaf,
                        self.shell_module_materialize(module, self.quantize_config.device),
                    )

    def pre_quantize_generate_hook_end(self):
        language_model = self._core_language_model()

        for attr_name in ("embed_tokens", "rotary_emb"):
            module = getattr(language_model, attr_name, None)
            if module is None:
                continue

            if self.quantize_config.offload_to_disk:
                offload_to_disk(
                    model=language_model,
                    module=module,
                    disk_path=self.quantize_config.offload_to_disk_path,
                )
            else:
                setattr(language_model, attr_name, move_to(module, device=CPU))

        if self._is_full_multimodal_region_active():
            for path, parent, leaf, module in self._iter_multimodal_module_refs():
                if self._is_active_region_path(path):
                    continue
                if self.quantize_config.offload_to_disk:
                    continue
                else:
                    setattr(parent, leaf, move_to(module, device=CPU))

    def should_skip_layer_stage(self, *, module, layer_name: str, region) -> bool:
        del layer_name
        is_cross_attention_layer = module.__class__.__name__.lower() == "mllamacrossattentiondecoderlayer"
        if not is_cross_attention_layer:
            return False
        return not (
            region is not None
            and getattr(region, "name", None) == "mllama_text"
        )

    @staticmethod
    def _prepare_first_layer_attention_mask(attention_mask):
        if attention_mask is None or not torch.is_tensor(attention_mask):
            return attention_mask

        if attention_mask.ndim <= 2 and bool(attention_mask.to(dtype=torch.bool).all().item()):
            return None

        return attention_mask

    def run_input_capture(self, example, use_cache: bool, data_device):
        if self._is_multimodal_example(example):
            return super().run_input_capture(example, use_cache=use_cache, data_device=data_device)

        input_ids = example.get("input_ids")
        if input_ids is None:
            return super().run_input_capture(example, use_cache=use_cache, data_device=data_device)

        language_model = self._core_language_model()
        attention_mask = example.get("attention_mask")
        position_ids = example.get("position_ids")
        past_key_values = example.get("past_key_values")

        embedding_weight = getattr(language_model.embed_tokens, "weight", None)
        if torch.is_tensor(embedding_weight) and input_ids.device != embedding_weight.device:
            input_ids = input_ids.to(device=embedding_weight.device)

        # Input capture only needs the tensors entering the first decoder layer.
        # Calling it directly avoids the multimodal wrapper's mask construction
        # path, which can inspect meta tensors during lazy quantization.
        inputs_embeds = language_model.embed_tokens(input_ids)
        if getattr(inputs_embeds, "is_meta", False):
            raise RuntimeError("Mllama input capture produced meta inputs_embeds after materializing embed_tokens.")

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)
        elif position_ids.device != inputs_embeds.device:
            position_ids = position_ids.to(device=inputs_embeds.device)

        position_embeddings = language_model.rotary_emb(inputs_embeds, position_ids=position_ids)
        first_layer_attention_mask = self._prepare_first_layer_attention_mask(attention_mask)

        return language_model.layers[0](
            inputs_embeds,
            attention_mask=first_layer_attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
        )

    def capture_first_layer_positional_inputs(
        self,
        args: tuple[Any, ...],
        kwargs: Dict[str, Any],
        batch_device: torch.device,
    ) -> List[torch.Tensor]:
        if kwargs.get("hidden_state") is not None:
            return [move_to(kwargs["hidden_state"], device=batch_device)]
        return super().capture_first_layer_positional_inputs(
            args=args,
            kwargs=kwargs,
            batch_device=batch_device,
        )

    def capture_first_layer_input_kwargs(
        self,
        args: tuple[Any, ...],
        kwargs: Dict[str, Any],
        batch_device: torch.device,
        layer_input_kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        layer_input_kwargs = super().capture_first_layer_input_kwargs(
            args=args,
            kwargs=kwargs,
            batch_device=batch_device,
            layer_input_kwargs=layer_input_kwargs,
        )
        for key in (
            "cross_attention_states",
            "cross_attention_mask",
            "full_text_row_masked_out_mask",
            "position_embeddings",
            "cache_position",
        ):
            if key in kwargs:
                layer_input_kwargs[key] = nested_move_to(kwargs[key], device=batch_device)
        return layer_input_kwargs

    @staticmethod
    def _text_axis_for(value, seq_len: int) -> int:
        if value.ndim >= 4:
            return value.ndim - 2
        if value.ndim == 3:
            if value.shape[1] >= seq_len:
                return 1
            return 2
        return 1

    @staticmethod
    def _batch_gather_text_axis(value, positions, axis: int):
        view_shape = [positions.shape[0]] + [1] * (value.ndim - 1)
        view_shape[axis] = positions.shape[1]
        expand_shape = list(value.shape)
        expand_shape[axis] = positions.shape[1]
        gather_index = positions.reshape(view_shape).expand(expand_shape)
        return value.gather(axis, gather_index)

    @classmethod
    def _slice_text_axis(cls, value, hidden_states, position_ids=None, cache_position=None):
        if not torch.is_tensor(value) or not torch.is_tensor(hidden_states) or hidden_states.ndim < 2:
            return value

        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]
        if seq_len <= 0:
            return value

        axis = cls._text_axis_for(value, seq_len)
        axis_size = value.shape[axis]

        if torch.is_tensor(cache_position) and cache_position.numel() > 0:
            try:
                positions = cache_position.reshape(-1).to(device=value.device, dtype=torch.long)
                if (
                    positions.numel() == seq_len
                    and int(positions.min().item()) >= 0
                    and int(positions.max().item()) < axis_size
                ):
                    return value.index_select(axis, positions)
            except Exception:
                pass

        if axis_size == seq_len:
            return value

        if torch.is_tensor(position_ids) and position_ids.numel() > 0:
            try:
                positions = position_ids.to(device=value.device, dtype=torch.long)
                if positions.ndim == 1:
                    positions = positions.unsqueeze(0)
                if (
                    positions.ndim == 2
                    and positions.shape[0] == batch_size
                    and positions.shape[1] == seq_len
                    and int(positions.min().item()) >= 0
                    and int(positions.max().item()) < axis_size
                    and value.shape[0] == batch_size
                ):
                    return cls._batch_gather_text_axis(value, positions, axis)
            except Exception:
                pass

        if value.ndim >= 4 and value.shape[-2] != seq_len and value.shape[-2] >= seq_len:
            return value[..., :seq_len, :]
        if value.ndim == 3 and value.shape[axis] != seq_len and value.shape[axis] >= seq_len:
            return value.narrow(axis, 0, seq_len)
        if value.ndim >= 2 and value.shape[1] != seq_len and value.shape[1] >= seq_len:
            return value[:, :seq_len]
        return value

    def prepare_layer_replay_kwargs(self, layer, layer_input, additional_inputs, target_device):
        additional_inputs = super().prepare_layer_replay_kwargs(
            layer=layer,
            layer_input=layer_input,
            additional_inputs=additional_inputs,
            target_device=target_device,
        )
        region = getattr(self, "_active_quantization_region", None)
        region_name = getattr(region, "name", None)
        if region_name in {"mllama_vision_local", "mllama_vision_global"}:
            attention_mask = additional_inputs.get("attention_mask")
            if attention_mask is None:
                return {}
            return {"attention_mask": nested_move_to(attention_mask, device=target_device)}
        if region_name == "mllama_multi_modal_projector":
            return {}

        hidden_states = layer_input[0] if layer_input else None
        position_ids = additional_inputs.get("position_ids")
        cache_position = additional_inputs.get("cache_position")

        for key in ("cross_attention_mask", "full_text_row_masked_out_mask"):
            if key in additional_inputs:
                additional_inputs[key] = self._slice_text_axis(
                    additional_inputs[key],
                    hidden_states,
                    position_ids=position_ids,
                    cache_position=cache_position,
                )

        for key in ("cross_attention_states", "cross_attention_mask", "full_text_row_masked_out_mask"):
            if key in additional_inputs:
                additional_inputs[key] = nested_move_to(additional_inputs[key], device=target_device)

        return additional_inputs

    @staticmethod
    def _as_list_dataset(calibration_dataset) -> List[Any]:
        if isinstance(calibration_dataset, list):
            return calibration_dataset
        if hasattr(calibration_dataset, "to_list"):
            return calibration_dataset.to_list()
        return list(calibration_dataset)

    @staticmethod
    def _batch_tensor_examples(examples: List[Dict[str, Any]], batch_size: int) -> List[Dict[str, Any]]:
        batches = []
        for start in range(0, len(examples), batch_size):
            rows = examples[start:start + batch_size]
            batch: Dict[str, Any] = {}
            for key in rows[0].keys():
                values = [row[key] for row in rows if key in row]
                if not values:
                    continue
                if len(values) != len(rows):
                    raise ValueError(
                        "Mllama multimodal calibration tensor batch has inconsistent "
                        f"optional key `{key}`; tensorized rows must use identical keys."
                    )
                if all(torch.is_tensor(value) for value in values):
                    normalized = [
                        value.unsqueeze(0) if value.ndim == 1 else value
                        for value in values
                    ]
                    try:
                        batch[key] = torch.cat(normalized, dim=0)
                    except RuntimeError as exc:
                        shapes = [tuple(value.shape) for value in normalized]
                        raise ValueError(
                            "Mllama multimodal calibration tensor batch is ragged for "
                            f"`{key}` with shapes {shapes}. Pass raw image/text samples "
                            "through the processor path or pre-pad this tensor key."
                        ) from exc
                    if batch[key].shape[0] != len(rows):
                        raise ValueError(
                            "Mllama multimodal calibration tensor batch has incorrect "
                            f"batch size for `{key}`: expected {len(rows)}, got {batch[key].shape[0]}."
                        )
                else:
                    batch[key] = values
            batches.append(batch)
        return batches

    def _get_processor(self):
        processor = getattr(self, "processor", None)
        if processor is not None:
            return processor
        model_path = getattr(self, "model_local_path", None)
        if model_path is None:
            raise ValueError("Mllama multimodal calibration requires an AutoProcessor or model_local_path.")
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=self.require_trust_remote_code)
        self.processor = processor
        return processor

    @staticmethod
    def _sample_text(sample: Dict[str, Any], processor) -> Any:
        for key in ("text", "prompt", "content"):
            if key in sample:
                return sample[key]
        messages = sample.get("messages") or sample.get("conversation") or sample.get("conversations")
        if messages is not None and hasattr(processor, "apply_chat_template"):
            return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return messages

    @staticmethod
    def _sample_images(sample: Dict[str, Any]) -> Any:
        if "images" in sample:
            return sample["images"]
        if "image" in sample:
            return sample["image"]
        return None

    def prepare_dataset(
        self,
        calibration_dataset,
        calibration_dataset_concat_size=None,
        calibration_dataset_sort=None,
        batch_size: int = 1,
        calibration_data_min_length: int = 10,
        calibration_concat_separator=None,
    ):
        rows = self._as_list_dataset(calibration_dataset)
        if not rows:
            return []

        has_multimodal = any(isinstance(row, dict) and self._is_multimodal_example(row) for row in rows)
        if not has_multimodal:
            return super().prepare_dataset(
                calibration_dataset=calibration_dataset,
                calibration_dataset_concat_size=calibration_dataset_concat_size,
                calibration_dataset_sort=calibration_dataset_sort,
                batch_size=batch_size,
                calibration_data_min_length=calibration_data_min_length,
                calibration_concat_separator=calibration_concat_separator,
            )

        if calibration_dataset_concat_size is not None:
            raise ValueError(
                "Mllama multimodal calibration cannot use text concatenation; "
                "rebuild image/text masks with the processor."
            )

        tensorized_rows = [
            index
            for index, row in enumerate(rows)
            if isinstance(row, dict) and any(key in row for key in self._MULTIMODAL_TENSOR_KEYS)
        ]
        if tensorized_rows:
            missing_by_row = {
                index: sorted(self._MULTIMODAL_REQUIRED_KEYS.difference(rows[index].keys()))
                for index in tensorized_rows
                if not self._MULTIMODAL_REQUIRED_KEYS.issubset(rows[index].keys())
            }
            if missing_by_row:
                details = "; ".join(
                    f"row {index}: {', '.join(missing)}"
                    for index, missing in missing_by_row.items()
                )
                raise ValueError(
                    "Mllama multimodal calibration rows with preprocessed vision tensors "
                    "must include all required tensor keys; missing "
                    + details
                )
            if len(tensorized_rows) != len(rows):
                raise ValueError(
                    "Mllama multimodal calibration cannot mix preprocessed tensor rows "
                    "with raw image/text rows in the same dataset."
                )
            reference_keys = set(rows[tensorized_rows[0]].keys())
            inconsistent_key_rows = []
            for index in tensorized_rows[1:]:
                row_keys = set(rows[index].keys())
                if row_keys != reference_keys:
                    missing = sorted(reference_keys.difference(row_keys))
                    extra = sorted(row_keys.difference(reference_keys))
                    detail = f"row {index}"
                    if missing:
                        detail += f" missing {', '.join(missing)}"
                    if extra:
                        detail += f" extra {', '.join(extra)}"
                    inconsistent_key_rows.append(detail)
            if inconsistent_key_rows:
                raise ValueError(
                    "Mllama multimodal calibration tensorized rows must use identical keys; "
                    + "; ".join(inconsistent_key_rows)
                )

        tensor_ready = all(
            isinstance(row, dict) and self._MULTIMODAL_REQUIRED_KEYS.issubset(row.keys())
            for row in rows
        )
        if tensor_ready:
            return self._batch_tensor_examples(rows, batch_size=batch_size)

        processor = self._get_processor()
        prepared = []
        for start in range(0, len(rows), batch_size):
            samples = rows[start:start + batch_size]
            images = [self._sample_images(sample) for sample in samples]
            texts = [self._sample_text(sample, processor) for sample in samples]
            encoded = processor(
                images=images,
                text=texts,
                return_tensors="pt",
                padding=True,
            )
            encoded = dict(encoded)
            missing = sorted(self._MULTIMODAL_REQUIRED_KEYS.difference(encoded.keys()))
            if missing:
                raise ValueError(
                    "Mllama multimodal calibration processor output is missing required keys: "
                    + ", ".join(missing)
                )
            prepared.append(encoded)

        return prepared
