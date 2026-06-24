# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from types import SimpleNamespace
import inspect

import pytest
import torch
from torch import nn

import gptqmodel.models.definitions.mllama as mllama_def
from gptqmodel.looper.stage_layer import run_layer_stage
from gptqmodel.models.definitions.mllama import MLlamaQModel
from gptqmodel.utils.model import get_layers_with_prefixes
from gptqmodel.utils.structure import LazyTurtle


class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 4, bias=False)
        self.k_proj = nn.Linear(4, 4, bias=False)
        self.v_proj = nn.Linear(4, 4, bias=False)
        self.o_proj = nn.Linear(4, 4, bias=False)


class _MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(4, 4, bias=False)
        self.up_proj = nn.Linear(4, 4, bias=False)
        self.down_proj = nn.Linear(4, 4, bias=False)


class _DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(4)
        self.self_attn = _Attention()
        self.post_attention_layernorm = nn.LayerNorm(4)
        self.mlp = _MLP()
        self.last_kwargs = None

    def forward(self, hidden_states, **kwargs):
        self.last_kwargs = kwargs
        return hidden_states


class _Rotary(nn.Module):
    def forward(self, x, position_ids=None):
        shape = (*x.shape[:2], x.shape[-1] // 2)
        return x.new_zeros(shape), x.new_zeros(shape)


class _RecordingEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(8, 4, device="meta"))
        self.last_input_device = None

    def forward(self, input_ids):
        self.last_input_device = input_ids.device
        return torch.zeros((*input_ids.shape, 4))


class _LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(8, 4)
        self.layers = nn.ModuleList([_DecoderLayer(), _DecoderLayer()])
        self.norm = nn.LayerNorm(4)
        self.rotary_emb = _Rotary()


class _CurrentMllamaWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = _LanguageModel()
        self.model.vision_model = nn.Identity()
        self.model.multi_modal_projector = nn.Linear(4, 4, bias=False)
        self.lm_head = nn.Linear(4, 4, bias=False)


class _LegacyMllamaWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = nn.Module()
        self.language_model.model = _LanguageModel()
        self.lm_head = nn.Linear(4, 4, bias=False)


def test_mllama_module_tree_supports_current_transformers_layout():
    model = _CurrentMllamaWrapper()

    layers, layer_names = get_layers_with_prefixes(
        model,
        MLlamaQModel.extract_layers_node(),
    )

    assert MLlamaQModel.extract_layers_node()[0] == "model.language_model.layers"
    assert len(layers) == 2
    assert layer_names == [
        "model.language_model.layers.0",
        "model.language_model.layers.1",
    ]
    assert MLlamaQModel.pre_lm_head_norm_module == "model.language_model.norm"
    base_modules = set(MLlamaQModel.get_base_modules(model))
    assert "model.vision_model" in base_modules
    assert "model.multi_modal_projector" in base_modules


def test_mllama_module_tree_keeps_legacy_language_model_layout():
    model = _LegacyMllamaWrapper()

    layers, layer_names = get_layers_with_prefixes(
        model,
        MLlamaQModel.extract_layers_node(),
    )

    assert len(layers) == 2
    assert layer_names == [
        "language_model.model.layers.0",
        "language_model.model.layers.1",
    ]


def test_mllama_pre_quantize_hooks_materialize_text_input_modules():
    instance = object.__new__(MLlamaQModel)
    nn.Module.__init__(instance)
    instance.model = _CurrentMllamaWrapper()
    instance.quantize_config = SimpleNamespace(
        device="cpu",
        offload_to_disk=False,
        offload_to_disk_path="/tmp/unused",
    )

    instance.pre_quantize_generate_hook_start()
    instance.pre_quantize_generate_hook_end()

    language_model = instance.model.model.language_model
    assert language_model.embed_tokens.weight.device.type == "cpu"
    assert next(language_model.rotary_emb.parameters(), None) is None


def test_mllama_full_multimodal_hooks_materialize_and_restore_vision_modules(monkeypatch):
    instance = object.__new__(MLlamaQModel)
    nn.Module.__init__(instance)
    instance.model = _CurrentMllamaWrapper()
    instance.quantize_config = SimpleNamespace(
        device="cpu",
        offload_to_disk=False,
        offload_to_disk_path="/tmp/unused",
    )
    instance._active_quantization_region = SimpleNamespace(name="mllama_text")

    materialized = []
    moved = []

    def fake_materialize(module, device, **kwargs):
        del kwargs
        materialized.append((module, device))
        return module

    def fake_move_to(module, device, **kwargs):
        del kwargs
        moved.append((module, device))
        return module

    instance.shell_module_materialize = fake_materialize
    monkeypatch.setattr(mllama_def, "move_to", fake_move_to)

    instance.pre_quantize_generate_hook_start()
    instance.pre_quantize_generate_hook_end()

    language_model = instance.model.model.language_model
    expected_materialized = {
        id(language_model.embed_tokens),
        id(language_model.rotary_emb),
        id(instance.model.model.vision_model),
        id(instance.model.model.multi_modal_projector),
    }
    assert {id(module) for module, _device in materialized} == expected_materialized
    assert id(instance.model.model.vision_model) in {id(module) for module, _device in moved}
    assert id(instance.model.model.multi_modal_projector) in {id(module) for module, _device in moved}


def test_mllama_full_multimodal_hooks_keep_active_vision_region_materialized(monkeypatch):
    instance = object.__new__(MLlamaQModel)
    nn.Module.__init__(instance)
    instance.model = _CurrentMllamaWrapper()
    instance.quantize_config = SimpleNamespace(
        device="cpu",
        offload_to_disk=False,
        offload_to_disk_path="/tmp/unused",
    )
    instance._active_quantization_region = SimpleNamespace(
        name="mllama_vision_local",
        layers_node="model.vision_model.transformer.layers",
    )

    moved = []

    def fake_materialize(module, device, **kwargs):
        del kwargs
        return module

    def fake_move_to(module, device, **kwargs):
        del kwargs
        moved.append((module, device))
        return module

    instance.shell_module_materialize = fake_materialize
    monkeypatch.setattr(mllama_def, "move_to", fake_move_to)

    instance.pre_quantize_generate_hook_start()
    instance.pre_quantize_generate_hook_end()

    moved_ids = {id(module) for module, _device in moved}
    assert id(instance.model.model.vision_model) not in moved_ids
    assert id(instance.model.model.multi_modal_projector) in moved_ids


def test_mllama_text_only_hooks_do_not_materialize_vision_modules():
    instance = object.__new__(MLlamaQModel)
    nn.Module.__init__(instance)
    instance.model = _CurrentMllamaWrapper()
    instance.quantize_config = SimpleNamespace(
        device="cpu",
        offload_to_disk=False,
        offload_to_disk_path="/tmp/unused",
    )
    instance._active_quantization_region = SimpleNamespace(name="default")

    materialized = []

    def fake_materialize(module, device, **kwargs):
        del kwargs
        materialized.append((module, device))
        return module

    instance.shell_module_materialize = fake_materialize

    instance.pre_quantize_generate_hook_start()

    language_model = instance.model.model.language_model
    assert {id(module) for module, _device in materialized} == {
        id(language_model.embed_tokens),
        id(language_model.rotary_emb),
    }


def test_mllama_run_input_capture_calls_first_decoder_layer_directly():
    instance = object.__new__(MLlamaQModel)
    nn.Module.__init__(instance)
    instance.model = _CurrentMllamaWrapper()
    instance.quantize_config = SimpleNamespace(
        device="cpu",
        offload_to_disk=False,
        offload_to_disk_path="/tmp/unused",
    )

    example = {
        "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
        "attention_mask": torch.ones((1, 3), dtype=torch.bool),
    }

    instance.pre_quantize_generate_hook_start()
    instance.run_input_capture(example, use_cache=False, data_device=torch.device("cpu"))

    first_layer = instance.model.model.language_model.layers[0]
    assert first_layer.last_kwargs["attention_mask"] is None
    assert first_layer.last_kwargs["position_ids"].shape == (1, 3)
    assert first_layer.last_kwargs["position_embeddings"][0].shape == (1, 3, 2)


def test_mllama_lazy_turtle_conversion_map_matches_checkpoint_prefixes():
    renamings = LazyTurtle._normalize_runtime_to_checkpoint_renamings(
        MLlamaQModel.resolve_hf_conversion_map_reversed(target_model=_CurrentMllamaWrapper())
    )

    def convert(path):
        out = [path]
        for renaming in renamings:
            renamed, _ = renaming.rename_source_key(path)
            out.append(renamed)
        return out

    assert "language_model.model.embed_tokens" in convert("model.language_model.embed_tokens")
    assert "language_model.model.layers.0" in convert("model.language_model.layers.0")
    assert "language_model.lm_head" in convert("lm_head")
    assert "vision_model" in convert("model.vision_model")
    assert "multi_modal_projector" in convert("model.multi_modal_projector")


def test_mllama_run_input_capture_moves_input_ids_to_embedding_device():
    instance = object.__new__(MLlamaQModel)
    nn.Module.__init__(instance)
    instance.model = _CurrentMllamaWrapper()
    instance.quantize_config = SimpleNamespace(
        device="meta",
        offload_to_disk=False,
        offload_to_disk_path="/tmp/unused",
    )

    embedding = _RecordingEmbedding()
    language_model = instance.model.model.language_model
    language_model.embed_tokens = embedding

    example = {
        "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
        "attention_mask": torch.ones((1, 3), dtype=torch.bool),
    }

    instance.run_input_capture(example, use_cache=False, data_device=torch.device("cpu"))

    assert embedding.last_input_device == torch.device("meta")


def test_mllama_capture_first_layer_positional_inputs_accepts_vision_hidden_state_kwarg():
    instance = object.__new__(MLlamaQModel)
    nn.Module.__init__(instance)

    hidden_state = torch.ones((1, 2, 4))
    layer_input = instance.capture_first_layer_positional_inputs(
        args=(),
        kwargs={"hidden_state": hidden_state},
        batch_device=torch.device("cpu"),
    )

    assert len(layer_input) == 1
    assert layer_input[0] is hidden_state


def _multimodal_calibration_row(seq_len=3, vision_tokens=5):
    return {
        "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
        "attention_mask": torch.ones((1, seq_len), dtype=torch.bool),
        "pixel_values": torch.zeros((1, 1, 3, 4, 4)),
        "aspect_ratio_ids": torch.zeros((1, 1), dtype=torch.long),
        "aspect_ratio_mask": torch.ones((1, 1), dtype=torch.bool),
        "cross_attention_mask": torch.ones((1, 1, seq_len, vision_tokens), dtype=torch.bool),
    }


def test_mllama_full_quantization_regions_are_multimodal_first():
    instance = object.__new__(MLlamaQModel)
    nn.Module.__init__(instance)
    instance.quantize_config = SimpleNamespace(lm_head=False)

    processor = SimpleNamespace(calibration_dataset=[_multimodal_calibration_row()])

    regions = instance.quantization_regions(processors=[processor])

    assert [region.name for region in regions] == [
        "mllama_vision_local",
        "mllama_vision_global",
        "mllama_multi_modal_projector",
        "mllama_text",
    ]
    assert [region.layers_node for region in regions] == [
        "model.vision_model.transformer.layers",
        "model.vision_model.global_transformer.layers",
        "model.multi_modal_projector",
        "model.language_model.layers",
    ]


def test_mllama_text_only_calibration_keeps_decoder_only_region():
    instance = object.__new__(MLlamaQModel)
    nn.Module.__init__(instance)
    instance.quantize_config = SimpleNamespace(lm_head=False)

    processor = SimpleNamespace(
        calibration_dataset=[
            {
                "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
                "attention_mask": torch.ones((1, 3), dtype=torch.bool),
            }
        ]
    )

    regions = instance.quantization_regions(processors=[processor])

    assert [region.name for region in regions] == ["default"]
    assert regions[0].layers_node == MLlamaQModel.extract_layers_node()


def test_mllama_region_module_mappings_cover_vision_projector_self_and_cross_attention():
    instance = object.__new__(MLlamaQModel)
    nn.Module.__init__(instance)
    instance.quantize_config = SimpleNamespace(lm_head=False, dynamic=None)

    regions = instance.quantization_regions(
        processors=[SimpleNamespace(calibration_dataset=[_multimodal_calibration_row()])]
    )
    modules_by_region = {
        region.name: MLlamaQModel.simple_layer_modules_for_region(
            region,
            model_config=SimpleNamespace(),
            quantize_config=SimpleNamespace(dynamic=None),
        )
        for region in regions
    }

    assert ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"] in modules_by_region["mllama_vision_local"]
    assert [""] in modules_by_region["mllama_multi_modal_projector"]
    assert ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"] in modules_by_region["mllama_text"]
    assert ["cross_attn.q_proj", "cross_attn.k_proj", "cross_attn.v_proj"] in modules_by_region["mllama_text"]
    assert ["mlp.gate_proj", "mlp.up_proj"] in modules_by_region["mllama_text"]
    selected_modules = [
        item
        for blocks in modules_by_region.values()
        for block in blocks
        for item in block
    ]
    assert "mlp.fc1" in selected_modules
    assert "mlp.fc2" in selected_modules
    assert all("patch_embedding" not in item for item in selected_modules)
    assert all("embed_tokens" not in item for item in selected_modules)


def test_mllama_region_base_modules_exclude_active_region_roots():
    instance = object.__new__(MLlamaQModel)
    nn.Module.__init__(instance)
    instance.model = _CurrentMllamaWrapper()
    instance.quantize_config = SimpleNamespace(lm_head=False)

    regions = {
        region.name: region
        for region in instance.quantization_regions(
            processors=[SimpleNamespace(calibration_dataset=[_multimodal_calibration_row()])]
        )
    }

    vision_base_modules = instance.quantization_region_base_modules(regions["mllama_vision_local"])
    projector_base_modules = instance.quantization_region_base_modules(regions["mllama_multi_modal_projector"])
    text_base_modules = instance.quantization_region_base_modules(regions["mllama_text"])

    assert "model.vision_model" not in vision_base_modules
    assert "model.multi_modal_projector" in vision_base_modules
    assert "model.vision_model" in projector_base_modules
    assert "model.multi_modal_projector" not in projector_base_modules
    assert "model.vision_model" in text_base_modules
    assert "model.multi_modal_projector" in text_base_modules


def test_mllama_text_only_input_capture_base_modules_exclude_multimodal_roots():
    instance = object.__new__(MLlamaQModel)
    nn.Module.__init__(instance)
    instance.model = _CurrentMllamaWrapper()
    instance.quantize_config = SimpleNamespace(lm_head=False)

    default_capture_modules = instance.input_capture_base_modules(
        SimpleNamespace(name="default", layers_node=MLlamaQModel.extract_layers_node())
    )

    assert "model.vision_model" in MLlamaQModel.get_base_modules(instance.model)
    assert "model.multi_modal_projector" in MLlamaQModel.get_base_modules(instance.model)
    assert "model.vision_model" not in default_capture_modules
    assert "model.multi_modal_projector" not in default_capture_modules

    regions = {
        region.name: region
        for region in instance.quantization_regions(
            processors=[SimpleNamespace(calibration_dataset=[_multimodal_calibration_row()])]
        )
    }
    multimodal_text_capture_modules = instance.input_capture_base_modules(regions["mllama_text"])
    assert "model.vision_model" in multimodal_text_capture_modules
    assert "model.multi_modal_projector" in multimodal_text_capture_modules


def test_mllama_cross_attention_layers_are_not_hard_skipped():
    assert "MllamaCrossAttentionDecoderLayer" not in inspect.getsource(run_layer_stage)


def test_mllama_skips_cross_attention_layers_only_for_text_only_region():
    instance = object.__new__(MLlamaQModel)
    nn.Module.__init__(instance)

    CrossLayer = type("MllamaCrossAttentionDecoderLayer", (nn.Module,), {})
    cross_layer = CrossLayer()
    self_layer = nn.Linear(4, 4)

    assert instance.should_skip_quantization_layer(
        module=cross_layer,
        layer_name="model.language_model.layers.1",
        region=SimpleNamespace(name="default", layers_node="model.language_model.layers"),
    )
    assert not instance.should_skip_quantization_layer(
        module=cross_layer,
        layer_name="model.language_model.layers.1",
        region=SimpleNamespace(name="mllama_text", layers_node="model.language_model.layers"),
    )
    assert not instance.should_skip_quantization_layer(
        module=self_layer,
        layer_name="model.language_model.layers.0",
        region=SimpleNamespace(name="default", layers_node="model.language_model.layers"),
    )


def test_mllama_replay_kwargs_preserve_and_slice_cross_attention_side_channel_tensors():
    instance = object.__new__(MLlamaQModel)
    nn.Module.__init__(instance)

    hidden_states = torch.zeros((1, 3, 4))
    cross_attention_mask = torch.arange(35, dtype=torch.float32).reshape(1, 1, 5, 7)
    row_mask = torch.arange(5, dtype=torch.float32).reshape(1, 1, 5, 1)
    cross_attention_states = torch.ones((1, 7, 4))
    additional_inputs = {
        "position_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
        "cross_attention_states": cross_attention_states,
        "cross_attention_mask": cross_attention_mask,
        "full_text_row_masked_out_mask": row_mask,
    }

    refreshed = instance.prepare_layer_replay_kwargs(
        layer=nn.Module(),
        layer_input=[hidden_states],
        additional_inputs=additional_inputs,
        target_device=torch.device("cpu"),
    )

    assert refreshed["cross_attention_states"] is cross_attention_states
    assert refreshed["cross_attention_mask"].shape == (1, 1, 3, 7)
    expected_cross_mask = cross_attention_mask.index_select(-2, torch.tensor([1, 2, 3]))
    assert torch.equal(refreshed["cross_attention_mask"], expected_cross_mask)
    assert refreshed["full_text_row_masked_out_mask"].shape == (1, 1, 3, 1)
    assert torch.equal(refreshed["full_text_row_masked_out_mask"], row_mask.index_select(-2, torch.tensor([1, 2, 3])))


def test_mllama_vision_replay_kwargs_drop_decoder_only_arguments():
    instance = object.__new__(MLlamaQModel)
    nn.Module.__init__(instance)
    instance._active_quantization_region = SimpleNamespace(name="mllama_vision_local")

    attention_mask = torch.ones((1, 1, 2, 2))
    refreshed = instance.prepare_layer_replay_kwargs(
        layer=nn.Module(),
        layer_input=[torch.zeros((1, 2, 4))],
        additional_inputs={
            "attention_mask": attention_mask,
            "position_ids": torch.arange(2).unsqueeze(0),
            "use_cache": False,
        },
        target_device=torch.device("cpu"),
    )

    assert list(refreshed.keys()) == ["attention_mask"]
    assert refreshed["attention_mask"] is attention_mask


def test_mllama_projector_replay_kwargs_drop_all_generic_layer_arguments():
    instance = object.__new__(MLlamaQModel)
    nn.Module.__init__(instance)
    instance._active_quantization_region = SimpleNamespace(name="mllama_multi_modal_projector")

    refreshed = instance.prepare_layer_replay_kwargs(
        layer=nn.Linear(4, 4),
        layer_input=[torch.zeros((1, 2, 4))],
        additional_inputs={
            "attention_mask": None,
            "use_cache": False,
        },
        target_device=torch.device("cpu"),
    )

    assert refreshed == {}


def test_mllama_replay_kwargs_batch_gather_cross_attention_masks():
    instance = object.__new__(MLlamaQModel)
    nn.Module.__init__(instance)

    hidden_states = torch.zeros((2, 3, 4))
    cross_attention_mask = torch.arange(2 * 1 * 5 * 7, dtype=torch.float32).reshape(2, 1, 5, 7)
    row_mask = torch.arange(2 * 1 * 5 * 1, dtype=torch.float32).reshape(2, 1, 5, 1)
    positions = torch.tensor([[1, 3, 4], [0, 2, 4]], dtype=torch.long)

    refreshed = instance.prepare_layer_replay_kwargs(
        layer=nn.Module(),
        layer_input=[hidden_states],
        additional_inputs={
            "position_ids": positions,
            "cross_attention_mask": cross_attention_mask,
            "full_text_row_masked_out_mask": row_mask,
        },
        target_device=torch.device("cpu"),
    )

    expected_cross_mask = torch.stack(
        [
            cross_attention_mask[0:1].index_select(-2, positions[0]),
            cross_attention_mask[1:2].index_select(-2, positions[1]),
        ],
        dim=0,
    ).squeeze(1)
    expected_row_mask = torch.stack(
        [
            row_mask[0:1].index_select(-2, positions[0]),
            row_mask[1:2].index_select(-2, positions[1]),
        ],
        dim=0,
    ).squeeze(1)

    assert refreshed["cross_attention_mask"].shape == (2, 1, 3, 7)
    assert torch.equal(refreshed["cross_attention_mask"], expected_cross_mask)
    assert refreshed["full_text_row_masked_out_mask"].shape == (2, 1, 3, 1)
    assert torch.equal(refreshed["full_text_row_masked_out_mask"], expected_row_mask)


def test_mllama_multimodal_prepare_dataset_rejects_text_concatenation():
    instance = object.__new__(MLlamaQModel)
    nn.Module.__init__(instance)

    with pytest.raises(ValueError, match="cannot use text concatenation"):
        instance.prepare_dataset(
            [_multimodal_calibration_row()],
            calibration_dataset_concat_size=128,
            batch_size=1,
        )


def test_mllama_multimodal_prepare_dataset_batches_processor_outputs():
    instance = object.__new__(MLlamaQModel)
    nn.Module.__init__(instance)

    rows = [_multimodal_calibration_row(), _multimodal_calibration_row()]
    batches = instance.prepare_dataset(rows, batch_size=2)

    assert len(batches) == 1
    assert batches[0]["input_ids"].shape == (2, 3)
    assert batches[0]["pixel_values"].shape[0] == 2
    assert batches[0]["cross_attention_mask"].shape[:3] == (2, 1, 3)


def test_mllama_multimodal_prepare_dataset_rejects_partial_tensorized_rows():
    instance = object.__new__(MLlamaQModel)
    nn.Module.__init__(instance)

    row = _multimodal_calibration_row()
    del row["cross_attention_mask"]

    with pytest.raises(ValueError, match="missing.*cross_attention_mask"):
        instance.prepare_dataset([row], batch_size=1)


def test_mllama_multimodal_prepare_dataset_rejects_inconsistent_optional_tensor_keys():
    instance = object.__new__(MLlamaQModel)
    nn.Module.__init__(instance)

    rows = [_multimodal_calibration_row(), _multimodal_calibration_row()]
    rows[1]["labels"] = torch.tensor([[1, 2, 3]], dtype=torch.long)

    with pytest.raises(ValueError, match="identical keys.*labels"):
        instance.prepare_dataset(rows, batch_size=2)


def test_mllama_multimodal_prepare_dataset_rejects_ragged_tensor_batches():
    instance = object.__new__(MLlamaQModel)
    nn.Module.__init__(instance)

    rows = [_multimodal_calibration_row(), _multimodal_calibration_row(seq_len=4)]

    with pytest.raises(ValueError, match="ragged.*attention_mask"):
        instance.prepare_dataset(rows, batch_size=2)
