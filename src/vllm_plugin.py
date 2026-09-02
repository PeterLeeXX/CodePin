"""Small vLLM compatibility registrations for CodePin checkpoints."""

from dataclasses import replace


def register_qwen35_text() -> None:
    """Register vLLM's text-only Qwen3.5 implementation.

    vLLM 0.23.0 ships the implementation but omits it from its built-in model
    registry, so a checkpoint whose architecture is Qwen3_5ForCausalLM is
    incorrectly routed to the multimodal conditional-generation class.
    """

    import torch
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        QwenGatedDeltaNetAttention,
    )
    from vllm.model_executor.models.qwen3_5 import (
        MambaStateCopyFuncCalculator,
        MambaStateDtypeCalculator,
        MambaStateShapeCalculator,
        Qwen3_5ForCausalLM,
    )
    from vllm.model_executor.models.registry import ModelRegistry
    from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper

    original_get_kv_cache_spec = QwenGatedDeltaNetAttention.get_kv_cache_spec

    def get_padded_kv_cache_spec(self, vllm_config):
        spec = original_get_kv_cache_spec(self, vllm_config)
        if spec is None or spec.page_size_padded is not None:
            return spec

        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        bytes_per_token = (
            model_config.get_num_kv_heads(vllm_config.parallel_config)
            * model_config.get_head_size()
            * torch.tensor([], dtype=torch.float32).element_size()
        )
        page_unit = cache_config.block_size * bytes_per_token
        padded_page_size = (
            (spec.page_size_bytes + page_unit - 1) // page_unit
        ) * page_unit
        return replace(spec, page_size_padded=padded_page_size)

    QwenGatedDeltaNetAttention.get_kv_cache_spec = get_padded_kv_cache_spec

    class CodePinQwen35ForCausalLM(Qwen3_5ForCausalLM):
        # The HF artifact is the text submodel saved from the full Qwen3.5
        # wrapper, so its language weights carry one extra prefix.
        hf_to_vllm_mapper = WeightsMapper(
            orig_to_new_prefix={"model.language_model.": "model."}
        )

        def load_weights(self, weights):
            loader = AutoWeightsLoader(self, skip_prefixes=["mtp."])
            return loader.load_weights(self.hf_to_vllm_mapper.apply(weights))

        @classmethod
        def get_mamba_state_dtype_from_config(cls, vllm_config):
            return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
                vllm_config.model_config.dtype,
                vllm_config.cache_config.mamba_cache_dtype,
                vllm_config.cache_config.mamba_ssm_cache_dtype,
            )

        @classmethod
        def get_mamba_state_shape_from_config(cls, vllm_config):
            config = vllm_config.model_config.hf_text_config
            tp_size = vllm_config.parallel_config.tensor_parallel_size
            num_spec = (
                vllm_config.speculative_config.num_speculative_tokens
                if vllm_config.speculative_config
                else 0
            )
            return MambaStateShapeCalculator.gated_delta_net_state_shape(
                tp_size,
                config.linear_num_key_heads,
                config.linear_num_value_heads,
                config.linear_key_head_dim,
                config.linear_value_head_dim,
                config.linear_conv_kernel_dim,
                num_spec,
            )

        @classmethod
        def get_mamba_state_copy_func(cls):
            return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()

    ModelRegistry.register_model("Qwen3_5ForCausalLM", CodePinQwen35ForCausalLM)
