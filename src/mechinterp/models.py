"""Loading small pretrained models into Penzai for JAX analysis."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from penzai import pz
from penzai.models.transformer import model_parts, variants

DEFAULT_MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
PYTHIA_MODEL_ID = "EleutherAI/pythia-70m-deduped"


@dataclass(frozen=True)
class LoadedModel:
    model: model_parts.TransformerLM
    tokenizer: object
    model_id: str
    num_layers: int


def load_pythia(model_id: str = PYTHIA_MODEL_ID) -> LoadedModel:
    """Download GPT-NeoX/Pythia weights and convert to a pure JAX model.

    PyTorch is used *only* to read the checkpoint via Hugging Face. Forward
    passes, gradients, interventions, and analysis all use JAX/Penzai.
    """
    from transformers import AutoTokenizer, GPTNeoXForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    hf_model = GPTNeoXForCausalLM.from_pretrained(model_id)
    hf_model.eval()
    num_layers = int(hf_model.config.num_hidden_layers)

    # Penzai 0.2.5 rejects checkpoint-path metadata during config validation.
    # This does not affect architecture or weights.
    hf_model.config._name_or_path = ""

    model = variants.gpt_neox.gpt_neox_from_huggingface_model(
        hf_model, use_layer_stack=False
    )
    del hf_model
    return LoadedModel(
        model=pz.freeze_params(model),
        tokenizer=tokenizer,
        model_id=model_id,
        num_layers=num_layers,
    )


def convert_llama(hf_model) -> model_parts.TransformerLM:
    """Convert a frozen Llama-style model, including SmolLM2's tied weights.

    Penzai 0.2.5's Llama converter stores input/output embeddings separately.
    For inference this is equivalent to tied embeddings with identical values.
    Keep architecture validation enabled and normalize only known metadata.
    """
    import torch
    from transformers import LlamaConfig

    original_config = hf_model.config
    config = deepcopy(original_config)
    reference = LlamaConfig()
    config._name_or_path = reference._name_or_path
    # Nonstandard checkpoint flags: require their supported meanings.
    for name, supported in (("is_llama_config", True), ("rope_interleaved", False)):
        if hasattr(config, name):
            if getattr(config, name) != supported:
                raise ValueError(f"Unsupported Llama configuration: {name}")
            delattr(config, name)

    if config.tie_word_embeddings:
        input_weight = hf_model.get_input_embeddings().weight
        output_weight = hf_model.get_output_embeddings().weight
        if not torch.equal(input_weight, output_weight):
            raise ValueError("Expected identical tied input and output embeddings")
        # The converter reads both weights from state_dict. Do not retie,
        # reinitialize or otherwise modify the parameters.
        config.tie_word_embeddings = False

    hf_model.config = config
    hf_model.config.initializer_range = 0.02
    try:
        model = variants.llama.llama_from_huggingface_model(
            hf_model, use_layer_stack=False,
            upcast_activations_to_float32=True,
        )
    finally:
        hf_model.config = original_config
    return pz.freeze_params(model)


def load_smollm2(model_id: str = DEFAULT_MODEL_ID) -> LoadedModel:
    """Load the base SmolLM2 model and convert it to JAX/Penzai in float32."""
    import torch
    from transformers import AutoTokenizer, LlamaForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    hf_model = LlamaForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32, attn_implementation="eager"
    )
    hf_model.eval()
    num_layers = int(hf_model.config.num_hidden_layers)
    model = convert_llama(hf_model)
    del hf_model
    return LoadedModel(model, tokenizer, model_id, num_layers)


def single_token_id(tokenizer: object, text: str) -> int:
    """Require one next-token candidate to avoid comparing partial words."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(
            f"{text!r} tokenizes to {len(ids)} tokens ({ids}); "
            "choose single-token target/foil strings."
        )
    return int(ids[0])
