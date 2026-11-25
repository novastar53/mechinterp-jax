"""Loading small pretrained models into Penzai for JAX analysis."""

from __future__ import annotations

from dataclasses import dataclass

from penzai import pz
from penzai.models.transformer import model_parts, variants

DEFAULT_MODEL_ID = "EleutherAI/pythia-70m-deduped"


@dataclass(frozen=True)
class LoadedModel:
    model: model_parts.TransformerLM
    tokenizer: object
    model_id: str
    num_layers: int


def load_pythia(model_id: str = DEFAULT_MODEL_ID) -> LoadedModel:
    """Download GPT-NeoX/Pythia weights and convert to a pure JAX model.

    PyTorch is used *only* to read the checkpoint via Hugging Face. Forward
    passes, gradients, interventions, and analysis all use JAX/Penzai.
    """
    from transformers import AutoTokenizer, GPTNeoXForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    hf_model = GPTNeoXForCausalLM.from_pretrained(model_id)
    hf_model.eval()
    num_layers = int(hf_model.config.num_hidden_layers)

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


def single_token_id(tokenizer: object, text: str) -> int:
    """Require one next-token candidate to avoid comparing partial words."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(
            f"{text!r} tokenizes to {len(ids)} tokens ({ids}); "
            "choose single-token target/foil strings."
        )
    return int(ids[0])
