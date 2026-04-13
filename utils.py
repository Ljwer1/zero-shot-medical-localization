import torch

from CLIP.tokenizer import tokenize
from prompt import (
    PROMPT_MODES,
    PROMPT_STATE,
    PROMPT_TEMPLATES,
    REAL_NAME,
)


def _deduplicate(items):
    return list(dict.fromkeys(items))


def _resolve_prompt_object_name(obj):
    return REAL_NAME.get(obj, obj)


def resolve_prompt_mode(prompt_mode, checkpoint=None):
    if prompt_mode == "auto":
        if isinstance(checkpoint, dict):
            checkpoint_prompt_mode = checkpoint.get("prompt_mode")
            if checkpoint_prompt_mode in PROMPT_MODES:
                return checkpoint_prompt_mode
        return "upstream"

    if prompt_mode not in PROMPT_MODES:
        raise ValueError(f"Unsupported prompt_mode: {prompt_mode}")
    return prompt_mode


def encode_text_with_prompt_ensemble(model, obj, device, prompt_mode="upstream"):
    prompt_mode = resolve_prompt_mode(prompt_mode)
    if prompt_mode != "upstream":
        raise ValueError(f"Unsupported resolved prompt_mode: {prompt_mode}")

    obj = _resolve_prompt_object_name(obj)
    text_features = []
    for prompt_state in PROMPT_STATE:
        prompted_state = [state.format(obj) for state in prompt_state]
        prompted_sentences = []
        for state in prompted_state:
            for template in PROMPT_TEMPLATES:
                prompted_sentences.append(template.format(state))

        prompted_sentences = tokenize(_deduplicate(prompted_sentences)).to(device)
        class_embeddings = model.encode_text(prompted_sentences)
        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
        class_embedding = class_embeddings.mean(dim=0)
        class_embedding /= class_embedding.norm()
        text_features.append(class_embedding)
    return torch.stack(text_features, dim=1).to(device)


def fuse_layer_outputs(layer_outputs, layer_weights):
    stacked = torch.stack(layer_outputs, dim=0)
    view_shape = (layer_weights.shape[0],) + (1,) * (stacked.dim() - 1)
    return (stacked * layer_weights.view(view_shape)).sum(dim=0)


def resolve_fusion_weights(features, learned_weights, fusion_mode="learned", selected_layer=None):
    if fusion_mode == "learned":
        return learned_weights

    if fusion_mode == "equal":
        return torch.ones_like(learned_weights)

    if fusion_mode == "single":
        if selected_layer is None:
            raise ValueError("selected_layer must be provided when fusion_mode='single'")
        if selected_layer not in features:
            raise ValueError(f"selected_layer should be one of {features}, got {selected_layer}")

        single_weights = torch.zeros_like(learned_weights)
        single_weights[features.index(selected_layer)] = 1.0
        return single_weights

    raise ValueError(f"Unsupported fusion_mode: {fusion_mode}")
