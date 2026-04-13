import torch

from CLIP.tokenizer import tokenize
from prompt import (
    GENERIC_ABNORMAL_STATES,
    GENERIC_NORMAL_STATES,
    GENERIC_SUBJECTS,
    MEDICAL_PROMPT_TEMPLATES,
    PROMPT_METADATA,
)


def _deduplicate(items):
    return list(dict.fromkeys(items))


def _build_prompt_sentences(obj, abnormal):
    metadata = PROMPT_METADATA.get(obj, {})
    subjects = metadata.get("subjects", GENERIC_SUBJECTS)
    state_patterns = metadata.get(
        "abnormal" if abnormal else "normal",
        GENERIC_ABNORMAL_STATES if abnormal else GENERIC_NORMAL_STATES,
    )

    descriptive_phrases = []
    for subject in subjects:
        for state_pattern in state_patterns:
            descriptive_phrases.append(state_pattern.format(subject))

    prompted_sentences = []
    for phrase in _deduplicate(descriptive_phrases):
        prompted_sentences.append(phrase)
        for template in MEDICAL_PROMPT_TEMPLATES:
            prompted_sentences.append(template.format(phrase))

    return _deduplicate(prompted_sentences)


def encode_text_with_prompt_ensemble(model, obj, device):
    text_features = []
    for abnormal in [False, True]:
        prompted_sentences = tokenize(_build_prompt_sentences(obj, abnormal)).to(device)
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
