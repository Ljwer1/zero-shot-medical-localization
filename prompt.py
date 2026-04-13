PROMPT_MODES = ("upstream",)


PROMPT_NORMAL = [
    "{}",
    "flawless {}",
    "perfect {}",
    "unblemished {}",
    "{} without flaw",
    "{} without defect",
    "{} without damage",
]


PROMPT_ABNORMAL = [
    "damaged {}",
    "broken {}",
    "{} with flaw",
    "{} with defect",
    "{} with damage",
]


PROMPT_STATE = [PROMPT_NORMAL, PROMPT_ABNORMAL]


PROMPT_TEMPLATES = [
    "a bad photo of a {}.",
    "a low resolution photo of the {}.",
    "a bad photo of the {}.",
    "a cropped photo of the {}.",
    "a bright photo of a {}.",
    "a dark photo of the {}.",
    "a photo of my {}.",
    "a photo of the cool {}.",
    "a close-up photo of a {}.",
    "a black and white photo of the {}.",
    "a bright photo of the {}.",
    "a cropped photo of a {}.",
    "a jpeg corrupted photo of a {}.",
    "a blurry photo of the {}.",
    "a photo of the {}.",
    "a good photo of the {}.",
    "a photo of one {}.",
    "a close-up photo of the {}.",
    "a photo of a {}.",
    "a low resolution photo of a {}.",
    "a photo of a large {}.",
    "a blurry photo of a {}.",
    "a jpeg corrupted photo of the {}.",
    "a good photo of a {}.",
    "a photo of the small {}.",
    "a photo of the large {}.",
    "a black and white photo of a {}.",
    "a dark photo of a {}.",
    "a photo of a cool {}.",
    "a photo of a small {}.",
    "there is a {} in the scene.",
    "there is the {} in the scene.",
    "this is a {} in the scene.",
    "this is the {} in the scene.",
    "this is one {} in the scene.",
]


REAL_NAME = {
    "Brain": "Brain",
    "Liver": "Liver",
    "Retina_RESC": "retinal OCT",
    "Chest": "Chest X-ray film",
    "Retina_OCT2017": "retinal OCT",
    "Histopathology": "histopathological image",
}
