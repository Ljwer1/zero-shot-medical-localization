MEDICAL_PROMPT_TEMPLATES = [
    "{}.",
    "a medical image of {}.",
    "a clinical image of {}.",
    "a diagnostic image showing {}.",
    "an image showing {}.",
    "the image shows {}.",
    "this is {}.",
    "a detailed medical image of {}.",
]


PROMPT_METADATA = {
    "Brain": {
        "subjects": [
            "a brain MRI scan",
            "a brain MRI image",
            "a brain FLAIR MRI scan",
        ],
        "normal": [
            "{} with normal brain anatomy",
            "{} without visible intracranial lesion",
            "{} with preserved brain structure",
            "{} with no focal brain abnormality",
            "{} with no abnormal signal change",
        ],
        "abnormal": [
            "{} with a visible brain lesion",
            "{} with abnormal intracranial finding",
            "{} with abnormal signal intensity",
            "{} with focal brain abnormality",
            "{} with structural distortion in brain tissue",
        ],
    },
    "Liver": {
        "subjects": [
            "a liver CT scan",
            "a liver CT image",
            "an abdominal CT image of the liver",
        ],
        "normal": [
            "{} with normal liver parenchyma",
            "{} without focal hepatic lesion",
            "{} with homogeneous liver appearance",
            "{} with preserved liver contour",
            "{} with no visible liver abnormality",
        ],
        "abnormal": [
            "{} with a focal liver lesion",
            "{} with abnormal liver parenchyma",
            "{} with hepatic structural irregularity",
            "{} with suspicious hepatic finding",
            "{} with visible liver abnormality",
        ],
    },
    "Retina_RESC": {
        "subjects": [
            "a retinal OCT image",
            "a retinal OCT scan",
            "an optical coherence tomography image of the retina",
        ],
        "normal": [
            "{} with normal retinal layers",
            "{} without retinal fluid or lesion",
            "{} with preserved retinal structure",
            "{} with regular retinal morphology",
            "{} with no visible retinal abnormality",
        ],
        "abnormal": [
            "{} with abnormal retinal layers",
            "{} with retinal fluid or lesion",
            "{} with structural irregularity in the retina",
            "{} with visible retinal abnormality",
            "{} with distorted retinal morphology",
        ],
    },
}


GENERIC_SUBJECTS = [
    "a medical image",
    "a diagnostic image",
    "a clinical image",
]


GENERIC_NORMAL_STATES = [
    "{} with normal appearance",
    "{} without visible abnormality",
    "{} with no abnormal finding",
    "{} with preserved structure",
    "{} showing no focal lesion",
]


GENERIC_ABNORMAL_STATES = [
    "{} with abnormal appearance",
    "{} with visible lesion",
    "{} with suspicious abnormal finding",
    "{} with structural irregularity",
    "{} with focal abnormality",
]
