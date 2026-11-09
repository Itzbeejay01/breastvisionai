"""Shared semantic mapping for binary model outputs.

Training uses ImageDataGenerator.flow_from_directory with benign/malignant
directories. Directory names are sorted alphabetically, and each training
script's classification report names output column 0 Benign and column 1
Malignant. The saved stacking estimator has classes_ [0, 1].
"""

CLASS_LABELS = {0: "benign", 1: "malignant"}
LABEL_TO_INDEX = {name: index for index, name in CLASS_LABELS.items()}
MALIGNANT_LABEL = "malignant"
MALIGNANT_INDEX = LABEL_TO_INDEX[MALIGNANT_LABEL]


def semantic_class(value):
    """Resolve numeric estimator labels or semantic strings to class names."""
    if isinstance(value, str) and value.lower() in LABEL_TO_INDEX:
        return value.lower()
    try:
        return CLASS_LABELS[int(value)]
    except (ValueError, TypeError, KeyError):
        raise ValueError(f"Unsupported binary class label: {value!r}")


def malignant_column(estimator):
    """Find the malignant probability column from the fitted estimator labels."""
    classes = list(estimator.classes_)
    for index, label in enumerate(classes):
        if semantic_class(label) == MALIGNANT_LABEL:
            return index
    raise ValueError(f"Estimator classes do not contain {MALIGNANT_LABEL!r}: {classes!r}")


def prediction_from_malignancy_probability(probability):
    """Convert a malignant probability to the one shared binary label config."""
    malignant = float(probability) >= 0.5
    return MALIGNANT_LABEL if malignant else CLASS_LABELS[0]
