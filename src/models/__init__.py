"""Classical model architectures used as EXAQC baselines."""

from src.models.classical_image_classifier import (
    CLASSICAL_IMAGE_MODELS,
    CNNClassifier,
    LinearClassifier,
    MLPClassifier,
    create_classical_image_model,
)
from src.models.torchvision_image_classifier import (
    TORCHVISION_IMAGE_MODELS,
    create_torchvision_image_model,
)

__all__ = [
    "CLASSICAL_IMAGE_MODELS",
    "TORCHVISION_IMAGE_MODELS",
    "CNNClassifier",
    "LinearClassifier",
    "MLPClassifier",
    "create_classical_image_model",
    "create_torchvision_image_model",
]
