from __future__ import annotations

from collections.abc import Sequence
from typing import Callable

import torch
from torchvision import models

TORCHVISION_IMAGE_MODELS = (
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "densenet121",
    "densenet161",
    "densenet169",
    "densenet201",
    "efficientnet_b0",
    "efficientnet_b1",
    "efficientnet_b2",
    "efficientnet_b3",
    "efficientnet_v2_s",
    "mobilenet_v3_small",
    "mobilenet_v3_large",
    "convnext_tiny",
    "convnext_small",
    "convnext_base",
    "vgg11",
    "vgg13",
    "vgg16",
    "vgg19",
    "regnet_y_400mf",
    "regnet_y_800mf",
    "regnet_y_1_6gf",
)


def create_torchvision_image_model(
    model_name: str,
    input_shape: Sequence[int],
    n_classes: int,
    pretrained: bool = False,
    small_image_stem: bool = True,
) -> torch.nn.Module:
    """Creates a standard Torchvision image-classification model.

    Args:
        model_name: Name of a supported Torchvision model.
        input_shape: Per-sample image shape ``[channels, height, width]``.
        n_classes: Number of output classes.
        pretrained: Whether to initialize with Torchvision default pretrained
            weights.
        small_image_stem: Whether to adapt supported architectures for small
            images such as CIFAR-10.

    Returns:
        Configured PyTorch image-classification model.

    Raises:
        ValueError: If the model name or input shape is unsupported.
    """
    if len(input_shape) != 3:
        raise ValueError("input_shape must contain channels, height, and width.")

    input_channels = int(input_shape[0])

    model = _create_base_model(
        model_name=model_name,
        pretrained=pretrained,
    )

    model = _replace_classifier(
        model=model,
        model_name=model_name,
        n_classes=n_classes,
    )

    # Do not replace a pretrained RGB stem unless required because changing
    # its shape discards the pretrained first-layer weights.
    if input_channels != 3:
        model = _replace_input_channels(
            model=model,
            model_name=model_name,
            input_channels=input_channels,
        )

    if small_image_stem:
        model = _adapt_small_image_stem(
            model=model,
            model_name=model_name,
            input_channels=input_channels,
            pretrained=pretrained,
        )

    return model


def _weights(
    enum_class: type,
    pretrained: bool,
):
    """Returns default Torchvision weights when requested.

    Args:
        enum_class: Torchvision weight enum class.
        pretrained: Whether pretrained weights are requested.

    Returns:
        The default weights object or ``None``.
    """
    return enum_class.DEFAULT if pretrained else None


def _create_base_model(
    model_name: str,
    pretrained: bool,
) -> torch.nn.Module:
    """Creates a base Torchvision model.

    Args:
        model_name: Supported architecture name.
        pretrained: Whether to use default pretrained weights.

    Returns:
        Torchvision model.

    Raises:
        ValueError: If ``model_name`` is unsupported.
    """
    builders: dict[str, tuple[Callable, object]] = {
        "resnet18": (
            models.resnet18,
            _weights(models.ResNet18_Weights, pretrained),
        ),
        "resnet34": (
            models.resnet34,
            _weights(models.ResNet34_Weights, pretrained),
        ),
        "resnet50": (
            models.resnet50,
            _weights(models.ResNet50_Weights, pretrained),
        ),
        "resnet101": (
            models.resnet101,
            _weights(models.ResNet101_Weights, pretrained),
        ),
        "resnet152": (
            models.resnet152,
            _weights(models.ResNet152_Weights, pretrained),
        ),
        "densenet121": (
            models.densenet121,
            _weights(models.DenseNet121_Weights, pretrained),
        ),
        "densenet161": (
            models.densenet161,
            _weights(models.DenseNet161_Weights, pretrained),
        ),
        "densenet169": (
            models.densenet169,
            _weights(models.DenseNet169_Weights, pretrained),
        ),
        "densenet201": (
            models.densenet201,
            _weights(models.DenseNet201_Weights, pretrained),
        ),
        "efficientnet_b0": (
            models.efficientnet_b0,
            _weights(models.EfficientNet_B0_Weights, pretrained),
        ),
        "efficientnet_b1": (
            models.efficientnet_b1,
            _weights(models.EfficientNet_B1_Weights, pretrained),
        ),
        "efficientnet_b2": (
            models.efficientnet_b2,
            _weights(models.EfficientNet_B2_Weights, pretrained),
        ),
        "efficientnet_b3": (
            models.efficientnet_b3,
            _weights(models.EfficientNet_B3_Weights, pretrained),
        ),
        "efficientnet_v2_s": (
            models.efficientnet_v2_s,
            _weights(models.EfficientNet_V2_S_Weights, pretrained),
        ),
        "mobilenet_v3_small": (
            models.mobilenet_v3_small,
            _weights(models.MobileNet_V3_Small_Weights, pretrained),
        ),
        "mobilenet_v3_large": (
            models.mobilenet_v3_large,
            _weights(models.MobileNet_V3_Large_Weights, pretrained),
        ),
        "convnext_tiny": (
            models.convnext_tiny,
            _weights(models.ConvNeXt_Tiny_Weights, pretrained),
        ),
        "convnext_small": (
            models.convnext_small,
            _weights(models.ConvNeXt_Small_Weights, pretrained),
        ),
        "convnext_base": (
            models.convnext_base,
            _weights(models.ConvNeXt_Base_Weights, pretrained),
        ),
        "vgg11": (
            models.vgg11,
            _weights(models.VGG11_Weights, pretrained),
        ),
        "vgg13": (
            models.vgg13,
            _weights(models.VGG13_Weights, pretrained),
        ),
        "vgg16": (
            models.vgg16,
            _weights(models.VGG16_Weights, pretrained),
        ),
        "vgg19": (
            models.vgg19,
            _weights(models.VGG19_Weights, pretrained),
        ),
        "regnet_y_400mf": (
            models.regnet_y_400mf,
            _weights(models.RegNet_Y_400MF_Weights, pretrained),
        ),
        "regnet_y_800mf": (
            models.regnet_y_800mf,
            _weights(models.RegNet_Y_800MF_Weights, pretrained),
        ),
        "regnet_y_1_6gf": (
            models.regnet_y_1_6gf,
            _weights(models.RegNet_Y_1_6GF_Weights, pretrained),
        ),
    }

    try:
        builder, weights = builders[model_name]
    except KeyError as error:
        raise ValueError(f"Unsupported Torchvision model: {model_name}") from error

    return builder(weights=weights)


def _replace_classifier(
    model: torch.nn.Module,
    model_name: str,
    n_classes: int,
) -> torch.nn.Module:
    """Replaces the final classifier with an ``n_classes`` output layer.

    Args:
        model: Torchvision model.
        model_name: Architecture name.
        n_classes: Number of output classes.

    Returns:
        Updated model.

    Raises:
        ValueError: If the model family is unsupported.
    """
    if model_name.startswith("resnet"):
        model.fc = torch.nn.Linear(
            model.fc.in_features,
            n_classes,
        )

    elif model_name.startswith("densenet"):
        model.classifier = torch.nn.Linear(
            model.classifier.in_features,
            n_classes,
        )

    elif (
        model_name.startswith("efficientnet")
        or model_name.startswith("mobilenet")
        or model_name.startswith("convnext")
        or model_name.startswith("vgg")
        or model_name.startswith("regnet")
    ):
        final_layer = model.classifier[-1]
        model.classifier[-1] = torch.nn.Linear(
            final_layer.in_features,
            n_classes,
        )

    else:
        raise ValueError(f"Classifier replacement not implemented for {model_name}.")

    return model


def _replace_input_channels(
    model: torch.nn.Module,
    model_name: str,
    input_channels: int,
) -> torch.nn.Module:
    """Changes the first convolution for non-RGB inputs.

    Args:
        model: Torchvision model.
        model_name: Architecture name.
        input_channels: Number of desired input channels.

    Returns:
        Updated model.

    Raises:
        ValueError: If input-channel adaptation is not implemented.
    """
    if model_name.startswith("resnet"):
        old_layer = model.conv1
        model.conv1 = _copy_conv_with_input_channels(
            old_layer,
            input_channels,
        )

    elif model_name.startswith("densenet"):
        old_layer = model.features.conv0
        model.features.conv0 = _copy_conv_with_input_channels(
            old_layer,
            input_channels,
        )

    elif model_name.startswith("efficientnet"):
        old_layer = model.features[0][0]
        model.features[0][0] = _copy_conv_with_input_channels(
            old_layer,
            input_channels,
        )

    elif model_name.startswith("mobilenet"):
        old_layer = model.features[0][0]
        model.features[0][0] = _copy_conv_with_input_channels(
            old_layer,
            input_channels,
        )

    elif model_name.startswith("convnext"):
        old_layer = model.features[0][0]
        model.features[0][0] = _copy_conv_with_input_channels(
            old_layer,
            input_channels,
        )

    elif model_name.startswith("vgg"):
        old_layer = model.features[0]
        model.features[0] = _copy_conv_with_input_channels(
            old_layer,
            input_channels,
        )

    elif model_name.startswith("regnet"):
        old_layer = model.stem[0]
        model.stem[0] = _copy_conv_with_input_channels(
            old_layer,
            input_channels,
        )

    else:
        raise ValueError(f"Input-channel adaptation not implemented for {model_name}.")

    return model


def _copy_conv_with_input_channels(
    layer: torch.nn.Conv2d,
    input_channels: int,
) -> torch.nn.Conv2d:
    """Creates a convolution matching ``layer`` with new input channels.

    Args:
        layer: Existing convolution.
        input_channels: Number of desired input channels.

    Returns:
        Replacement convolution.
    """
    return torch.nn.Conv2d(
        in_channels=input_channels,
        out_channels=layer.out_channels,
        kernel_size=layer.kernel_size,
        stride=layer.stride,
        padding=layer.padding,
        dilation=layer.dilation,
        groups=layer.groups if layer.groups == 1 else 1,
        bias=layer.bias is not None,
        padding_mode=layer.padding_mode,
    )


def _adapt_small_image_stem(
    model: torch.nn.Module,
    model_name: str,
    input_channels: int,
    pretrained: bool,
) -> torch.nn.Module:
    """Adapts selected architectures for small images such as CIFAR-10.

    The ResNet ImageNet stem uses a 7x7 stride-2 convolution followed by
    max-pooling. For 32x32 images this aggressively reduces spatial
    resolution, so scratch-trained ResNets can use a 3x3 stride-1 stem and
    remove the initial max-pool.

    Pretrained ResNet models retain the ImageNet stem because replacing it
    would discard the pretrained stem weights.

    Args:
        model: Torchvision model.
        model_name: Architecture name.
        input_channels: Number of input image channels.
        pretrained: Whether the model uses pretrained weights.

    Returns:
        Adapted model.
    """
    if model_name.startswith("resnet") and not pretrained:
        model.conv1 = torch.nn.Conv2d(
            input_channels,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        model.maxpool = torch.nn.Identity()

    return model
