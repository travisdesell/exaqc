from __future__ import annotations

import copy
import torch

from abc import ABC, abstractmethod
from loguru import logger

from typing import TYPE_CHECKING, Mapping, Sequence, Any

if TYPE_CHECKING:
    from src.circuits.circuit import CircuitGenome

ENCODING_OPTIONS = ["identity", "linear", "cnn"]


def initialize_encoder(
    target: str,
    encoding_str: str,
    n_inputs: int,
    n_outputs: int,
    config: Mapping[str, Any] | None = None,
) -> Encoder:
    """
    Given the target system (e.g., pennylane or qiskit) create a
    new encoder to perform encodings from classical to quantum
    which will set the input parameter and input gate for each wire.

    Args:
        target: the target system (pennylane or qiskit)
        encoding_str: a string representation of the encoder
            to be created
        n_inputs: how many classical input features will be used.
        n_outputs: how many values in the output of the encoder, which
            will be used as inputs for the quantum circuit.
        config: Optional encoder-specific configuration. The CNN encoder
            requires ``input_channels``, ``input_height``, and ``input_width``.

    Returns:
            A configured encoder instance.

    Raises:
        ValueError: If the requested encoder is unknown or its configuration is
            invalid.
    """

    logger.info(
        f"creating {encoding_str} encoder with n_inputs: {n_inputs} and n_outputs: {n_outputs}"
    )

    encoder = None
    encoder_config = dict(config or {})

    if encoding_str == "linear":
        encoder = LinearEncoder(n_inputs, n_outputs)

    elif encoding_str == "identity":
        encoder = IdentityEncoder(n_inputs, n_outputs)

    elif encoding_str == "cnn":
        required = {
            "input_channels",
            "input_height",
            "input_width",
        }
        missing = required.difference(encoder_config)
        if missing:
            raise ValueError(
                "CNN encoder configuration is missing required fields: "
                f"{sorted(missing)}"
            )

        return CNNEncoder(
            n_inputs=n_inputs,
            n_outputs=n_outputs,
            input_channels=int(encoder_config["input_channels"]),
            input_height=int(encoder_config["input_height"]),
            input_width=int(encoder_config["input_width"]),
            conv_blocks=encoder_config["conv_blocks"],
            adaptive_pool_size=tuple(
                int(value)
                for value in encoder_config.get(
                    "adaptive_pool_size",
                    (4, 4),
                )
            ),
            fully_connected_layers=tuple(
                int(value)
                for value in encoder_config.get(
                    "fully_connected_layers",
                    (),
                )
            ),
            dropout=float(
                encoder_config.get(
                    "dropout",
                    0.0,
                )
            ),
            output_activation=str(
                encoder_config.get(
                    "output_activation",
                    "tanh",
                )
            ),
        )

    else:
        raise ValueError(f"Unknown encoder={encoding_str} for target={target}")

    return encoder


class Encoder(ABC):
    def __init__(self, n_inputs: int, n_outputs: int):
        """
        Base constructor for a decoder, as each will need to know how many
        qubits it needs to decode to.

        Args:
            n_inputs: how many classical input features will be used.
            n_outputs: how many values in the output of the encoder, which
                will be used as inputs for the quantum circuit.
        """
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

    @abstractmethod
    def __call__(self, inputs: torch.Tensor, genome: CircuitGenome):
        """
        Given the torch tensor of input values and the circuit genome,
        this should set the input qubits of the quantum circuit model
        being used for pytorch forward/backward passes.

        Args:
            inputs: the x (input) tensor for a sample
            genome: the circuit genome whose quantum circuit
                inputs are being set
        """
        pass

    def get_constructor_args(self) -> dict[str, Any]:
        """Returns constructor arguments required to rebuild the encoder.

        Returns:
            Constructor arguments for this encoder.
        """
        return {
            "n_inputs": self.n_inputs,
            "n_outputs": self.n_outputs,
        }

    def to_dict(self) -> dict[str, any]:
        """
        This converts the encoder to a dict so it can be written as
        JSON for passing via MPI message passing, or saved to disk.

        The default is just to write the class name and then arguments
        used to construct the encoder in the args entry. This allows
        us to use a standard method with reflection for construcing
        the encoders backfrom JSON.
        """

        constructor_args = self.get_constructor_args()
        response_dict = {
            "class": type(self).__name__,
            "args": constructor_args,
        }

        # for Decoders that are pytorch modules, also save
        # their parameters (so we don't need to reimplement
        # this for each of them).
        if isinstance(self, torch.nn.Module):
            state_dict = self.state_dict()

            serialized_dict = {
                name: tensor.cpu().numpy().tolist()
                for name, tensor in state_dict.items()
            }

            """
            logger.debug("serialized nn.Module tensor dict:")
            logger.debug(json.dumps(serialized_dict, indent=4))
            """

            response_dict["args"]["state_dict"] = serialized_dict

        # logger.debug("encoder json is:")
        # logger.debug(json.dumps(response_dict, ensure_ascii=False, indent=4))

        return response_dict

    @classmethod
    def from_dict(cls, serialized: dict[str, any]) -> Encoder:
        """
        Given a dict created from the serialized JSON generated by the
        `to_dict` method, create the class with the given args from
        the JSON.

        Args:
            serialized: is the dict created from the serialized JSON.

        Returns:
            A constructed Decoder object created from the given JSON.
        """

        # Copy the arguments so the serialized dictionary is not modified.
        constructor_args = dict(serialized["args"])

        # remove the module parameters so we can call the default
        # constructor
        serialized_state_dict = None
        if "state_dict" in constructor_args:
            serialized_state_dict = constructor_args.pop("state_dict")

        class_init = globals()[serialized["class"]]
        class_object = class_init(**constructor_args)

        if (
            isinstance(class_object, torch.nn.Module)
            and serialized_state_dict is not None
        ):
            # convert the JSON values back into tensors
            state_dict = {
                name: torch.tensor(list_data)
                for name, list_data in serialized_state_dict.items()
            }

            # set the module's parameter tensors
            class_object.load_state_dict(state_dict)

        return class_object

    def copy(self) -> Encoder:
        """
        Creates a copy of the encoder (if needed) so modifying a genome's
        encoder does not effect the one of its parent.
        """
        pass


class IdentityEncoder(Encoder):
    def __call__(self, inputs: torch.Tensor, genome: CircuitGenome):
        """
        A default encoder for basis and angle encodings, which simply returns
        the same inputs.

        Args:
            inputs: the x (input) tensor for a sample, this will be the
                same as the output.
            genome: the circuit genome whose quantum circuit
                inputs are being set
        """
        if inputs.ndim > 2:
            inputs = torch.flatten(
                inputs,
                start_dim=1,
            )
            
        return inputs

    def copy(self) -> Encoder:
        """
        This encoder has no state so we don't need to do a copy.
        """
        return self


class LinearEncoder(Encoder, torch.nn.Module):
    def __init__(self, n_inputs: int, n_outputs: int):
        """
        Base constructor for a decoder, as each will need to know how many
        outputs it needs to decode to.

        Args:
            n_inputs: how many classical input features will be used.
            n_outputs: how many outputs from the linear encoder, which should
                be the same as the number of qubits that will be used as inputs
                for the quantum circuit.
        """
        # initialize both superclasses
        torch.nn.Module.__init__(self)
        Encoder.__init__(self, n_inputs, n_outputs)

        logger.debug(
            f"creating linear encoder with n_inputs: {n_inputs} and n_outputs: {n_outputs}"
        )

        self.layer = torch.nn.Linear(self.n_inputs, self.n_outputs)

        # initalize the layer weights randomly
        torch.nn.init.xavier_uniform_(self.layer.weight)
        torch.nn.init.zeros_(self.layer.bias)

    def __call__(self, inputs: torch.Tensor, genome: CircuitGenome):
        """
        Applies a linear embedding from the input values to the input
        wires of the quantum circuit as U3 gates.

        Args:
            inputs: the x (input) tensor for a sample
            genome: the circuit genome whose quantum circuit
                inputs are being set
        """

        """
        print(f"forward on linear incoder, inputs: {inputs}, type: {inputs.dtype}")
        print(f"weights: {self.layer.weight}, type: {self.layer.weight.dtype}")
        print(f"biases: {self.layer.bias}, type: {self.layer.bias.dtype}")
        """

        # Flatten all dimensions except the batch dimension.
        if inputs.ndim > 2:
            inputs = torch.flatten(
                inputs,
                start_dim=1,
            )

        if inputs.shape[-1] != self.n_inputs:
            raise ValueError(
                f"LinearEncoder expected {self.n_inputs} input features, "
                f"but received {inputs.shape[-1]}."
            )

        # linear layer requires float32 values
        encoding = self.layer(inputs.float())

        return encoding

    def copy(self) -> Encoder:
        """
        Returns:
            A deepcopy of this decoder so if it can be reused by child
            genomes without modifying the parents decoder.
        """
        return copy.deepcopy(self)


class CNNEncoder(Encoder, torch.nn.Module):
    """Configurable convolutional encoder for image classification.

    The encoder processes an image batch and produces one quantum-input vector
    per image. Both the convolutional feature extractor and fully connected
    projection head are configurable.

    Args:
        n_inputs: Flattened image size.
        n_outputs: Number of values required by the quantum circuit.
        input_channels: Number of input image channels.
        input_height: Input image height.
        input_width: Input image width.
        conv_blocks: Sequence of convolution-block configurations.
        adaptive_pool_size: Final adaptive pooling height and width.
        fully_connected_layers: Hidden dimensions in the projection head.
        dropout: Dropout probability applied between fully connected layers.
        output_activation: Activation applied to the final quantum inputs.
            Supported values are ``tanh``, ``sigmoid``, and ``identity``.

    Raises:
        ValueError: If the architecture configuration is invalid.
    """

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        input_channels: int,
        input_height: int,
        input_width: int,
        conv_blocks: Sequence[Mapping[str, Any]],
        adaptive_pool_size: Sequence[int] = (4, 4),
        fully_connected_layers: Sequence[int] = (),
        dropout: float = 0.0,
        output_activation: str = "tanh",
    ) -> None:
        torch.nn.Module.__init__(self)
        Encoder.__init__(self, n_inputs, n_outputs)

        if n_inputs <= 0:
            raise ValueError("n_inputs must be positive.")
        if n_outputs <= 0:
            raise ValueError("n_outputs must be positive.")
        if input_channels <= 0:
            raise ValueError("input_channels must be positive.")
        if input_height <= 0 or input_width <= 0:
            raise ValueError("Input height and width must be positive.")
        # if not conv_blocks:
        #     raise ValueError("At least one convolution block is required.")
        if len(adaptive_pool_size) != 2:
            raise ValueError("adaptive_pool_size must contain height and width.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        expected_inputs = input_channels * input_height * input_width
        if n_inputs != expected_inputs:
            raise ValueError(
                f"n_inputs={n_inputs} does not match image shape "
                f"{input_channels}x{input_height}x{input_width}="
                f"{expected_inputs}."
            )

        self.n_inputs = int(n_inputs)
        self.n_outputs = int(n_outputs)
        self.input_channels = int(input_channels)
        self.input_height = int(input_height)
        self.input_width = int(input_width)
        self.conv_blocks_config = [dict(block) for block in conv_blocks]
        self.adaptive_pool_size = tuple(int(value) for value in adaptive_pool_size)
        self.fully_connected_layers = tuple(
            int(value) for value in fully_connected_layers
        )
        self.dropout = float(dropout)
        self.output_activation = output_activation

        self.features, final_channels = self._build_feature_extractor()
        self.projection = self._build_projection_head(final_channels)

        self._initialize_parameters()

    @staticmethod
    def _activation(name: str) -> torch.nn.Module:
        """Creates an activation module.

        Args:
            name: Activation name.

        Returns:
            PyTorch activation module.

        Raises:
            ValueError: If the activation is unsupported.
        """
        activations = {
            "relu": torch.nn.ReLU,
            "gelu": torch.nn.GELU,
            "silu": torch.nn.SiLU,
            "leaky_relu": torch.nn.LeakyReLU,
            "tanh": torch.nn.Tanh,
            "sigmoid": torch.nn.Sigmoid,
            "identity": torch.nn.Identity,
        }

        try:
            return activations[name]()
        except KeyError as error:
            raise ValueError(f"Unsupported activation: {name}") from error

    @staticmethod
    def _pooling(
        config: Mapping[str, Any] | None,
    ) -> torch.nn.Module | None:
        """Creates a pooling module.

        Args:
            config: Pooling configuration or ``None``.

        Returns:
            Pooling module or ``None``.

        Raises:
            ValueError: If the pooling type is unsupported.
        """
        if config is None:
            return None

        pool_type = str(config.get("type", "max"))
        kernel_size = int(config.get("kernel_size", 2))
        stride = config.get("stride", kernel_size)

        if pool_type == "max":
            return torch.nn.MaxPool2d(
                kernel_size=kernel_size,
                stride=stride,
            )

        if pool_type == "avg":
            return torch.nn.AvgPool2d(
                kernel_size=kernel_size,
                stride=stride,
            )

        raise ValueError(f"Unsupported pooling type: {pool_type}")

    def _build_feature_extractor(
        self,
    ) -> tuple[torch.nn.Sequential, int]:
        """Builds the configured convolutional feature extractor.

        Returns:
            Feature extractor and final channel count.
        """
        if not self.conv_blocks_config:
            return torch.nn.Identity(), None

        layers: list[torch.nn.Module] = []
        in_channels = self.input_channels

        for index, block in enumerate(self.conv_blocks_config):
            out_channels = int(block["out_channels"])
            kernel_size = int(block.get("kernel_size", 3))
            stride = int(block.get("stride", 1))
            padding = int(block.get("padding", 1))
            dilation = int(block.get("dilation", 1))
            use_bias = not bool(block.get("batch_norm", True))

            if out_channels <= 0:
                raise ValueError("Conv block " f"{index} has invalid out_channels.")

            layers.append(
                torch.nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    bias=use_bias,
                )
            )

            if bool(block.get("batch_norm", True)):
                layers.append(torch.nn.BatchNorm2d(out_channels))

            layers.append(
                self._activation(
                    str(
                        block.get(
                            "activation",
                            "relu",
                        )
                    )
                )
            )

            block_dropout = float(block.get("dropout", 0.0))
            if block_dropout > 0.0:
                layers.append(torch.nn.Dropout2d(block_dropout))

            pooling = self._pooling(block.get("pool"))
            if pooling is not None:
                layers.append(pooling)

            in_channels = out_channels

        layers.append(torch.nn.AdaptiveAvgPool2d(self.adaptive_pool_size))

        return (
            torch.nn.Sequential(*layers),
            in_channels,
        )

    def _build_projection_head(
        self,
        final_channels: int,
    ) -> torch.nn.Sequential:
        """Builds the configurable fully connected projection head.

        Args:
            final_channels: Number of channels produced by the feature
                extractor.

        Returns:
            Projection head producing ``n_outputs`` values.
        """
        if final_channels is None:
            flattened_size = self.n_inputs
        else:
            pooled_height, pooled_width = self.adaptive_pool_size
            flattened_size = final_channels * pooled_height * pooled_width

        layers: list[torch.nn.Module] = [torch.nn.Flatten(start_dim=1)]

        current_size = flattened_size

        for hidden_size in self.fully_connected_layers:
            if hidden_size <= 0:
                raise ValueError("Fully connected layer sizes " "must be positive.")

            layers.append(
                torch.nn.Linear(
                    current_size,
                    hidden_size,
                )
            )
            layers.append(torch.nn.ReLU())

            if self.dropout > 0.0:
                layers.append(torch.nn.Dropout(self.dropout))

            current_size = hidden_size

        layers.append(
            torch.nn.Linear(
                current_size,
                self.n_outputs,
            )
        )
        layers.append(self._activation(self.output_activation))

        return torch.nn.Sequential(*layers)

    def _initialize_parameters(self) -> None:
        """Initializes convolutional and linear parameters."""
        for module in self.modules():
            if isinstance(
                module,
                torch.nn.Conv2d,
            ):
                torch.nn.init.kaiming_normal_(
                    module.weight,
                    nonlinearity="relu",
                )
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

            elif isinstance(
                module,
                torch.nn.Linear,
            ):
                torch.nn.init.xavier_uniform_(module.weight)
                torch.nn.init.zeros_(module.bias)

    def __call__(
        self,
        inputs: torch.Tensor,
        genome: Any | None = None,
    ) -> torch.Tensor:
        """Encodes a batch of images.

        Args:
            inputs: Tensor shaped
                ``[batch_size, channels, height, width]``.
            genome: Unused owning EXAQC model.

        Returns:
            Tensor shaped
            ``[batch_size, n_quantum_inputs]``.

        Raises:
            ValueError: If the image shape is invalid.
        """
        if inputs.ndim != 4:
            raise ValueError(
                "CNNEncoder expects "
                "[batch_size, channels, height, width], "
                f"received {tuple(inputs.shape)}."
            )

        expected_shape = (
            self.input_channels,
            self.input_height,
            self.input_width,
        )

        if tuple(inputs.shape[1:]) != (expected_shape):
            raise ValueError(
                f"Expected image shape "
                f"{expected_shape}, received "
                f"{tuple(inputs.shape[1:])}."
            )

        features = self.features(inputs.float())
        return self.projection(features)

    def get_constructor_args(
        self,
    ) -> dict[str, Any]:
        """Returns serialization constructor arguments.

        Returns:
            JSON-serializable constructor arguments.
        """
        return {
            "n_inputs": self.n_inputs,
            "n_outputs": self.n_outputs,
            "input_channels": (self.input_channels),
            "input_height": self.input_height,
            "input_width": self.input_width,
            "conv_blocks": [
                copy.deepcopy(block) for block in (self.conv_blocks_config)
            ],
            "adaptive_pool_size": list(self.adaptive_pool_size),
            "fully_connected_layers": list(self.fully_connected_layers),
            "dropout": self.dropout,
            "output_activation": (self.output_activation),
        }

    def copy(self) -> CNNEncoder:
        """Returns an independent CNN copy.

        Returns:
            Deep copy of this encoder.
        """
        return copy.deepcopy(self)
