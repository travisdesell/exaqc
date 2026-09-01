from __future__ import annotations

import copy
import torch

from abc import ABC, abstractmethod
from loguru import logger

from typing import TYPE_CHECKING, Mapping, Any

from src.circuits.layer_spec import LayerSpec

if TYPE_CHECKING:
    from src.circuits.circuit import CircuitGenome


DECODING_OPTIONS = ["clipped", "linear", "quantum_conv"]


def initialize_decoder(
    target: str,
    decoding_str: str,
    n_inputs: int,
    n_outputs: int,
    config: Mapping[str, Any] | None = None,
) -> Decoder:
    """
    Given the target system (e.g., pennylane or qiskit) create a
    new decoder to perform decodings from classical to quantum
    which will set the input parameter and input gate for each wire.

    Args:
        target: the target system (pennylane or qiskit)
        decoding_str: a string representation of the decoder
            to be created
        n_outputs: how many output values are required for the loss
            function, which the decoder will perform some operation
            to reduce its input tensor to.
        config: Optional decoder-specific configuration. The CNN decoder
            requires ``input_channels``, ``input_height``, and ``input_width``.

    Raises:
        ValueError: If the decoder is unknown, or a clipped decoder is requested
            with ``n_outputs > n_inputs`` (it can only keep a prefix of its
            input, so it cannot produce more outputs than it receives).
    """

    logger.info(
        f"creating {decoding_str} decoder with n_inputs: {n_inputs} and n_outputs: {n_outputs}"
    )

    decoder = None

    if decoding_str == "clipped":
        # Validate before constructing: a clipped decoder keeps only the first
        # n_outputs values, so it cannot expand its input.
        if n_outputs > n_inputs:
            raise ValueError(
                f"ClippedDecoder requires n_outputs <= n_inputs, but got "
                f"n_outputs={n_outputs} and n_inputs={n_inputs}. It keeps only "
                f"the first n_outputs values, so it cannot produce more outputs "
                f"than it receives; reduce n_outputs (or use a linear decoder to "
                f"expand the size)."
            )
        decoder = ClippedDecoder(n_inputs, n_outputs)

    elif decoding_str == "linear":
        decoder = LinearDecoder(n_inputs, n_outputs)

    elif decoding_str == "quantum_conv":
        decoder_config = dict(config or {})

        decoder = QuantumConvDecoder(
            n_inputs=n_inputs,
            n_outputs=n_outputs,
            feature_height=int(
                decoder_config.get(
                    "quantum_feature_height",
                    4,
                )
            ),
            feature_width=int(
                decoder_config.get(
                    "quantum_feature_width",
                    4,
                )
            ),
            out_channels=int(
                decoder_config.get(
                    "quantum_out_channels",
                    128,
                )
            ),
            adaptive_pool_size=tuple(
                decoder_config.get(
                    "adaptive_pool_size",
                    [2, 2],
                )
            ),
            fully_connected_layers=tuple(
                decoder_config.get(
                    "fully_connected_layers",
                    [256, 128],
                )
            ),
        )

    else:
        raise ValueError(f"Unknown decoder={decoding_str} for target={target}")

    return decoder


class Decoder(ABC):
    def __init__(self, n_inputs: int, n_outputs: int):
        """
        Base constructor for a decoder, as each will need to know how many
        outputs it needs to decode to.

        Args:
            n_inputs: how many input values will be passed into the decoder,
                which we need to initialize more complex decoders.
            n_outputs: how many output values are required for the loss
                function, which the decoder will perform some operation
                to reduce its input tensor to.
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

    def to_dict(self) -> dict[str, any]:
        """
        This converts the decoder to a dict so it can be written as
        JSON for passing via MPI message passing, or saved to disk.

        The default is just to write the class name and then arguments
        used to construct the decoder in the args entry. This allows
        us to use a standard method with reflection for construcing
        the decoders backfrom JSON.
        """

        response_dict = {
            "class": type(self).__name__,
            "args": {
                "n_inputs": self.n_inputs,
                "n_outputs": self.n_outputs,
            },
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

        # logger.debug("decoder json is:")
        # logger.debug(json.dumps(response_dict, ensure_ascii=False, indent=4))

        return response_dict

    @classmethod
    def from_dict(cls, serialized: dict[str, any]) -> Decoder:
        """
        Given a dict created from the serialized JSON generated by the
        `to_dict` method, create the class with the given args from
        the JSON.

        Args:
            serialized: is the dict created from the serialized JSON.

        Returns:
            A constructed Decoder object created from the given JSON.
        """

        # remove the module parameters so we can call the default
        # constructor
        serialized_state_dict = None
        if "state_dict" in serialized["args"].keys():
            serialized_state_dict = serialized["args"].pop("state_dict")

        class_init = globals()[serialized["class"]]
        class_object = class_init(**serialized["args"])

        if isinstance(class_object, torch.nn.Module):
            # convert the JSON values back into tensors
            state_dict = {
                name: torch.tensor(list_data)
                for name, list_data in serialized_state_dict.items()
            }

            # set the module's parameter tensors
            class_object.load_state_dict(state_dict)

        return class_object

    def copy(self) -> Decoder:
        """
        Creates a copy of the decoder (if needed) so modifying a genome's
        decoder does not effect the one of its parent.
        """
        pass

    def describe_layers(self) -> list[LayerSpec]:
        """Describes this decoder as an ordered list of drawable layers.

        Used by the architecture diagram compositor
        (:func:`src.utils.helpers.draw_network`). The base implementation
        returns a single generic block; subclasses override this to expose
        their real layer structure.

        Returns:
            A list of :class:`~src.circuits.layer_spec.LayerSpec` describing the
            decoder's layers in input-to-output order.
        """
        return [
            LayerSpec(
                kind="block",
                label=type(self).__name__,
                in_shape=(self.n_inputs,),
                out_shape=(self.n_outputs,),
            )
        ]


class ClippedDecoder(Decoder):
    def __call__(self, inputs: torch.Tensor, genome: CircuitGenome):
        """
        Returns the first N values, for the decoding discarding
        the rest, where N is the expected number of outputs
        for the loss function.

        The values are returned unmodified (no sum-to-1 rescaling): every
        downstream consumer re-applies its own normalization -- classification
        uses ``CrossEntropyLoss`` (an internal ``log_softmax``) and the discrete
        reinforcement-learning policies use ``Categorical(logits=...)`` (an
        internal softmax), while the continuous policies read per-dimension mean
        and log-std slots directly -- so none of them require a normalized input.

        Args:
            inputs: the z (output) tensor from a quantum circuit.
            genome: the circuit genome whose quantum circuit
                inputs are being set

        Returns:
            The leading ``self.n_outputs`` values of ``inputs`` along the last
            dimension, unmodified.
        """

        return inputs[..., : self.n_outputs]

    def describe_layers(self) -> list[LayerSpec]:
        """Describes the clipped decoder as a single pass-through block.

        Returns:
            A one-element list with a ``"passthrough"`` :class:`LayerSpec` (it
            keeps the leading ``n_outputs`` values without trainable weights).
        """
        return [
            LayerSpec(
                kind="passthrough",
                label="Clip",
                in_shape=(self.n_inputs,),
                out_shape=(self.n_outputs,),
            )
        ]

    def copy(self) -> Decoder:
        return self


class LinearDecoder(torch.nn.Module, Decoder):
    def __init__(self, n_inputs: int, n_outputs: int):
        """
        Base constructor for a decoder, as each will need to know how many
        outputs it needs to decode to.

        Args:
            n_inputs: how many inputs will be passed to the decoder, which
                is typically the number of output qubits times 2.
            n_outputs: how many output values are required for the loss
                function, which the decoder will perform some operation
                to reduce its input tensor to.
        """
        # initialize both superclasses
        torch.nn.Module.__init__(self)
        Decoder.__init__(self, n_inputs, n_outputs)

        logger.debug(
            f"creating decoder with n_inputs: {n_inputs} and n_outputs: {n_outputs}"
        )

        self.layer = torch.nn.Linear(self.n_inputs, self.n_outputs)

        # initalize the layer weights randomly
        torch.nn.init.xavier_uniform_(self.layer.weight)
        torch.nn.init.zeros_(self.layer.bias)

    def __call__(self, inputs: torch.Tensor, genome: CircuitGenome):
        """
        Returns the first N values, for the decoding discarding
        the rest, where N is the expected number of outputs
        for the loss function.

        Args:
            inputs: the z (output) tensor from a quantum circuit.
            genome: the circuit genome whose quantum circuit
                inputs are being set
        """

        """
        print(f"forward on linear, inputs: {inputs}, type: {inputs.dtype}")
        print(f"weights: {self.layer.weight}, type: {self.layer.weight.dtype}")
        print(f"biases: {self.layer.bias}, type: {self.layer.bias.dtype}")
        """

        # linear layer requires float32 values
        return self.layer(inputs.float())

    def describe_layers(self) -> list[LayerSpec]:
        """Describes the linear decoder as a single fully connected layer.

        Returns:
            A one-element list with an ``"fc"`` :class:`LayerSpec` mapping
            ``n_inputs`` -> ``n_outputs``.
        """
        return [
            LayerSpec(
                kind="fc",
                label="Linear",
                in_shape=(self.n_inputs,),
                out_shape=(self.n_outputs,),
            )
        ]

    def copy(self) -> Decoder:
        """
        Returns:
            A deepcopy of this decoder so if it can be reused by child
            genomes without modifying the parents decoder.
        """
        return copy.deepcopy(self)


class QuantumConvDecoder(
    torch.nn.Module,
    Decoder,
):
    """Reconstructs a feature map after quantum convolution.

    Each image has been converted into several patches by
    ``QuantumConvEncoder``. The same evolved quantum circuit is evaluated
    on every patch. This decoder reconstructs those patch outputs into a
    spatial feature map and continues through the classical CNN head.

    Args:
        n_inputs: Number of outputs from each quantum patch.
        n_outputs: Number of classification classes.
        feature_height: Height of reconstructed quantum feature map.
        feature_width: Width of reconstructed quantum feature map.
        out_channels: Number of channels after the quantum block.
        adaptive_pool_size: Spatial dimensions before flattening.
        fully_connected_layers: Dense classification layers.
    """

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        feature_height: int = 4,
        feature_width: int = 4,
        out_channels: int = 128,
        adaptive_pool_size: tuple[int, int] = (2, 2),
        fully_connected_layers: tuple[int, ...] = (
            256,
            128,
        ),
    ) -> None:
        torch.nn.Module.__init__(self)
        Decoder.__init__(
            self,
            n_inputs,
            n_outputs,
        )

        self.feature_height = int(feature_height)
        self.feature_width = int(feature_width)
        self.out_channels = int(out_channels)

        self.adaptive_pool_size = tuple(int(value) for value in adaptive_pool_size)

        self.fully_connected_layers = tuple(
            int(value) for value in fully_connected_layers
        )

        # Quantum outputs become channels of the new
        # spatial feature map.
        self.channel_expansion = torch.nn.Conv2d(
            in_channels=self.n_inputs,
            out_channels=self.out_channels,
            kernel_size=1,
        )

        self.batch_norm = torch.nn.BatchNorm2d(self.out_channels)

        self.activation = torch.nn.ReLU()

        self.pool = torch.nn.AdaptiveAvgPool2d(self.adaptive_pool_size)

        flattened_size = (
            self.out_channels * self.adaptive_pool_size[0] * self.adaptive_pool_size[1]
        )

        layers: list[torch.nn.Module] = [torch.nn.Flatten(start_dim=1)]

        current_size = flattened_size

        for hidden_size in self.fully_connected_layers:
            layers.extend(
                [
                    torch.nn.Linear(
                        current_size,
                        hidden_size,
                    ),
                    torch.nn.ReLU(),
                ]
            )

            current_size = hidden_size

        layers.append(
            torch.nn.Linear(
                current_size,
                self.n_outputs,
            )
        )

        self.classifier = torch.nn.Sequential(*layers)

        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        """Initializes trainable classical parameters."""

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
        genome: CircuitGenome,
    ) -> torch.Tensor:
        """Reconstructs and classifies quantum feature maps.

        Args:
            inputs: Quantum outputs with shape
                ``[B * n_patches, n_inputs]``.
            genome: Owning EXAQC genome.

        Returns:
            Classification logits with shape
            ``[B, n_outputs]``.
        """

        n_patches = self.feature_height * self.feature_width

        if inputs.shape[0] % n_patches != 0:
            raise ValueError(
                "Quantum patch output count "
                f"{inputs.shape[0]} is not divisible by "
                f"{n_patches} patches per image."
            )

        batch_size = inputs.shape[0] // n_patches

        # [B * 16, 8]
        #       ->
        # [B, 4, 4, 8]
        features = inputs.reshape(
            batch_size,
            self.feature_height,
            self.feature_width,
            self.n_inputs,
        )

        # [B, 4, 4, 8]
        #       ->
        # [B, 8, 4, 4]
        features = features.permute(
            0,
            3,
            1,
            2,
        ).contiguous()

        # [B, 8, 4, 4]
        #       ->
        # [B, 128, 4, 4]
        features = self.channel_expansion(features.float())

        features = self.batch_norm(features)

        features = self.activation(features)

        # [B, 128, 4, 4]
        #       ->
        # [B, 128, 2, 2]
        features = self.pool(features)

        # 128 * 2 * 2 = 512
        #
        # 512 -> 256 -> 128 -> 10
        return self.classifier(features)

    def copy(self) -> "QuantumConvDecoder":
        """Returns an independent decoder copy."""

        return copy.deepcopy(self)

    def to_dict(self) -> dict[str, any]:
        """Serializes the decoder for EXAQC/MPI."""

        state_dict = {
            name: tensor.cpu().numpy().tolist()
            for name, tensor in self.state_dict().items()
        }

        return {
            "class": type(self).__name__,
            "args": {
                "n_inputs": self.n_inputs,
                "n_outputs": self.n_outputs,
                "feature_height": (self.feature_height),
                "feature_width": (self.feature_width),
                "out_channels": self.out_channels,
                "adaptive_pool_size": list(self.adaptive_pool_size),
                "fully_connected_layers": list(self.fully_connected_layers),
                "state_dict": state_dict,
            },
        }
