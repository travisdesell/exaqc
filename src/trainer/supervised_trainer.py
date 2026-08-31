from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import torch
from loguru import logger
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from src.circuits.circuit import CircuitGenome
from src.dropout.quantum_dropout import (
    gate_dropout,
    rotation_dropout,
    entangling_dropout,
    qubit_dropout,
    innovation_dropout,
)


class SupervisedTrainer:
    """Trains EXAQC hybrid models using a fully batched execution path."""

    def __init__(
        self,
        training_dataloader: DataLoader,
        validation_dataloader: DataLoader,
        training_loss_function: Callable[[Tensor, Tensor], Tensor],
        validation_loss_function: Callable[[Tensor, Tensor], Tensor],
        metrics: dict[str, Any],
        testing_dataloader: DataLoader | None = None,
        testing_loss_function: Callable[[Tensor, Tensor], Tensor] | None = None,
        device: str | None = None,
    ) -> None:
        """
        This creates a SupervisedTrainer object which can be (re)used to train circuit
        genomes given the provided training and validation dataloaders.

        Args:
            training_dataloader: a pytorch DataLoader object which can iterate over the
                training samples
            validation_dataloader: a pytorch DataLoader object which can iterate over the
                validation samples
            training_loss_function: provides the loss function used for training the
                genome
            validation_loss_function: provides the loss function used for calculating loss
                on the validation data. this may be different than the training data for
                example, when doing cross entropy loss where the class counts are different
                on training and validation data.
            metrics: is a dict where each key is the name of a metric, and each value is
                a function used to calculate a different metric (e.g., accuracy) which are
                different/in addition to the loss function.
            testing_dataloader: Optional held-out test dataloader.
            testing_loss_function: Optional held-out test loss function.
                Defaults to the validation loss function.
        """

        self.training_dataloader = training_dataloader
        self.validation_dataloader = validation_dataloader
        self.testing_dataloader = testing_dataloader
        self.training_loss_function = training_loss_function
        self.validation_loss_function = validation_loss_function
        self.testing_loss_function = (
            testing_loss_function
            if testing_loss_function is not None
            else validation_loss_function
        )
        self.metrics = metrics

        self.device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Move module-based loss functions and their weights to the same device.
        if isinstance(self.training_loss_function, torch.nn.Module):
            self.training_loss_function.to(self.device)

        if isinstance(self.validation_loss_function, torch.nn.Module):
            self.validation_loss_function.to(self.device)

        if self.testing_loss_function is not None and isinstance(
            self.testing_loss_function, torch.nn.Module
        ):
            self.testing_loss_function.to(self.device)

    def _apply_quantum_dropout(self, genome) -> None:
        """Samples and applies the configured quantum dropout strategy.

        Args:
            genome: The CircuitGenome on which to apply dropout

        Returns:
            None
        """
        genome.clear_quantum_dropout()

        dropout_type = genome.hyperparameters.get(
            "quantum_dropout_type",
            "none",
        )

        dropout_rate = float(
            genome.hyperparameters.get(
                "quantum_dropout_rate",
                0.0,
            )
        )

        if dropout_rate == 0.0 or dropout_type == "none":
            return

        if dropout_type == "gate":
            genome.dropout_gate_innovations = gate_dropout(
                genome.gates,
                dropout_rate,
            )

        elif dropout_type == "rotation":
            genome.dropout_gate_innovations = rotation_dropout(
                genome.gates,
                dropout_rate,
            )

        elif dropout_type == "entangling":
            genome.dropout_gate_innovations = entangling_dropout(
                genome.gates,
                dropout_rate,
            )

        elif dropout_type == "qubit":
            genome.dropout_qubits = qubit_dropout(
                genome.qubits,
                dropout_rate,
            )

        elif dropout_type == "innovation":
            genome.dropout_gate_innovations = innovation_dropout(
                genome.gates,
                dropout_rate,
                innovation_strength=float(
                    genome.hyperparameters.get(
                        "quantum_dropout_innovation_strength",
                        0.5,
                    )
                ),
            )

        else:
            raise ValueError(f"Unknown quantum dropout type: {dropout_type}")

    def get_metrics(
        self,
        genome: CircuitGenome,
        dataloader: DataLoader,
        loss_function: Callable[[Tensor, Tensor], Tensor],
        optimizer: Optimizer | None = None,
        epoch: int | None = None,
    ) -> dict[str, Any]:
        """
        Calculates the metrics for the provided genome and dataloader. Will optionally
            update gradients if is_training is set to true.

        Args:
            genome: the circuit genome to evaluate (without updating weights)
                on the validation data for this trainer.
            dataloader: a pytorch dataloader for the data to evaluate on.
            loss_function: the loss function to use for data evaluation.
            optimizer: is the optimizer use to train the genome if provided. if not
                provided metrics are just being gathered for inference/validation and
                weights should not be updated.
            epoch: the current epoch (if training), None otherwise. if specified this
                will be added to the metrics dict for better metadata parsing.

        Returns:
            A dictionary from each metric name to the metric value calculated
            over the validation data.

        Raises:
            ValueError: If the model's predictions are not 2-D
                ``[batch_size, n_classes]``, or if the prediction and target
                batch sizes differ.
        """
        is_training = optimizer is not None
        genome.hybrid_model.train() if is_training else genome.hybrid_model.eval()

        for metric in self.metrics.values():
            metric.reset()

        total_loss = 0.0
        total_samples = 0

        with torch.set_grad_enabled(is_training):
            for batch_index, (x_batch, y_batch) in enumerate(dataloader):
                logger.debug("batch: {} / {}", batch_index, len(dataloader))

                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                if is_training:
                    optimizer.zero_grad(set_to_none=True)

                    self._apply_quantum_dropout(genome)
                    if genome.dropout_qubits:
                        genome.hybrid_model.dropout_qubits = genome.dropout_qubits
                        genome.hybrid_model.output_qubits = genome.output_qubits
                        genome.hybrid_model.quantum_output_mode = (
                            genome.hyperparameters["quantum_output_mode"]
                        )

                else:
                    # Validation/test always uses the complete evolved circuit.
                    genome.clear_quantum_dropout()
                    if genome.dropout_qubits:
                        genome.hybrid_model.dropout_qubits = set()

                predictions = genome.forward(x_batch)

                if predictions.ndim != 2:
                    raise ValueError(
                        "Classification predictions must have shape "
                        "[batch_size, n_classes], received "
                        f"{tuple(predictions.shape)}."
                    )
                if predictions.shape[0] != y_batch.shape[0]:
                    raise ValueError(
                        "Prediction and target batch sizes differ: "
                        f"{predictions.shape[0]} != {y_batch.shape[0]}."
                    )

                loss = loss_function(predictions.float(), y_batch.long())

                if is_training:
                    loss.backward()
                    optimizer.step()

                current_batch_size = int(y_batch.shape[0])
                total_loss += float(loss.detach().item()) * current_batch_size
                total_samples += current_batch_size

                with torch.no_grad():
                    for prediction, target in zip(predictions, y_batch):
                        for metric in self.metrics.values():
                            metric.accumulate(prediction.float(), target.long())

        genome.clear_quantum_dropout()
        if hasattr(genome.hybrid_model, "dropout_qubits"):
            genome.hybrid_model.dropout_qubits = set()

        metric_results: dict[str, Any] = {
            "loss": (total_loss / total_samples if total_samples else float("nan"))
        }
        for metric_name, metric in self.metrics.items():
            metric_results[metric_name] = metric.calculate()

        if epoch is not None:
            metric_results["epoch"] = epoch
        return metric_results

    def train(self, genome: CircuitGenome) -> None:
        """
        Given the data loaders and loss functions provided to this SupervisedTrainer,
        this will train the provided circuit genome given its hyperparameter
        specifications.

        Args:
            genome: is the CircuitGenome to train. This method will initialize
                its ``hybrid_model`` (via ``genome.initialize_model()``) so it
                can be trained with pytorch.
        """
        genome.initialize_model()
        genome.hybrid_model.to(self.device)

        hyperparameters = genome.hyperparameters
        learning_rate = float(hyperparameters["learning_rate"])
        epochs = int(hyperparameters["epochs"])

        # initalize the epoch metrics for tracking/data mining
        genome.metadata["training_epoch_metrics"] = []
        genome.metadata["validation_epoch_metrics"] = []

        n_trainable_parameters = sum(
            p.numel() for p in genome.parameters() if p.requires_grad
        )
        genome.metadata["n_trainable_parameters"] = n_trainable_parameters
        genome.metadata["n_circuit_parameters"] = genome.get_genome_circuit_parameters()

        logger.debug(f"hybrid model n trainable parameters: {n_trainable_parameters}")

        if n_trainable_parameters == 0:
            # this model has no parameters so it can't be trained. instead
            # just evaluate it on the validation data.
            logger.info("Model has no trainable parameters; evaluating only.")

            # calculate the metrics on the training data
            training_metric_results = self.get_metrics(
                genome,
                dataloader=self.training_dataloader,
                loss_function=self.training_loss_function,
            )

            # calculate the metrics on the validation data
            validation_metric_results = self.get_metrics(
                genome,
                dataloader=self.validation_dataloader,
                loss_function=self.validation_loss_function,
            )
            genome.metadata["best_training_metrics"] = training_metric_results
            genome.metadata["best_validation_metrics"] = validation_metric_results

            logger.info(f"training metrics were: {training_metric_results}")
            logger.info(f"validation metrics were: {validation_metric_results}")

            return

        optimizer = torch.optim.Adam(
            genome.parameters(),
            lr=learning_rate,
            weight_decay=float(hyperparameters.get("weight_decay", 0.0)),
        )

        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer,
        #     mode="min",
        #     factor=0.5,
        #     patience=3,
        #     min_lr=1e-6,
        # )

        best_loss = math.inf
        best_epoch = 0
        improvement_cutoff = int(hyperparameters.get("improvement_cutoff", 2))
        best_parameters = genome.clone_state_dict()

        for epoch in range(epochs):
            training_metric_results = self.get_metrics(
                genome,
                dataloader=self.training_dataloader,
                loss_function=self.training_loss_function,
                optimizer=optimizer,
                epoch=epoch,
            )
            logger.info(
                "[epoch {}] training metrics: {}",
                epoch,
                training_metric_results,
            )
            genome.metadata["training_epoch_metrics"].append(training_metric_results)

            # calculate the metrics on the validation data
            validation_metric_results = self.get_metrics(
                genome,
                dataloader=self.validation_dataloader,
                loss_function=self.validation_loss_function,
                epoch=epoch,
            )
            logger.info(
                "[epoch {}] validation metrics: {}",
                epoch,
                validation_metric_results,
            )
            genome.metadata["validation_epoch_metrics"].append(
                validation_metric_results
            )

            # TODO: try using the average of validation and training loss for fitness
            validation_loss = validation_metric_results["loss"]
            training_loss = training_metric_results["loss"]

            # scheduler.step(validation_loss)

            avg_loss = (validation_loss + training_loss) / 2.0

            if best_loss > avg_loss:
                best_loss = avg_loss
                best_epoch = epoch

                genome.metadata["best_training_metrics"] = training_metric_results
                genome.metadata["best_validation_metrics"] = validation_metric_results
                genome.metadata["best_epoch"] = best_epoch

                # get a copy of the current state dict of the hybrid model, this will be
                # all the weights
                best_parameters = genome.clone_state_dict()
            elif epoch - best_epoch > improvement_cutoff:
                logger.info(
                    "Stopping at epoch {} because the last improvement "
                    "occurred at epoch {}.",
                    epoch,
                    best_epoch,
                )
                break

        logger.info(
            "Best loss found at epoch {} of {}.",
            best_epoch,
            epochs,
        )

        # set the genome's parameters to the ones from the best validation loss
        genome.set_state_dict(best_parameters)

        return

    def test(
        self,
        genome: CircuitGenome,
    ) -> dict[str, Any]:
        """Evaluates a trained genome on the held-out test set.

        Args:
            genome: Trained genome to evaluate.

        Returns:
            Test loss and configured classification metrics.

        Raises:
            ValueError: If no test dataloader was provided.
        """
        if self.testing_dataloader is None:
            raise ValueError("No testing dataloader was provided to SupervisedTrainer.")

        test_metrics = self.get_metrics(
            genome=genome,
            dataloader=self.testing_dataloader,
            loss_function=self.testing_loss_function,
        )

        logger.info(
            "test metrics: {}",
            test_metrics,
        )

        genome.metadata["testing_metrics"] = test_metrics

        return test_metrics
