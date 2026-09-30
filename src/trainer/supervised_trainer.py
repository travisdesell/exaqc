from __future__ import annotations

import argparse
from collections.abc import Callable
from typing import Any

import torch
from loguru import logger
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from src.circuits.circuit import CircuitGenome
from src.dropout.quantum_dropout import sample_quantum_dropout


class SupervisedTrainer:
    """Trains EXAQC hybrid models using a fully batched execution path.

    The trainer is task-agnostic: it drives ``genome.forward`` over the supplied
    dataloaders and hands each batch's predictions and targets to the caller's
    loss function and metrics. Targets reach the loss function exactly as the
    dataloader yielded them, so the same trainer serves classification (integer
    class indices with cross-entropy) and regression-style tasks such as
    quantum-teacher imitation (float target vectors with an MSE/KL/fidelity
    loss).
    """

    @staticmethod
    def initialize_parser(parser: argparse.ArgumentParser) -> None:
        """Adds the supervised-training command-line arguments to a parser.

        The classification and teacher entry points both train genomes with a
        :class:`SupervisedTrainer`, so they share the same training-loop knobs.
        Registering them here (mirroring
        :meth:`~src.evolution.exaqc.EXAQC.initialize_parser` and
        :meth:`~src.circuits.circuit.CircuitGenome.initialize_parser`) keeps the
        two entry points in sync. The values become per-genome hyperparameters
        the trainer reads at train time.

        Args:
            parser: The parser to add the arguments to.

        Returns:
            None. Mutates ``parser`` by adding ``--epochs``,
            ``--learning_rate``/``-lr``, ``--weight_decay``,
            ``--improvement_cutoff`` and ``--batch_size``.
        """

        parser.add_argument(
            "--epochs",
            type=int,
            default=30,
            help=(
                "Maximum number of training epochs per genome, after the epoch-0 "
                "evaluation of its inherited weights; 0 scores the inherited weights only."
            ),
        )

        parser.add_argument(
            "--learning_rate",
            "-lr",
            type=float,
            default=5e-3,
            help="Adam learning rate used when training each genome.",
        )

        parser.add_argument(
            "--weight_decay",
            type=float,
            default=0.0,
            help="Adam weight decay (L2 regularization) used when training each genome.",
        )

        parser.add_argument(
            "--improvement_cutoff",
            type=int,
            default=3,
            help="Stop training a genome after this many epochs without validation improvement.",
        )

        parser.add_argument(
            "--batch_size",
            type=int,
            default=5,
            help="Training batch size.",
        )

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

        Note:
            Quantum dropout is controlled per genome via its
            ``quantum_dropout`` hyperparameter (read from
            ``genome.hyperparameters`` at train time by :meth:`get_metrics`),
            not by this trainer -- so the evolutionary search can carry and
            mutate it per genome. When the ``quantum_dropout`` hyperparameter is
            falsy (the default) no quantum dropout is ever applied, regardless
            of the ``quantum_dropout_type``/``quantum_dropout_rate``
            hyperparameters; when truthy, dropout is sampled and applied per
            training batch according to those hyperparameters.
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
            loss_function: the loss function to use for data evaluation. It is
                called as ``loss_function(predictions, targets)`` with the
                targets exactly as the dataloader yielded them, so it owns the
                target dtype contract (integer class indices for cross-entropy,
                float target vectors for regression-style tasks).
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
                ``[batch_size, n_outputs]``, or if the prediction and target
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
                    # Quantum dropout is a per-genome hyperparameter (so the
                    # evolutionary search can carry/mutate it), read from the
                    # genome here rather than from trainer-level state.
                    if genome.hyperparameters.get("quantum_dropout", False):
                        sample_quantum_dropout(genome)
                    else:
                        # Train on the complete evolved circuit; clear any stale
                        # dropout state.
                        genome.clear_quantum_dropout()
                else:
                    # Validation/test always uses the complete evolved circuit.
                    genome.clear_quantum_dropout()

                predictions = genome.forward(x_batch)

                if predictions.ndim != 2:
                    raise ValueError(
                        "Predictions must have shape [batch_size, n_outputs], "
                        f"received {tuple(predictions.shape)}."
                    )
                if predictions.shape[0] != y_batch.shape[0]:
                    raise ValueError(
                        "Prediction and target batch sizes differ: "
                        f"{predictions.shape[0]} != {y_batch.shape[0]}."
                    )

                # Targets are passed through untouched so the loss function owns
                # the dtype contract: classification supplies integer class
                # indices for cross-entropy, while regression-style tasks (e.g.
                # quantum-teacher imitation) supply float target vectors.
                loss = loss_function(predictions.float(), y_batch)

                # A parameterized gate can be disabled (structurally) or dropped
                # (transiently, by quantum dropout) so that no enabled gate uses
                # the circuit weights on this forward pass. When that happens the
                # loss is disconnected from every trainable parameter and has no
                # grad_fn, so calling backward() would raise. Skip the update for
                # such batches; the metrics below are still accumulated.
                if is_training and loss.requires_grad:
                    loss.backward()
                    optimizer.step()

                current_batch_size = int(y_batch.shape[0])
                total_loss += float(loss.detach().item()) * current_batch_size
                total_samples += current_batch_size

                with torch.no_grad():
                    for prediction, target in zip(predictions, y_batch):
                        for metric in self.metrics.values():
                            # As with the loss, targets reach the metric exactly
                            # as the dataloader yielded them, so each metric owns
                            # its own target contract (class indices for accuracy,
                            # float target vectors for fidelity/KL/MSE).
                            metric.accumulate(prediction.float(), target)

        genome.clear_quantum_dropout()

        metric_results: dict[str, Any] = {
            "loss": (total_loss / total_samples if total_samples else float("nan"))
        }
        for metric_name, metric in self.metrics.items():
            metric_results[metric_name] = metric.calculate()

        if epoch is not None:
            metric_results["epoch"] = epoch
        return metric_results

    def train(self, genome: CircuitGenome) -> None:
        """Evaluates a genome's inherited weights, then trains it.

        Epoch 0 is always a pre-training evaluation of the weights the genome
        inherited, so the per-epoch history shows whether training improved on
        them. Training epochs are then numbered ``1`` to ``epochs`` inclusive,
        and whichever epoch -- including epoch 0 -- has the lowest mean of
        training and validation loss is kept. A genome with no trainable
        parameters, or an ``epochs`` hyperparameter of 0, is only evaluated, so
        its fitness comes from its inherited weights.

        Only the validation history has an epoch 0: no training has happened
        before the first epoch, so ``training_epoch_metrics`` starts at epoch 1.
        The training data is still evaluated at epoch 0 (so epoch 0 can be
        compared against the training epochs), and that evaluation becomes
        ``best_training_metrics`` if the inherited weights win.

        Args:
            genome: The CircuitGenome to train. Its ``hybrid_model`` is built
                here (via ``genome.initialize_model()``) from the genome's
                inherited parameters.

        Returns:
            None. Sets ``genome.metadata`` entries ``training_epoch_metrics``
            (epochs ``1..epochs``), ``validation_epoch_metrics`` (epochs
            ``0..epochs``), ``n_trainable_parameters``,
            ``best_training_metrics``, ``best_validation_metrics`` and
            ``best_epoch`` (0 when the inherited weights were never improved
            on), and leaves the genome holding the best epoch's weights.
        """
        genome.initialize_model()
        genome.hybrid_model.to(self.device)

        hyperparameters = genome.hyperparameters
        learning_rate = float(hyperparameters["learning_rate"])
        epochs = int(hyperparameters["epochs"])

        # initalize the epoch metrics for tracking/data mining
        genome.metadata["training_epoch_metrics"] = []
        genome.metadata["validation_epoch_metrics"] = []

        # what the optimizer actually updates (encoder/decoder weights and enabled
        # gates' parameters), recorded the same way by every trainer
        n_trainable_parameters = genome.count_trainable_parameters()
        genome.metadata["n_trainable_parameters"] = n_trainable_parameters

        logger.debug(f"hybrid model n trainable parameters: {n_trainable_parameters}")

        # epoch 0: evaluate the inherited weights before any training. only the
        # validation metrics are recorded in the per-epoch history, since there
        # has been no training yet; the training data evaluation is kept so the
        # inherited weights can be compared with (and win over) trained ones.
        initial_training_metrics = self.get_metrics(
            genome,
            dataloader=self.training_dataloader,
            loss_function=self.training_loss_function,
            epoch=0,
        )
        initial_validation_metrics = self.get_metrics(
            genome,
            dataloader=self.validation_dataloader,
            loss_function=self.validation_loss_function,
            epoch=0,
        )
        logger.info(
            "[epoch 0] pre-training validation metrics: {}",
            initial_validation_metrics,
        )
        genome.metadata["validation_epoch_metrics"].append(initial_validation_metrics)

        best_loss = (
            initial_training_metrics["loss"] + initial_validation_metrics["loss"]
        ) / 2.0
        best_epoch = 0
        genome.metadata["best_training_metrics"] = initial_training_metrics
        genome.metadata["best_validation_metrics"] = initial_validation_metrics
        genome.metadata["best_epoch"] = best_epoch

        if n_trainable_parameters == 0 or epochs <= 0:
            # nothing connected to the loss can be trained, or no training was
            # asked for: the genome's fitness is that of its inherited weights
            if n_trainable_parameters == 0:
                logger.info(
                    "Model has no trainable (enabled) parameters; evaluating only."
                )
            else:
                logger.info("Training for 0 epochs; evaluating only.")
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

        improvement_cutoff = int(hyperparameters.get("improvement_cutoff", 2))
        best_parameters = genome.clone_state_dict()

        # epoch 0 was the pre-training evaluation, so training runs 1..epochs
        for epoch in range(1, epochs + 1):
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
            elif improvement_cutoff > 0 and epoch - best_epoch > improvement_cutoff:
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

        # set the genome's parameters to the ones from the best epoch (the
        # inherited ones if training never improved on them)
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
