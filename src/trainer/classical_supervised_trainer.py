from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from torch import Tensor
from torch.utils.data import DataLoader

from src.metrics.mean_class_accuracy import MeanClassAccuracy


class ClassicalSupervisedTrainer:
    """Trains classical PyTorch classification baselines."""

    def __init__(
        self,
        training_dataloader: DataLoader,
        validation_dataloader: DataLoader,
        testing_dataloader: DataLoader | None = None,
        device: str | None = None,
    ) -> None:
        """Initializes the classical classification trainer.

        Args:
            training_dataloader: Batched training dataloader.
            validation_dataloader: Batched validation dataloader.
            testing_dataloader: Optional held-out test dataloader.
            device: PyTorch device string. CUDA is selected automatically
                when available if not provided.
        """
        self.training_dataloader = training_dataloader
        self.validation_dataloader = validation_dataloader
        self.testing_dataloader = testing_dataloader

        self.device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        logger.info(
            "ClassicalSupervisedTrainer using device: {}",
            self.device,
        )

    def _evaluate_loader(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        loss_function: Callable[[Tensor, Tensor], Tensor],
    ) -> dict[str, Any]:
        """Evaluates a model on a dataloader.

        Args:
            model: Model to evaluate.
            dataloader: Dataloader to evaluate on.
            loss_function: Classification loss function.

        Returns:
            Mean loss and mean-class-accuracy metrics.
        """
        model.eval()
        metric = MeanClassAccuracy(n_labels=dataloader.n_labels)

        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for x_batch, y_batch in dataloader:
                x_batch = x_batch.to(
                    self.device,
                    non_blocking=True,
                )
                y_batch = y_batch.to(
                    self.device,
                    non_blocking=True,
                )

                predictions = model(x_batch)
                loss = loss_function(
                    predictions.float(),
                    y_batch.long(),
                )

                batch_size = int(y_batch.shape[0])
                total_loss += float(loss.item()) * batch_size
                total_samples += batch_size

                metric.accumulate(
                    predictions.detach().cpu(),
                    y_batch.detach().cpu(),
                )

        return {
            "loss": (total_loss / total_samples if total_samples else float("nan")),
            "mean_class_accuracy": metric.calculate(),
        }

    def train(
        self,
        model: torch.nn.Module,
        epochs: int,
        learning_rate: float,
        weight_decay: float = 0.0,
        improvement_cutoff: int = 5,
        label_smoothing: float = 0.0,
    ) -> dict[str, Any]:
        """Trains a classical classifier with validation early stopping.

        Args:
            model: PyTorch classifier.
            epochs: Maximum number of training epochs.
            learning_rate: Adam learning rate.
            weight_decay: Adam weight decay.
            improvement_cutoff: Number of epochs without validation-loss
                improvement before stopping.
            label_smoothing: Cross-entropy label smoothing used during
                training only.

        Returns:
            Dictionary containing training history and final metrics.
        """
        model.to(self.device)

        training_loss = torch.nn.CrossEntropyLoss(
            weight=self.training_dataloader.label_weights.to(self.device),
            reduction="mean",
            label_smoothing=label_smoothing,
        )
        validation_loss = torch.nn.CrossEntropyLoss(
            weight=self.validation_dataloader.label_weights.to(self.device),
            reduction="mean",
        )

        n_trainable_parameters = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )

        logger.debug(f"hybrid model n trainable parameters: {n_trainable_parameters}")

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        history: dict[str, list[dict[str, Any]]] = {
            "training": [],
            "validation": [],
        }

        best_validation_loss = math.inf
        best_epoch = 0
        best_state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
        }

        for epoch in range(epochs):
            model.train()

            train_metric = MeanClassAccuracy(n_labels=self.training_dataloader.n_labels)
            train_loss_sum = 0.0
            train_samples = 0

            for x_batch, y_batch in self.training_dataloader:
                x_batch = x_batch.to(
                    self.device,
                    non_blocking=True,
                )
                y_batch = y_batch.to(
                    self.device,
                    non_blocking=True,
                )

                optimizer.zero_grad(set_to_none=True)

                predictions = model(x_batch)
                loss = training_loss(
                    predictions.float(),
                    y_batch.long(),
                )

                loss.backward()
                optimizer.step()

                batch_size = int(y_batch.shape[0])
                train_loss_sum += float(loss.detach().item()) * batch_size
                train_samples += batch_size

                train_metric.accumulate(
                    predictions.detach().cpu(),
                    y_batch.detach().cpu(),
                )

            training_metrics = {
                "epoch": epoch,
                "loss": train_loss_sum / train_samples,
                "mean_class_accuracy": train_metric.calculate(),
            }

            validation_metrics = self._evaluate_loader(
                model,
                self.validation_dataloader,
                validation_loss,
            )
            validation_metrics["epoch"] = epoch

            history["training"].append(training_metrics)
            history["validation"].append(validation_metrics)

            logger.info(
                "[epoch {}] training metrics: {}",
                epoch,
                training_metrics,
            )
            logger.info(
                "[epoch {}] validation metrics: {}",
                epoch,
                validation_metrics,
            )

            current_validation_loss = float(validation_metrics["loss"])

            if current_validation_loss < best_validation_loss:
                best_validation_loss = current_validation_loss
                best_epoch = epoch
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.state_dict().items()
                }
            elif epoch - best_epoch > improvement_cutoff:
                logger.info(
                    "Stopping at epoch {} because the last validation "
                    "improvement occurred at epoch {}.",
                    epoch,
                    best_epoch,
                )
                break

        model.load_state_dict(best_state)
        model.to(self.device)

        final_training_metrics = self._evaluate_loader(
            model,
            self.training_dataloader,
            torch.nn.CrossEntropyLoss(
                weight=self.training_dataloader.label_weights.to(self.device),
                reduction="mean",
            ),
        )
        final_validation_metrics = self._evaluate_loader(
            model,
            self.validation_dataloader,
            validation_loss,
        )

        results: dict[str, Any] = {
            "parameters": n_trainable_parameters,
            "best_epoch": best_epoch,
            "training_metrics": final_training_metrics,
            "validation_metrics": final_validation_metrics,
            "history": history,
        }

        if self.testing_dataloader is not None:
            testing_loss = torch.nn.CrossEntropyLoss(
                weight=self.testing_dataloader.label_weights.to(self.device),
                reduction="mean",
            )
            results["testing_metrics"] = self._evaluate_loader(
                model,
                self.testing_dataloader,
                testing_loss,
            )

        return results

    @staticmethod
    def save_checkpoint(
        model: torch.nn.Module,
        output_path: str | Path,
    ) -> None:
        """Saves a model state dictionary on CPU.

        Args:
            model: Model to save.
            output_path: Checkpoint path.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        state_dict = {
            name: tensor.detach().cpu() for name, tensor in model.state_dict().items()
        }
        torch.save(state_dict, output_path)
