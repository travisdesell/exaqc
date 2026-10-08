"""The quantum-teacher imitation task: its objective and genome ordering.

Shared by the ``src.examples.teacher`` search and by the single-genome tools
(``src.examples.refine_genome``). It lives outside the entry point so the tools
can build a teacher objective without importing the search driver, and with it
MPI.
"""

from __future__ import annotations

from torch.utils.data import DataLoader

from src.circuits.circuit import CircuitGenome
from src.evolution.objective import Objective
from src.metrics.teacher_losses import get_teacher_loss
from src.metrics.teacher_metrics import build_teacher_metrics
from src.trainer.supervised_trainer import SupervisedTrainer


def compare(
    genome1: CircuitGenome,
    genome2: CircuitGenome,
) -> int:
    """Compares genomes using the minimized loss objective.

    Args:
        genome1: First genome.
        genome2: Second genome.

    Returns:
        Negative when ``genome1`` is better, positive when ``genome2`` is
        better, and zero when they are equal.
    """

    return genome1.fitness["loss"] - genome2.fitness["loss"]


class TeacherObjective(Objective):
    """Teacher-imitation objective backed by :class:`SupervisedTrainer`."""

    def __init__(
        self,
        training_dataloader: DataLoader,
        validation_dataloader: DataLoader,
        loss_name: str,
        device: str | None = None,
    ) -> None:
        """Initializes the teacher-imitation objective.

        Args:
            training_dataloader: Generated training loader.
            validation_dataloader: Generated validation loader.
            loss_name: Which measure to optimize; one of
                :data:`~src.metrics.teacher_losses.TEACHER_LOSS_NAMES`.
            device: PyTorch device to train on, or ``None`` to auto-select.

        Returns:
            None. Sets ``loss_name`` and ``trainer``.
        """

        self.loss_name = loss_name
        loss_function = get_teacher_loss(loss_name)

        # Every measure is reported each epoch, whichever one is optimized, so
        # runs using different losses stay comparable after the fact.
        self.trainer = SupervisedTrainer(
            training_dataloader=training_dataloader,
            validation_dataloader=validation_dataloader,
            training_loss_function=loss_function,
            validation_loss_function=loss_function,
            metrics=build_teacher_metrics(),
            device=device,
        )

    def __call__(self, genome: CircuitGenome) -> None:
        """Trains a genome and assigns teacher-imitation fitness.

        Args:
            genome: Genome to train and evaluate.

        Returns:
            None. Sets ``genome.fitness`` with a minimized ``"loss"`` (the mean
            of the best training and validation loss) and a ``"target_metric"``
            holding the corresponding mean fidelity, matching the fitness keys
            the classification objective writes so the analysis tooling reads
            both the same way.
        """

        self.trainer.train(genome)

        training = genome.metadata["best_training_metrics"]
        validation = genome.metadata["best_validation_metrics"]

        genome.fitness = {
            "loss": (float(training["loss"]) + float(validation["loss"])) / 2.0,
            "target_metric": (
                float(training["fidelity"]["mean"])
                + float(validation["fidelity"]["mean"])
            )
            / 2.0,
        }
