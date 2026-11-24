import torch

from torch import Tensor

from src.metrics.metric import Metric


class MeanClassAccuracy(Metric):
    """
    Instead of calculating an overall loss across all samples, this will
    calculate the overall loss for each class and then provide a mean
    of that loss.
    """

    def __init__(self, n_labels: int):
        """
        Initializes the mean class accuracy calculator.

        Args:
            n_labels: is the number of possible output labels for the
                learning task.
        """
        self.n_labels = n_labels

        self.target_correct = {}
        self.target_total = {}

        self.reset()

    def reset(self):
        """
        Resets the metric calculator to a zeroed out state for the
        beginning of a new epoch.
        """

        for i in range(self.n_labels):
            self.target_correct[i] = 0
            self.target_total[i] = 0

    def accumulate(self, output: Tensor, target: Tensor):
        """
        Accumulates the number of correct predictions for each
        target, as well as the total number of samples for each
        target so the overall accuracy for each can be accumulated.

        Args:
            outputs: is a tensor of length equal to the number of classes
                in the dataset. the index of the largest value will be
                used as the predicted label.
            target: should be a single integer value which is the target
                class index.
        """

        # predicted = int(torch.argmax(output).item())
        # target = int(target.item())

        # self.target_total[target] += 1
        # self.target_correct[target] += predicted == target

        if output.ndim == 1:
            output = output.unsqueeze(0)

        if target.ndim == 0:
            target = target.unsqueeze(0)

        if output.shape[0] != target.shape[0]:
            raise ValueError(
                "Output and target batch sizes must match: "
                f"{output.shape[0]} != {target.shape[0]}."
            )

        predicted = torch.argmax(output, dim=1)

        for predicted_label, target_label in zip(predicted, target):
            predicted_value = int(predicted_label.item())
            target_value = int(target_label.item())

            self.target_total[target_value] += 1
            self.target_correct[target_value] += predicted_value == target_value

    def calculate(self) -> any:
        """
        Calculates the per class and mean class accuracy for each target in the
        dataset.

        Returns:
            A dict with entries for the mean and per class accuracies.
        """

        accuracy_metric = {}

        mean_accuracy = 0

        for target in self.target_total.keys():
            correct = self.target_correct[target]
            total = self.target_total[target]

            target_accuracy = float(correct) / float(total)

            accuracy_metric[target] = {
                "acc": target_accuracy,
                "correct": correct,
                "total": total,
            }

            mean_accuracy += target_accuracy

        mean_accuracy /= len(self.target_total)

        accuracy_metric["mean"] = mean_accuracy

        return accuracy_metric
