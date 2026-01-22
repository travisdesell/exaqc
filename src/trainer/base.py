from abc import ABC, abstractmethod


class Trainer(ABC):

    @abstractmethod
    def train(self, circuit, params, losses):
        """
        Returns:
            trained_params, loss_history
        """
        pass


class PennyLaneTrainer(Trainer):

    def __init__(self, optimizer, steps=100):
        self.optimizer = optimizer
        self.steps = steps

    def train(self, qnode_tuple, params, losses):
        _, qnode = qnode_tuple
        loss_history = {k: [] for k in losses}

        for _ in range(self.steps):

            def total_loss(p):
                return sum(loss.compute(qnode, p) for loss in losses.values())

            params = self.optimizer.step(total_loss, params)

            for name, loss in losses.items():
                loss_history[name].append(loss.compute(qnode, params))

        return params, loss_history


class QiskitTrainer(Trainer):

    def __init__(self, optimizer):
        self.optimizer = optimizer

    def train(self, circuit, params, losses):

        def objective(p):
            # bind parameters → circuit
            bound = circuit.assign_parameters(p)
            return sum(loss.compute(bound) for loss in losses.values())

        result = self.optimizer.minimize(
            fun=objective,
            x0=params,
        )

        loss_history = {k: [result.fun] for k in losses}
        return result.x, loss_history
