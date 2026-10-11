"""Running statistics used to normalize RL observations and scale rewards.

Two pieces of PPO-style input/target conditioning live here:

* :class:`RunningMeanStd` keeps a running per-dimension mean and variance
  (Chan et al.'s parallel update, so a batch of any size can be folded in);
* :class:`ObservationNormalizer` uses one to standardize observations to zero
  mean and unit variance, clipped to ``[-clip, clip]``. Its statistics are part
  of the policy -- the encoder's weights are learned against normalized inputs
  -- so they round-trip through :meth:`RunningMeanStd.to_dict` into genome
  metadata;
* :class:`RewardScaler` divides each reward by the running standard deviation
  of the discounted return, as Stable-Baselines3's ``VecNormalize`` does. It
  only conditions the training targets, so it is rebuilt for every training run
  and never saved.
"""

from __future__ import annotations

from typing import Any

import numpy as np

#: Default bound normalized observations and scaled rewards are clipped to.
DEFAULT_CLIP: float = 10.0

#: Numerical-stability term added to the variance before taking its root.
EPSILON: float = 1e-8


class RunningMeanStd:
    """A running per-dimension mean and variance.

    Attributes:
        mean: The running mean, of the tracked shape.
        var: The running (population) variance, of the tracked shape.
        count: How many samples have been folded in, starting from a tiny
            pseudo-count so the first update is not divided by zero.
    """

    def __init__(self, shape: tuple[int, ...] = (), epsilon: float = 1e-4) -> None:
        """Starts the statistics at mean 0 and variance 1.

        Args:
            shape: The shape of one sample (``()`` for scalars).
            epsilon: Initial pseudo-count.
        """

        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = float(epsilon)

    def update(self, batch: np.ndarray) -> None:
        """Folds a batch of samples into the running statistics.

        Args:
            batch: Samples of shape ``(n, *shape)``.

        Returns:
            None. Updates :attr:`mean`, :attr:`var` and :attr:`count`.
        """

        batch = np.asarray(batch, dtype=np.float64)
        batch_count = batch.shape[0]
        if batch_count == 0:
            return
        batch_mean = batch.mean(axis=0)
        batch_var = batch.var(axis=0)

        delta = batch_mean - self.mean
        total = self.count + batch_count
        m2 = (
            self.var * self.count
            + batch_var * batch_count
            + np.square(delta) * self.count * batch_count / total
        )
        self.mean = self.mean + delta * batch_count / total
        self.var = m2 / total
        self.count = total

    def to_dict(self) -> dict[str, Any]:
        """Serializes the statistics into JSON-friendly lists.

        Returns:
            A dict with ``mean``, ``var`` (lists, or floats for scalars) and
            ``count``.
        """

        return {
            "mean": self.mean.tolist(),
            "var": self.var.tolist(),
            "count": float(self.count),
        }

    def load_dict(self, state: dict[str, Any]) -> None:
        """Restores statistics written by :meth:`to_dict`.

        Args:
            state: A dict with ``mean``, ``var`` and ``count``.

        Returns:
            None. Replaces :attr:`mean`, :attr:`var` and :attr:`count`.
        """

        self.mean = np.asarray(state["mean"], dtype=np.float64)
        self.var = np.asarray(state["var"], dtype=np.float64)
        self.count = float(state["count"])


class ObservationNormalizer(RunningMeanStd):
    """Standardizes observations with running statistics.

    Attributes:
        clip: Normalized values are clipped to ``[-clip, clip]``.
    """

    def __init__(self, n_features: int, clip: float = DEFAULT_CLIP) -> None:
        """Starts an identity normalizer for ``n_features`` inputs.

        Args:
            n_features: Length of one observation vector.
            clip: Bound normalized values are clipped to.
        """

        super().__init__(shape=(n_features,))
        self.clip = float(clip)

    def normalize(self, values: np.ndarray) -> np.ndarray:
        """Standardizes observations with the current statistics.

        Args:
            values: One observation of shape ``(n_features,)`` or a batch of
                shape ``(n, n_features)``.

        Returns:
            ``clip((values - mean) / sqrt(var + eps), -clip, clip)``.
        """

        normalized = (np.asarray(values, dtype=np.float64) - self.mean) / np.sqrt(
            self.var + EPSILON
        )
        return np.clip(normalized, -self.clip, self.clip)

    def to_dict(self) -> dict[str, Any]:
        """Serializes the statistics and clip bound.

        Returns:
            :meth:`RunningMeanStd.to_dict` plus ``clip``.
        """

        state = super().to_dict()
        state["clip"] = self.clip
        return state

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> ObservationNormalizer:
        """Rebuilds a normalizer written by :meth:`to_dict`.

        Args:
            state: A dict with ``mean``, ``var``, ``count`` and optionally
                ``clip``.

        Returns:
            The restored normalizer.
        """

        normalizer = cls(len(state["mean"]), clip=state.get("clip", DEFAULT_CLIP))
        normalizer.load_dict(state)
        return normalizer


class RewardScaler:
    """Scales rewards by the running standard deviation of the return.

    Each reward is divided by the standard deviation of a running discounted
    return (reset at episode ends), which keeps the value targets near unit
    scale whatever the environment's reward magnitude. The mean is not
    subtracted, so the sign of each reward is preserved.

    Attributes:
        gamma: Discount used for the running return.
        clip: Scaled rewards are clipped to ``[-clip, clip]``.
        statistics: Running variance of the discounted return.
        running_return: The discounted return of the current episode so far.
    """

    def __init__(self, gamma: float, clip: float = DEFAULT_CLIP) -> None:
        """Starts the scaler with unit variance.

        Args:
            gamma: Discount used for the running return.
            clip: Bound scaled rewards are clipped to.
        """

        self.gamma = float(gamma)
        self.clip = float(clip)
        self.statistics = RunningMeanStd(shape=())
        self.running_return = 0.0

    def scale(self, reward: float, done: bool) -> float:
        """Updates the return statistics and scales one reward.

        Args:
            reward: The raw environment reward.
            done: Whether this step ended the episode; the running return is
                reset afterwards.

        Returns:
            The scaled, clipped reward.
        """

        self.running_return = self.running_return * self.gamma + float(reward)
        self.statistics.update(np.array([self.running_return]))
        scaled = float(reward) / float(np.sqrt(self.statistics.var + EPSILON))
        if done:
            self.running_return = 0.0
        return float(np.clip(scaled, -self.clip, self.clip))
