from collections import deque
from typing import Deque, Tuple
import torch

Transition = Tuple[torch.Tensor, int, float, torch.Tensor, float]  # (s, a, r, s2, done)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf: Deque[Transition] = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buf.append(
            (s.detach().cpu(), int(a), float(r), s2.detach().cpu(), float(done))
        )

    def sample(self, batch_size: int) -> list[Transition]:
        import random

        batch_size = min(batch_size, len(self.buf))
        return random.sample(self.buf, batch_size)

    def __len__(self):
        return len(self.buf)
