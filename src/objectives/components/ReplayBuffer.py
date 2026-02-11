from typing import Deque
import torch
import random

Transition = tuple[torch.Tensor, int, float, torch.Tensor, float]  # (s, a, r, s2, done)


class ReplayBuffer:
    '''
    Replay Buffer Class to store previous state-action-reward observations
    '''
    def __init__(self, capacity: int):
        self.buf: Deque[Transition] = Deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buf.append(
            (s.detach().cpu(), int(a), float(r), s2.detach().cpu(), float(done))
        )

    def sample(self, batch_size: int) -> list[Transition]:
        batch_size = min(batch_size, len(self.buf))
        return random.sample(self.buf, batch_size)

    def __len__(self):
        return len(self.buf)
