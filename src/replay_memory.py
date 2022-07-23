"""Replay memory for handling past experiences."""

from collections import deque
from random import sample

from _typings import Experience


class ReplayMemory:
    """Stores past experiences and handles adding and sampling from memory."""

    def __init__(self, replay_size: int) -> None:
        """__init__.

        Args:
            replay_size (int): replay_size

        Returns:
            None:
        """
        self.memory: deque[Experience] = deque([], maxlen=replay_size)

    def push(self, experience: Experience) -> None:
        """Pushes a new experience into the memory.

        Args:
            experience (Experience): New experience to add.
        """

        self.memory.append(experience)

    def sample(self, batch_size: int) -> list[Experience]:
        """Gets a sample of existing memory with specified size.

        Args:
            batch_size (int): The number of experiences to draw.

        Returns:
            list[Experience]: The list of experiences drawn.
        """

        return sample(self.memory, batch_size)
