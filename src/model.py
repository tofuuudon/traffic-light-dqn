"""MLP model for predicting Q-values."""
from typing import Any

from torch.functional import Tensor
from torch.nn import Linear, ReLU, Sequential, Module
from torch.optim import Adam


class PolicyModel(Module):
    """A simple MLP model"""

    def __init__(
        self,
        obs_space: Tensor,
        n_actions: int,
        hidden_layers: int = 128,
        learning_rate: float = 1e-3,
    ) -> None:

        super().__init__()

        # Network
        self.net = Sequential(
            Linear(obs_space.shape[0], hidden_layers),
            ReLU(),
            Linear(hidden_layers, n_actions),
        )

        self.optimizer = Adam(self.parameters(), lr=learning_rate)

    def forward(self, x: Tensor) -> Any:
        """Computes output tensors.

        Args:
            x (Tensor): Input tensor for computation.

        Returns:
            Tensor: Output tensor.
        """

        return self.net(x.float())
