"""MLP model for predicting Q-values."""
from typing import Any

from torch.functional import Tensor
from torch.nn.modules.activation import ReLU
from torch.nn.modules.container import Sequential
from torch.nn.modules.linear import Linear
from pytorch_lightning.core.lightning import LightningModule


class PolicyModel(LightningModule):
    """A simple MLP model"""

    def __init__(
        self,
        obs_space: Tensor,
        n_actions: int,
        hidden_layers: int = 128,
    ) -> None:

        super().__init__()

        # Network
        self.net = Sequential(
            Linear(obs_space.shape[0], hidden_layers),
            ReLU(),
            Linear(hidden_layers, n_actions),
        )

    def forward(self, x: Tensor) -> Any:  # type: ignore
        """Computes output tensors.

        Args:
            x (Tensor): Input tensor for computation.

        Returns:
            Tensor: Output tensor.
        """

        return self.net(x.float())
