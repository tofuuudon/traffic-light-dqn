"""MLP model for predicting Q-values."""
from typing import Any

from numpy import float64
from numpy._typing import NDArray
from pytorch_lightning.core.lightning import LightningModule
from torch.functional import Tensor
from torch.nn.modules.activation import ReLU
from torch.nn.modules.container import Sequential
from torch.nn.modules.linear import Linear


class PolicyModel(LightningModule):
    """A simple MLP model"""

    def __init__(
        self,
        obs_space: NDArray[float64],
        n_action: int,
        hidden_layers: int = 128,
    ) -> None:

        super().__init__()

        # Environment
        self.obs_space = obs_space
        self.n_action = n_action

        # Network
        self.net = Sequential(
            Linear(len(obs_space), hidden_layers),
            ReLU(),
            Linear(hidden_layers, n_action),
        )

    def forward(self, x: Tensor) -> Any:  # type: ignore
        """Computes output tensors.

        Args:
            x (Tensor): Input tensor for computation.

        Returns:
            Tensor: Output tensor.
        """

        return self.net(x.float())
