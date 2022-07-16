"""MLP model for predicting Q-values."""

from torch.functional import Tensor
from torch.nn import Module
from torch.nn.modules.activation import ReLU
from torch.nn.modules.container import Sequential
from torch.nn.modules.linear import Linear


class Model(Module):
    """MLP model"""

    def __init__(
        self,
        obs_space: tuple[int, ...],
        action_space: tuple[int, ...],
        hidden_layers: int = 128,
    ) -> None:

        super().__init__()

        # Environment
        self.obs_space = obs_space
        self.action_space = action_space

        # Network
        self.net = Sequential(
            Linear(len(obs_space), hidden_layers),
            ReLU(),
            Linear(hidden_layers, len(action_space)),
        )

    def forward(self, x: Tensor):
        """Computes output tensors.

        Args:
            x (Tensor): Input tensor for computation.

        Returns:
            Any: Output tensor.
        """

        return self.net(x.float())
