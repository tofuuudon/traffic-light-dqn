"""MLP model for predicting Q-values."""

from typing import Any

from torch.functional import Tensor
from torch.nn import Linear, ReLU, Module, Sequential
from torch.optim import Adam


class PolicyModel(Module):
    """A MLP model using LSTM."""

    def __init__(
        self,
        obs_space: Tensor,
        n_actions: int,
        hidden_layers: int = 128,
        num_layers: int = 2,
        learning_rate: float = 1e-3,
    ) -> None:

        super().__init__()

        # Parameters
        self.obs_space = obs_space
        self.n_actions = n_actions
        self.num_layers = num_layers
        self.learning_rate = learning_rate

        # Network
        self.model = Sequential(
            Linear(obs_space.shape[0], hidden_layers),
            ReLU(),
            Linear(hidden_layers, n_actions),
        )

        self.optimizer = Adam(self.parameters(), lr=learning_rate)

    def forward(self, x: Tensor) -> Any:
        """Propagates through the model.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """

        return self.model(x.float())
