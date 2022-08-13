"""MLP model for predicting Q-values."""

from typing import Any

from torch.functional import Tensor
from torch.nn import Linear, ReLU, Module, Sequential
from torch.optim import Adam


class PolicyModel(Module):
    """A simple MLP."""

    def __init__(
        self,
        obs_space: Tensor,
        n_actions: int,
        model_variant: int,
    ) -> None:

        super().__init__()

        # Parameters
        self.obs_space = obs_space
        self.n_actions = n_actions

        # Networks
        if model_variant == 2:
            self.model = Sequential(
                Linear(obs_space.shape[0], 128),
                ReLU(),
                Linear(128, 64),
                ReLU(),
                Linear(64, 32),
                ReLU(),
                Linear(32, n_actions),
            )
        elif model_variant == 1:
            self.model = Sequential(
                Linear(obs_space.shape[0], 128),
                ReLU(),
                Linear(128, 64),
                ReLU(),
                Linear(64, n_actions),
            )
        else:
            self.model = Sequential(
                Linear(obs_space.shape[0], 128),
                ReLU(),
                Linear(128, n_actions),
            )

        self.optimizer = Adam(self.parameters(), lr=1e-3)

    def forward(self, x: Tensor) -> Any:
        """Propagates through the model.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """

        return self.model(x.float())
