"""MLP model for predicting Q-values."""

from typing import Any

import torch
from torch.functional import Tensor
from torch.nn import LSTM, Linear, ReLU, Module
from torch.optim import Adam
from torch.autograd import Variable


class PolicyModel(Module):
    """A MLP model using LSTM."""

    def __init__(
        self,
        obs_space: Tensor,
        n_actions: int,
        hidden_size: int = 128,
        num_layers: int = 1,
        learning_rate: float = 1e-3,
    ) -> None:

        super().__init__()

        # Parameters
        self.obs_space = obs_space
        self.n_actions = n_actions
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate

        # Network
        self.lstm: LSTM = LSTM(
            input_size=obs_space.shape[1],
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc_1 = Linear(hidden_size, 128)
        self.fc_2 = Linear(128, n_actions)
        self.relu = ReLU()

        self.optimizer = Adam(self.parameters(), lr=learning_rate)

    def forward(self, x: Tensor) -> Any:
        """Propagates through the model.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """

        h_0 = Variable(torch.zeros(self.num_layers, self.hidden_size))
        c_0 = Variable(torch.zeros(self.num_layers, self.hidden_size))

        output, _ = self.lstm(x.type(torch.FloatTensor), (h_0, c_0))  # type: ignore
        out = output.view(-1, self.hidden_size)
        out = self.relu(out)
        out = self.fc_1(out)
        out = self.relu(out)
        out = self.fc_2(out)

        return out
