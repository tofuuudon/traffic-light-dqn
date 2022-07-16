"""Deep Q-learning (DQN) network."""

from torch.functional import Tensor
from pytorch_lightning import LightningModule
from agent import Agent

from model import Model


class DQN(LightningModule):
    """DQN model for TLS."""

    def __init__(
        self, gamma: float, alpha: float, batch_size: int, replay_size: int
    ) -> None:

        super().__init__()

        # Hyperparameters
        self.gamma = gamma
        self.alpha = alpha
        self.batch_size = batch_size
        self.replay_size = replay_size

        # Enviroment
        # TODO: Use real state and action space
        obs_space = (5,)
        action_space = (5,)

        # Instances
        self.net = Model(obs_space, action_space)
        self.agent = Agent(obs_space, action_space)

    def forward(self, x: Tensor) -> Tensor:
        """Computes output tensors.

        Args:
            x (Tensor): Input tensor for computation.

        Returns:
            Any: Output tensor.
        """

        return self.net(x)
