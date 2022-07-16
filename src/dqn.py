"""Deep Q-learning (DQN) network."""

from torch.functional import Tensor
from torch.optim import Adam
from pytorch_lightning import LightningModule
from agent import Agent

from model import Model


class DQN(LightningModule):
    """DQN model for TLS."""

    def __init__(
        self,
        gamma: float = 0.99,
        alpha: float = 1e-2,
        batch_size: int = 16,
        replay_size: int = 1000,
    ) -> None:
        """__init__.

        Args:
            gamma (float): gamma
            alpha (float): alpha
            batch_size (int): batch_size
            replay_size (int): replay_size

        Returns:
            None:
        """

        super().__init__()

        # Hyperparameters
        self.gamma = gamma
        self.alpha = alpha
        self.batch_size = batch_size
        self.replay_size = replay_size

        # Enviroment
        obs_space = (5,)
        action_space = (5,)

        # Instances
        self.net = Model(obs_space, action_space)
        self.agent = Agent(obs_space, action_space)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        """Computes output tensors.

        Args:
            x (Tensor): Input tensor for computation.

        Returns:
            Tensor: Output tensor.
        """

        return self.net(x)

    def configure_optimizers(self) -> Adam:
        """Uses Adam optimizer to find gradient.

        Returns:
            Adam: The optimizer.
        """

        return Adam(self.net.parameters(), self.alpha)
