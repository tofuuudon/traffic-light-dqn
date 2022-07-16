"""Deep Q-learning (DQN) network."""

from typing import Any
from numpy import array

import traci
from torch.functional import Tensor
from torch.optim import Adam
from pytorch_lightning import LightningModule

from model import PolicyModel
from _typings import TrafficLightSystem


class DQN(LightningModule):
    """DQN model for TLS."""

    def __init__(
        self,
        tls_nodes: tuple[TrafficLightSystem, ...],
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
        self.tls_nodes = tls_nodes

        obs_space = array([tls_nodes[0].lane_ids])
        n_actions = int(len(tls_nodes[0].phases) / 2)  # Half of the phases are yellows

        # Instances
        self.net = PolicyModel(obs_space, n_actions)
        self.target_net = PolicyModel(obs_space, n_actions)

    def training_step(self) -> None:  # type: ignore
        """Actions a single time-step."""

        traci.simulationStep()

    def forward(self, x: Tensor) -> Any:  # type: ignore
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
