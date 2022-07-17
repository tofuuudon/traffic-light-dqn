"""Deep Q-learning (DQN) agent."""

from typing import Any
from random import choice, random

import traci
from torch.functional import Tensor

from model import PolicyModel
from replay_memory import ReplayMemory
from _typings import TrafficLightSystem


class DQN:
    """DQN model for TLS."""

    def __init__(
        self,
        node: TrafficLightSystem,
        alpha: float = 1e-2,
        epsilon: float = 0.5,
        gamma: float = 0.99,
        batch_size: int = 16,
        replay_size: int = 1000,
    ) -> None:
        """__init__.

        Args:
            gamma (float): gamma
            alpha (float): alpha
            batch_size (int): batch_size
            replay_size (int): replay_size
        """

        super().__init__()

        # Hyperparameters
        self.alpha = alpha
        self.epision = epsilon
        self.gamma = gamma
        self.batch_size = batch_size

        # Enviroment
        self.node = node

        obs_space = self.__get_state().shape
        n_actions = int(len(node.phases) / 2)  # Half of the phases are yellows

        # Instances
        self.net = PolicyModel(obs_space, n_actions)
        self.target_net = PolicyModel(obs_space, n_actions)
        self.memory = ReplayMemory(replay_size)

    def __get_state(self) -> Tensor:

        return Tensor(
            [
                [
                    traci.lanearea.getJamLengthVehicle(f"{self.node.tls_id}-{lane_id}")
                    for lane_id in self.node.lane_ids
                ]
            ]
        )

    def __get_action(self) -> Any:
        """Gets an action from current simulation state."""

        state = self.__get_state()

        if random() < self.epision:
            action = choice(self.memory.sample(self.batch_size)).action
        else:
            action = self.net(state)
        return action

    def step(self) -> None:
        """Simulation step."""

        if len(self.memory) < self.batch_size:
            return

        self.__get_action()
