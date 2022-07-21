"""Deep Q-learning (DQN) agent."""

from typing import Any
from random import choice, random

import traci
from torch.functional import Tensor

from model import PolicyModel
from replay_memory import ReplayMemory
from _typings import TrafficLightSystem


class Agent:
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
        n_actions = len(node.phases)

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

    # pylint: disable-next=unused-private-member
    def __get_action(self) -> Any:
        """Gets an action from current simulation state."""

        state = self.__get_state()

        if random() < self.epision:
            action = choice(self.memory.sample(self.batch_size)).action
        else:
            action = self.net(state)
        return action

    def prepare_step(self) -> None:
        """Prepares the action to take before time step."""

        # TODO: Gets the current state

        # TODO: Gets the next action

        # TODO: Takes the action

        # TODO: Returns the state-action pair

    def evaluate_step(self) -> None:
        """Evaluates the action after the time step."""

        # TODO: Observe new state

        # TODO: Calculates the reward

        # TODO: Create new experience (state, action, reward, next_state)
