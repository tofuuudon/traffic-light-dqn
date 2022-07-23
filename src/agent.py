"""Deep Q-learning (DQN) agent."""

from typing import Any
from random import choice, random
import xml.etree.ElementTree as ET

import traci

from torch import Tensor, sum as t_sum  # pylint: disable=no-name-in-module

from model import PolicyModel
from replay_memory import ReplayMemory
from _typings import TrafficLightSystem, Experience

ADDI_TREE = ET.parse("data/train-network/osm.additional.xml")
DETECTOR_IDS = [tag.attrib["id"] for tag in ADDI_TREE.findall("e2Detector")]


class Agent:
    """Deep Q-learning (DQN) agent for a traffic lignt node."""

    def __init__(
        self,
        tls_node: TrafficLightSystem,
        alpha: float = 1e-2,
        epsilon_max: float = 0.9,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.001,
        gamma: float = 0.99,
        batch_size: int = 16,
        replay_size: int = 1000,
    ) -> None:
        """Instantiates the object.

        Args:
            gamma (float): gamma
            alpha (float): alpha
            batch_size (int): batch_size
            replay_size (int): replay_size
        """

        super().__init__()

        # Hyperparameters
        self.alpha = alpha
        self.epsilon = epsilon_max
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.batch_size = batch_size

        # Enviroment
        self.tls_node = tls_node

        obs_space = self.__get_state().shape
        n_actions = len(tls_node.phases)

        # Instances
        self.net = PolicyModel(obs_space, n_actions)
        self.target_net = PolicyModel(obs_space, n_actions)
        self.memory = ReplayMemory(replay_size)

    def __get_state(self) -> Tensor:
        """Gets the current environment's state as a tensor.

        Returns:
            Tensor: The current state.
        """

        return Tensor(
            [[traci.lanearea.getJamLengthVehicle(det_id) for det_id in DETECTOR_IDS]]
        )

    def __get_action(self) -> Any:
        """Gets an action from current simulation state."""

        state = self.__get_state()

        if random() < self.epsilon:
            return choice(self.memory.sample(self.batch_size)).action

        return self.net(state)

    def __get_reward(self, state: Tensor) -> float:
        """Gets the reward of the given state.

        Args:
            state (Tensor): The state used to generate a reward value.

        Returns:
            float: The calculated reward.
        """

        return t_sum(state[0]).item()

    def prepare_step(self) -> tuple[Tensor, int]:
        """Prepares the action to take before time step."""

        state = self.__get_state()
        action = self.__get_action()

        # Takes the action
        traci.trafficlight.setPhase(self.tls_node.tls_id, action)

        return (state, action)

    def evaluate_step(self, state: Tensor, action: int) -> None:
        """Evaluates the action after the time step."""

        next_state = self.__get_state()
        reward = self.__get_reward(next_state)

        self.memory.push(Experience(state, action, next_state, reward))

    def update_epsilon(self) -> None:
        """Updates the epsilon (explore-exploit) threshold."""

        self.epsilon = max(self.epsilon_min, self.epsilon_max * self.epsilon_decay)
