"""Deep Q-learning (DQN) agent."""

import xml.etree.ElementTree as ET
from typing import Any
from random import random, randrange
from math import exp

import torch
import traci
from torch.functional import Tensor
from torch.nn import MSELoss
from torch.optim import RMSprop

from model import PolicyModel
from replay_memory import ReplayMemory
from _typings import TrafficLightSystem, Experience, AgentConfig

# XML tree of SUMO's additional file
ADDI_TREE = ET.parse("data/train-network/osm.additional.xml")

# Gets all the detectors' IDs
DETECTOR_IDS = [tag.attrib["id"] for tag in ADDI_TREE.findall("e2Detector")]

# GPU support
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Agent:
    """Deep Q-learning (DQN) agent for a traffic lignt node."""

    def __init__(self, tls_node: TrafficLightSystem, config: AgentConfig) -> None:

        super().__init__()

        # Hyperparameters
        self.epsilon: float = config.epsilon
        self.epsilon_max: float = config.epsilon_max
        self.epsilon_min: float = config.epsilon_min
        self.epsilon_decay: int = config.epsilon_decay
        self.gamma: float = config.gamma
        self.batch_size: int = config.batch_size
        self.replay_size: int = config.replay_size
        self.sync_rate: int = config.sync_rate

        # Enviroment
        self.tls_node = tls_node

        self.obs_space = torch.reshape(self.__get_state(), (-1,))
        self.n_actions = len(tls_node.phases)

        # Instances
        self.net = PolicyModel(self.obs_space, self.n_actions)
        self.target_net = PolicyModel(self.obs_space, self.n_actions)
        self.memory = ReplayMemory(self.replay_size)
        self.optimizer = RMSprop(self.net.parameters())

        self.__update_target_net()

    def __update_target_net(self) -> None:
        """Synchronizes the target network."""

        self.target_net.load_state_dict(self.net.state_dict())

    def __get_state(self) -> Tensor:
        """Gets the current environment's state as a tensor.

        Returns:
            Tensor: The current state.
        """

        return torch.tensor(
            [
                [
                    traci.lanearea.getJamLengthVehicle(det_id)
                    for det_id in DETECTOR_IDS
                    if self.tls_node.tls_id in det_id
                ]
            ]
        )

    def __get_action(self, step: int) -> Any:
        """Gets an action from current simulation state."""

        state = self.__get_state()

        action: Any

        # Exploit and get max Q
        if random() > self.epsilon:
            with torch.no_grad():
                action = self.net(state).max(1)[1].view(1, 1)

        # Explore actions randomly
        action = torch.tensor(
            [[randrange(self.n_actions)]], device=device, dtype=torch.long
        )

        # Adjusts explore-exploit rate
        self.epsilon = self.epsilon_min + (self.epsilon_max - self.epsilon_min) * exp(
            -1.0 * step / self.epsilon_decay
        )

        return action

    def __get_reward(self, state: Tensor) -> Tensor:
        """Gets the reward of the given state.

        Args:
            state (Tensor): The state used to generate a reward value.

        Returns:
            float: The calculated reward.
        """

        return torch.tensor([torch.pow(1.5, -torch.sum(state))])

    def prepare_step(self, step: int) -> tuple[Tensor, int]:
        """Prepares the action to take before time step."""

        state = self.__get_state()
        action = self.__get_action(step)

        # Takes the action
        traci.trafficlight.setPhase(self.tls_node.tls_id, action)

        return (state, action)

    def evaluate_step(self, state: Tensor) -> tuple[Tensor, Tensor]:
        """Evaluates the action after the time step."""

        next_state = self.__get_state()
        reward = self.__get_reward(state)

        return (next_state, reward)

    def train(self, step: int) -> None:
        """Trains the agent using the replay memory.

        Args:
            step (int): Current time step of the simulation.
        """

        # Skips training if there's not enough experience
        if len(self.memory) < self.batch_size:
            return

        # List of all previous experiences
        exps = Experience(*zip(*self.memory.sample(self.batch_size)))

        state_exps = torch.cat(exps.state)
        action_exps = torch.cat(exps.action)
        reward_exps = torch.cat(exps.reward)
        next_states = torch.cat(exps.next_state)

        # Q-values from action action
        sa_values = self.net(state_exps).gather(1, action_exps)

        # Predicted next state q-values
        next_state_values = self.target_net(next_states).max(1)[0].detach()
        expected_sa_values = (next_state_values * self.gamma) + reward_exps

        # Loss function
        criterion = MSELoss()
        loss = criterion(sa_values, expected_sa_values.unsqueeze(1))

        # Update weights and optimize model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # Update target net according to sync rate
        if step % self.sync_rate == 0:
            self.__update_target_net()
