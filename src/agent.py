"""Deep Q-learning (DQN) agent."""

from typing import Any
from random import random, randrange
import xml.etree.ElementTree as ET

import torch
import traci
from torch.functional import Tensor
from torch.nn import SmoothL1Loss
from torch.optim import RMSprop

from model import PolicyModel
from replay_memory import ReplayMemory
from _typings import TrafficLightSystem, Experience

ADDI_TREE = ET.parse("data/train-network/osm.additional.xml")
DETECTOR_IDS = [tag.attrib["id"] for tag in ADDI_TREE.findall("e2Detector")]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Agent:
    """Deep Q-learning (DQN) agent for a traffic lignt node."""

    def __init__(
        self,
        tls_node: TrafficLightSystem,
        alpha: float = 1e-2,
        epsilon_max: float = 0.99,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.001,
        gamma: float = 0.99,
        batch_size: int = 16,
        replay_size: int = 1000,
        sync_rate: int = 10,
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
        self.sync_rate = sync_rate

        # Enviroment
        self.tls_node = tls_node

        self.obs_space = torch.reshape(self.__get_state(), (-1,))
        self.n_actions = len(tls_node.phases)

        # Instances
        self.net = PolicyModel(self.obs_space, self.n_actions)
        self.target_net = PolicyModel(self.obs_space, self.n_actions)
        self.memory = ReplayMemory(replay_size)
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

    def __get_action(self) -> Any:
        """Gets an action from current simulation state."""

        state = self.__get_state()

        if random() > self.epsilon:
            with torch.no_grad():
                return self.net(state).max(1)[1].view(1, 1)

        return torch.tensor(
            [[randrange(self.n_actions)]], device=device, dtype=torch.long
        )

    def __get_reward(self, state: Tensor, next_state: Tensor) -> Tensor:
        """Gets the reward of the given state.

        Args:
            state (Tensor): The state used to generate a reward value.

        Returns:
            float: The calculated reward.
        """

        return torch.tensor([[-torch.sum(state[0] - next_state[0]).item()]])

    def prepare_step(self) -> tuple[Tensor, int]:
        """Prepares the action to take before time step."""

        state = self.__get_state()
        action = self.__get_action()

        # Takes the action
        traci.trafficlight.setPhase(self.tls_node.tls_id, action)

        return (state, action)

    def evaluate_step(self, state: Tensor) -> tuple[Tensor, Tensor]:
        """Evaluates the action after the time step."""

        next_state = self.__get_state()
        reward = self.__get_reward(state, next_state)

        return (next_state, reward)

    def train(self, step: int) -> None:
        """Trains the agent using the replay memory.

        Args:
            step (int): Current time step of the simulation.
        """

        if len(self.memory) < self.batch_size:
            return

        exps = Experience(*zip(*self.memory.sample(self.batch_size)))

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, exps.next_state)),
            device=device,
            dtype=torch.bool,
        )
        non_final_next_states = torch.cat([s for s in exps.next_state if s is not None])

        state_exps = torch.cat(exps.state)
        action_exps = torch.cat(exps.action)
        reward_exps = torch.cat(exps.reward)

        sa_values = self.net(state_exps).gather(1, action_exps)

        next_state_values = torch.zeros(self.batch_size, device=device)
        next_state_values[non_final_mask] = (
            self.target_net(non_final_next_states).max(1)[0].detach()
        )
        expected_sa_values = (next_state_values * self.gamma) + reward_exps

        criterion = SmoothL1Loss()
        loss = criterion(sa_values, expected_sa_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        if step % self.sync_rate == 0:
            self.__update_target_net()
