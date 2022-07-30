"""Instantiates SUMO with pre-defined network and config for DQN."""

import os
import sys
from argparse import ArgumentParser

import traci
from sumolib import checkBinary
from torch.utils.tensorboard.writer import SummaryWriter

from agent import Agent
from _typings import TrafficLightSystem, Experience

# CLI parser
parser = ArgumentParser(description="CLI for SUMO RL")
parser.add_argument(
    "-e",
    "--episodes",
    metavar="\b",
    type=int,
    help="The number of episodes to run.",
    default=10,
)
parser.add_argument(
    "-m",
    "--max_step",
    metavar="\b",
    type=int,
    help="The maximum number of steps in each episode.",
    default=3600,
)
parser.add_argument(
    "-g",
    "--gui",
    action="store_true",
    help="Sets SUMO to launch with GUI.",
)
args = parser.parse_args()

print("\n========== Starting Simulation ==========")
print(f"Mode: {'GUI' if args.gui else 'No GUI'}")
print(f"Number of episodes: {args.episodes}")
print(f"Maximum of steps: {args.max_step}\n")

# Checks for SUMO_HOME enviroment
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

# Setup
sumoBinary = checkBinary("sumo-gui" if args.gui else "sumo")
sumoCmd = [sumoBinary, "-W", "-c", "data/train-network/osm.sumocfg"]

START_STATE_PATH = "data/train-network/start.state.xml"

# Hyperparameters
BATCH_SIZE = 32

traci.start(sumoCmd)

if not os.path.exists(START_STATE_PATH):
    traci.simulation.saveState(START_STATE_PATH)

# IDs of all traffic lights
TLS_AGENTS: tuple[Agent, ...] = tuple(
    Agent(
        TrafficLightSystem(
            tls_id,
            traci.trafficlight.getControlledLanes(tls_id),
            traci.trafficlight.getAllProgramLogics(tls_id)[0].phases,
        )
    )
    for tls_id in traci.trafficlight.getIDList()
)

# Tensorboard logger
writter = SummaryWriter()

TOTAL_REWARD: float = 0
for ep in range(args.episodes):
    EPS_REWARD: float = 0
    for step in range(args.max_step):

        sa_pairs = [agent.prepare_step(step) for agent in TLS_AGENTS]

        traci.simulationStep()

        for idx, agent in enumerate(TLS_AGENTS):
            state, action = sa_pairs[idx]
            next_state, reward = agent.evaluate_step(state)
            agent.memory.push(Experience(state, action, next_state, reward))
            agent.train(step)

            EPS_REWARD += reward.reshape(-1)[0].item()

    writter.add_scalar("Episode reward", EPS_REWARD, ep)
    TOTAL_REWARD += EPS_REWARD

    traci.simulation.loadState(START_STATE_PATH)


writter.close()
traci.close()
