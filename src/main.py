"""Instantiates SUMO with pre-defined network and config for DQN."""

import os
import sys

import traci
from sumolib import checkBinary

from agent import Agent
from _typings import TrafficLightSystem, Experience

# Checks for SUMO_HOME enviroment
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

sumoBinary = checkBinary("sumo")
sumoCmd = [sumoBinary, "-W", "-c", "data/train-network/osm.sumocfg"]

START_STATE_PATH = "data/train-network/start.state.xml"
EPISODES = 10
MAX_STEP = 1000

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

TOTAL_REWARD: float = 0
for ep in range(EPISODES):
    EPS_REWARD: float = 0
    for step in range(MAX_STEP):

        sa_pairs = [agent.prepare_step(step) for agent in TLS_AGENTS]

        traci.simulationStep()

        for idx, agent in enumerate(TLS_AGENTS):
            state, action = sa_pairs[idx]
            next_state, reward = agent.evaluate_step(state)
            agent.memory.push(Experience(state, action, next_state, reward))
            agent.train(step)

            EPS_REWARD += reward.reshape(-1)[0].item()

    TOTAL_REWARD += EPS_REWARD

    traci.simulation.loadState(START_STATE_PATH)


traci.close()
