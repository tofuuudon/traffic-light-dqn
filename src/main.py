"""Instantiates SUMO with pre-defined network and config for DQN."""

import os
import sys

import traci
from sumolib import checkBinary

from agent import Agent
from _typings import TrafficLightSystem

# Checks for SUMO_HOME enviroment
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

sumoBinary = checkBinary("sumo")
sumoCmd = [sumoBinary, "-W", "-c", "data/train-network/osm.sumocfg"]

STEP = 0
HOURS = 20
START_STATE_PATH = "data/train-network/start.state.xml"
EPISODES = 1
MAX_STEP = 500

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

# Runner
for ep in range(EPISODES):
    for step in range(MAX_STEP):

        for agent in TLS_AGENTS:

            if len(agent.memory) < BATCH_SIZE:
                continue

            agent.prepare_step()

        traci.simulationStep()
    traci.simulation.loadState(START_STATE_PATH)


traci.close()
