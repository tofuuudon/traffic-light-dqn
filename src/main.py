"""Instantiates SUMO with pre-defined network and config for DQN."""

import os
import sys

import traci
from sumolib import checkBinary

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
N_STEP = 1

traci.start(sumoCmd)

# IDs of all traffic lights
TLS_NODES: list[TrafficLightSystem] = [
    TrafficLightSystem(id, traci.trafficlight.getControlledLanes(id))
    for id in traci.trafficlight.getIDList()
]

# Runtime
while STEP <= N_STEP:
    traci.simulationStep()

    STEP += 1

traci.close()
