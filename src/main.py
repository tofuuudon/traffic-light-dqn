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
HOURS = 20
START_STATE_PATH = "data/train-network/start.state.xml"

traci.start(sumoCmd)

if not os.path.exists(START_STATE_PATH):
    traci.simulation.saveState(START_STATE_PATH)

# IDs of all traffic lights
TLS_NODES: tuple[TrafficLightSystem, ...] = tuple(
    TrafficLightSystem(
        tls_id,
        traci.trafficlight.getControlledLanes(tls_id),
        traci.trafficlight.getAllProgramLogics(tls_id)[0].phases,
    )
    for tls_id in traci.trafficlight.getIDList()
)

traci.close()
