from __future__ import print_function
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
import pandas as pd
import math
import vrp_efficient_singledepotroad as efficient_singledepotroad
import tsp_truck_gurobi as truck_gurobi
import complete_solver_gurobi
import comparison_vrp_fixedstops


numnodes = 50
numdrones = 3
dronerange = 1500
bound_length = 1000

#max number of vehicles is just one per node lol, vrp solver takes care of the rest
numvehicles = numnodes
dronevel = 15
truckvel = 15
    
velocityObjectives = []

x,y = np.random.randint(low=0,high=bound_length,size=(2,(numnodes)))

x = x.tolist()
y = y.tolist()

vrpdata, truckdata = efficient_singledepotroad.solve_singledepotroad_fromdata(numnodes, numvehicles, dronerange, x, y)

data = vrpdata['data']
manager = vrpdata['manager']
routing = vrpdata['routing']
solution = vrpdata['solution']
routedistances = truckdata['routedistances']
usednodecoords = truckdata['usednodecoords']

print("\nROUTEDISTANCES LENGTH:", len(routedistances))
print("\nUSEDNODECOORDS LENGTH:", len(usednodecoords))

for i in range(2):
    
    if(i == 0):
        depart_times, wait_times, deployments, route_times, objective, xc, yc, visit_order = truck_gurobi.tsp_truck_ortools(routedistances, usednodecoords, numdrones, dronevel, truckvel)

        usednodecoords = [0] + usednodecoords

        combined_route_times = sum(route_times.values())
        print("\n--------------------")
        print("\nOPTIMIZED DRONE DEPLOYMENT TIME:", objective)

        filename = "ortools_solution"

        complete_solver_gurobi.graph_solution(data, manager, routing, solution, depart_times, wait_times, deployments, route_times, objective, xc, yc, visit_order, usednodecoords, filename, True)

        usednodecoords.pop(0)
    else:
        depart_times, wait_times, deployments, route_times, objective, xc, yc, visit_order = truck_gurobi.tsp_truck(routedistances, usednodecoords, numdrones, dronevel, truckvel)

        usednodecoords = [0] + usednodecoords

        combined_route_times = sum(route_times.values())
        print("\n--------------------")
        print("\nOPTIMIZED DRONE DEPLOYMENT TIME:", objective)

        filename = "gurobi_solution"

        complete_solver_gurobi.graph_solution(data, manager, routing, solution, depart_times, wait_times, deployments, route_times, objective, xc, yc, visit_order, usednodecoords, filename, True)

        usednodecoords.pop(0)