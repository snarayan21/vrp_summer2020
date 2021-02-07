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
dronerange = 1500
numdrones = 4
bound_length = 1000
dronevel = 15
truckvels = np.logspace(-1, 3, 10)
    
#max number of vehicles is just one per node lol, vrp solver takes care of the rest
numvehicles = numnodes

overall_times = []
overall_waits = []
overall_moves = []

for j in range(10):

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

    finalObjectives = []
    finalWaits = []
    finalMoves = []

    for i in range(len(truckvels)):
        print("starting truck velocity ", truckvels[i], "solution")
        depart_times, wait_times, deployments, route_times, objective, xc, yc, visit_order = truck_gurobi.tsp_truck(routedistances, usednodecoords, numdrones, dronevel, truckvels[i])

        usednodecoords = [0] + usednodecoords

        combined_route_times = sum(route_times.values())
        print("\n--------------------")
        print("\nOPTIMIZED DRONE DEPLOYMENT TIME:", objective)
        #print("LOWER BOUND TIME:", combined_route_times/float(i))
        finalObjectives.append(objective)

        total_wait_time = 0
        for wt in wait_times.values():
            total_wait_time += wt
        
        print("TOTAL WAIT TIME: ", total_wait_time)
        print("TOTAL MOVE TIME: ", objective - total_wait_time)
        finalWaits.append(total_wait_time)
        finalMoves.append(objective - total_wait_time)

        filename = str(i) + "_index_truck_velocity_solution_complete_comparison"

        """ to_graph = True
        if(truckvels[i]/dronevel < 0.05):
            to_graph = False """

        complete_solver_gurobi.graph_solution(data, manager, routing, solution, depart_times, wait_times, deployments, route_times, objective, xc, yc, visit_order, usednodecoords, filename, False)

        usednodecoords.pop(0)

    print("TRUCK VELS: ", truckvels)
    print("FINAL OBJECTIVES: ", finalObjectives)
    print("FINAL WAITS: ", finalWaits)
    print("FINAL MOVES: ", finalMoves)
    overall_times.append(finalObjectives)
    overall_waits.append(finalWaits)
    overall_moves.append(finalMoves)

print("OVERALL TIMES: ", overall_times)
print("OVERALL WAITS: ", overall_waits)
print("OVERALL MOVES: ", overall_moves)