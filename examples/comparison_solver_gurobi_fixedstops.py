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

def main():
    """Solve the CVRP problem."""
    if len(sys.argv) != 7:
        print('Should be called as follows: python comparison_solver_gurobi_fixedstops.py [number of waypoints] [number of drones] [range of drone] [velocity of drone] [velocity of truck] [number of fixed depots for comparison]')
        return
    
    numnodes = int(sys.argv[1])
    numdrones = int(sys.argv[2])
    dronerange = math.ceil(float(sys.argv[3]))
    dronevel = float(sys.argv[4])
    truckvel = float(sys.argv[5])
    numdepots = int(sys.argv[6])
    bound_length = 1000
    
    #max number of vehicles is just one per node lol, vrp solver takes care of the rest
    numvehicles = numnodes

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
    
    depart_times, wait_times, deployments, route_times, objective, xc, yc, visit_order = truck_gurobi.tsp_truck(routedistances, usednodecoords, numdrones, dronevel, truckvel)

    usednodecoords = [0] + usednodecoords

    combined_route_times = sum(route_times.values())
    print("\n--------------------")
    print("\nOPTIMIZED DRONE DEPLOYMENT TIME:", objective)
    print("LOWER BOUND TIME:", combined_route_times/float(numdrones))
    
    complete_solver_gurobi.graph_solution(data, manager, routing, solution, depart_times, wait_times, deployments, route_times, objective, xc, yc, visit_order, usednodecoords)

    comparison_vrp_fixedstops.multi_fixed_depot_comparison(numnodes, numdrones, numdepots, dronerange, x, y, dronevel, truckvel)
    


if __name__ == '__main__':
    main()