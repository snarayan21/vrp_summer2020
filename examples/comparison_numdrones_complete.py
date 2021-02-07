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
    if len(sys.argv) != 3:
        print('Should be called as follows: python comparison_numdrones_complete.py [number of waypoints] [range of drone]')
        return
    
    numnodes = int(sys.argv[1])
    dronerange = math.ceil(float(sys.argv[2]))
    bound_length = 1000
    
    #max number of vehicles is just one per node lol, vrp solver takes care of the rest
    numvehicles = numnodes
    dronevel = 15
    truckvel = 5

    #overallObjectives = []

    #for k in range(3):
    """ if(k == 0):
            truckvel = 5
        
        if(k == 1):
            truckvel = 15
        
        if(k == 2):
            truckvel = 45 """
        
    velocityObjectives = []
    
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

        if(len(routedistances) > 10):
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

        for i in range(1,len(routedistances) + 2):
            print("starting ", i, " drone solution")
            depart_times, wait_times, deployments, route_times, objective, xc, yc, visit_order = truck_gurobi.tsp_truck(routedistances, usednodecoords, i, dronevel, truckvel)

            usednodecoords = [0] + usednodecoords

            combined_route_times = sum(route_times.values())
            print("\n--------------------")
            print("\nOPTIMIZED DRONE DEPLOYMENT TIME:", objective)
            print("LOWER BOUND TIME:", combined_route_times/float(i))
            finalObjectives.append(objective)

            filename = str(i) + "_drone_solution_complete_comparison_truckvel_" + str(int(truckvel)) + "_iter_" + str(j) + "_numroutes_" + str(len(routedistances))

            if(j == 0):
                complete_solver_gurobi.graph_solution(data, manager, routing, solution, depart_times, wait_times, deployments, route_times, objective, xc, yc, visit_order, usednodecoords, filename, True)
            else:
                complete_solver_gurobi.graph_solution(data, manager, routing, solution, depart_times, wait_times, deployments, route_times, objective, xc, yc, visit_order, usednodecoords, filename, False)

            usednodecoords.pop(0)
        
        velocityObjectives.append(finalObjectives)
        
        print("FINAL OBJECTIVES: ", finalObjectives)
    
    #overallObjectives.append(velocityObjectives)
    print("VELOCITY OBJECTIVES: ", velocityObjectives)
    
    #print("OVERALL OBJECTIVES: ", overallObjectives)
    

if __name__ == '__main__':
    main()