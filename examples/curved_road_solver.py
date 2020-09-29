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
import cv2
import shapely.geometry as geom

def main():
    """Solve the CVRP problem."""
    if len(sys.argv) != 6:
        print('Should be called as follows: python curved_road_solver.py [number of waypoints] [number of drones] [range of drone] [velocity of drone] [velocity of truck]')
        return
    
    img = cv2.imread('/Users/saaketh/Desktop/vrp_summer2020/edited_road_map_w_bkg.png',0)
    ret,thresh = cv2.threshold(img,5,230, cv2.THRESH_BINARY)
    height, width = img.shape
    print("height and width : ",height, width)
    size = img.size
    #imgplot = plt.imshow(thresh, 'gray')
    xlin = []
    ylin = []
    for i in range(len(thresh)):
        found = False
        ys = []
        for j in range(len(thresh)):
            curr = thresh[j,i]
            if (curr == 0 and found == False):
                found = True
                ys.append(len(thresh) - (j+1))
            elif (curr == 0 and found == True):
                ys.append(len(thresh) - (j+1))
            elif (curr != 0 and found == True):
                ylin.append(sum(ys)/len(ys))
                xlin.append(i)
                break
            else:
                pass
    
    numnodes = int(sys.argv[1])
    numdrones = int(sys.argv[2])
    dronerange = math.ceil(float(sys.argv[3]))
    dronevel = float(sys.argv[4])
    truckvel = float(sys.argv[5])
    bound_length = min(height, width)
    
    #max number of vehicles is just one per node lol, vrp solver takes care of the rest
    numvehicles = numnodes

    x,y = np.random.randint(low=0,high=bound_length,size=(2,(numnodes)))

    x = x.tolist()
    y = y.tolist()

    line = geom.LineString(list(zip(xlin,ylin)))
    depot = geom.Point(xlin[0], ylin[0])
    
    vrpdata, truckdata = efficient_singledepotroad.solve_singledepotroad_fromdata_curved(numnodes, numvehicles, dronerange, x, y, line, depot, bound_length)
    
    data = vrpdata['data']
    manager = vrpdata['manager']
    routing = vrpdata['routing']
    solution = vrpdata['solution']
    routedistances = truckdata['routedistances']
    usednodecoords = truckdata['usednodecoords']

    print("\nROUTEDISTANCES LENGTH:", len(routedistances))
    print("\nUSEDNODECOORDS LENGTH:", len(usednodecoords[0]))

    finalObjectives = []
    
    depart_times, wait_times, deployments, route_times, objective, xc, yc, visit_order = truck_gurobi.tsp_truck_curved(routedistances, usednodecoords, numdrones, dronevel, truckvel, depot, line, bound_length)

    print(usednodecoords)

    usednodecoords[0] = [depot.x] + usednodecoords[0]
    usednodecoords[1] = [depot.y] + usednodecoords[1]

    combined_route_times = sum(route_times.values())
    print("\n--------------------")
    print("\nOPTIMIZED DRONE DEPLOYMENT TIME:", objective)
    print("LOWER BOUND TIME:", combined_route_times/float(i))

    filename = "curved_road_solution"

    complete_solver_gurobi.graph_solution_curved(data, manager, routing, solution, depart_times, wait_times, deployments, route_times, objective, xc, yc, visit_order, usednodecoords, filename, True, line)    
    

if __name__ == '__main__':
    main()