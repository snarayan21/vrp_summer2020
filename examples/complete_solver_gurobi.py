from __future__ import print_function
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
import math
import vrp_efficient_singledepotroad as efficient_singledepotroad
import tsp_truck_gurobi as truck_gurobi

#graphs solution using matplotlib
def graph_solution(data, manager, routing, solution, active_arcs, xc, yc, visit_order):
    dpi = 192
    plt.figure(figsize=(1000/dpi, 1000/dpi))
    num_nodes = data['num_nodes']
    plt.plot(data['x'][list(range(1, num_nodes))],data['y'][list(range(1, num_nodes))],'ko',markersize=7, zorder = 0)
    legendlines = []
    
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        while not routing.IsEnd(index):
            previous_index = index
            prevnodenum = manager.IndexToNode(previous_index)
            index = solution.Value(routing.NextVar(index))
            nodenum = manager.IndexToNode(index)
            linecreated = False
            if (prevnodenum != 0 and nodenum != 0):
                newline, = plt.plot([data['x'][prevnodenum],data['x'][nodenum]],[data['y'][prevnodenum],data['y'][nodenum]], '-',c='c', label='$Vehicle {i}$'.format(i=vehicle_id+1), zorder = 1)
            elif (prevnodenum == 0 and nodenum != 0):
                #plt.plot(data['roadnodecoords'][nodenum],data['roadnodecoords'][nodenum],'bD',markersize=7)
                newline, = plt.plot([data['roadnodecoords'][nodenum],data['x'][nodenum]],[data['roadnodecoords'][nodenum],data['y'][nodenum]], '-',c='c',label='$Vehicle {i}$'.format(i=vehicle_id+1), zorder = 1)
                linecreated = True
            elif (prevnodenum != 0 and nodenum == 0):
                #plt.plot(data['roadnodecoords'][prevnodenum],data['roadnodecoords'][prevnodenum],'bD',markersize=7)
                newline, = plt.plot([data['x'][prevnodenum],data['roadnodecoords'][prevnodenum]],[data['y'][prevnodenum],data['roadnodecoords'][prevnodenum]], '-',c='c', label='$Vehicle {i}$'.format(i=vehicle_id+1), zorder = 1)
            
            if(linecreated):
                legendlines.append(newline)
    
          
    for i, j in active_arcs:
        plt.plot([xc[0], max(xc)], [yc[0], max(yc)], color='0.3', zorder=0)
        plt.scatter(xc[1:], yc[1:], c = 'w', linewidth = 6, zorder = 3)
        
    plt.scatter([xc[0]], [yc[0]], c='r', linewidth='7')
        
    for i in range(len(visit_order)):
        dex = visit_order[i]
        plt.text(xc[dex],yc[dex],i+1, ha="center", va="center")
         
    #plt.legend(handles=legendlines, loc='best', prop={'size': 5})
    imagename = "solution_singledepotroad.png"
    plt.xlim(0,data['bound_length'])
    plt.ylim(0, data['bound_length'])
    plt.savefig(imagename, dpi=dpi)
    plt.show()
    plt.clf()
    

def main():
    """Solve the CVRP problem."""
    if len(sys.argv) != 6:
        print('Should be called as follows: python complete_solver_gurobi.py [number of waypoints] [number of drones] [range of drone] [velocity of drone] [velocity of truck]')
        return
    
    numnodes = int(sys.argv[1])
    numdrones = int(sys.argv[2])
    dronerange = math.ceil(float(sys.argv[3]))
    dronevel = float(sys.argv[4])
    truckvel = float(sys.argv[5])
    
    #max number of vehicles is just one per node lol, vrp solver takes care of the rest
    numvehicles = numnodes
    
    vrpdata, truckdata = efficient_singledepotroad.solve_singledepotroad(numnodes, numvehicles, dronerange)
    
    data = vrpdata['data']
    manager = vrpdata['manager']
    routing = vrpdata['routing']
    solution = vrpdata['solution']
    routedistances = truckdata['routedistances']
    usednodecoords = truckdata['usednodecoords']
    
    active_arcs, xc, yc, visit_order = truck_gurobi.tsp_truck(routedistances, usednodecoords, numdrones, dronevel, truckvel)
    
    graph_solution(data, manager, routing, solution, active_arcs, xc, yc, visit_order)
    


if __name__ == '__main__':
    main()