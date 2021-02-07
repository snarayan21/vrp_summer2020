"""Vehicles Routing Problem (VRP)."""

from __future__ import print_function
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import numpy as np
import matplotlib.pyplot as plt
import sys
import vrp_multidepot_solvefuncs as vrp
import pdb


"""
def create_data_model(num_nodes, num_vehicles):
    num_nodes = int(num_nodes)
    num_vehicles = int(num_vehicles)
    
    data = {}
    data['x'],data['y'] = np.random.randint(low=0,high=1000,size=(2,num_nodes));
    data['penalty'] = int(np.sqrt(2)*(1000)*num_nodes)
    
    def getdist(x1, y1, x2, y2):
        return np.sqrt((abs(x1-x2)^2) + (abs(y1-y2)^2))
        
    #return euclidean distance matrix
    dist_mat = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        x1 = data['x'][i]
        y1 = data['y'][i]
        for j in range(num_nodes):
            x2 = data['x'][j]
            y2 = data['y'][j]
            dist_mat[i][j] = getdist(x1,y1,x2,y2)
    
    data['distance_matrix'] = dist_mat.tolist()
    data['num_vehicles'] = num_vehicles
    data['depot'] = 0
    
    del dist_mat
    
    return data
"""

def print_solution(data, manager, routing, solution):
    """Prints solution on console."""
    max_route_distance = 0
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {}:\n'.format(vehicle_id + 1)
        route_distance = 0
        while not routing.IsEnd(index):
            plan_output += ' {} -> '.format(manager.IndexToNode(index))
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id)
        plan_output += '{}\n'.format(manager.IndexToNode(index))
        plan_output += 'Distance of the route: {}m\n'.format(route_distance)
        print(plan_output)
        max_route_distance = max(route_distance, max_route_distance)
    print('Maximum of the route distances: {}m'.format(max_route_distance))

#graphs solution using matplotlib
def graph_solution(data, manager, routing, solution, visited_nodes, currdepot):
    max_route_distance = 0
    plt.plot(data['x'],data['y'],'ko',markersize=10)
    
    if(visited_nodes != None):
        for nodedex in visited_nodes:
            plt.plot(data['x'][nodedex],data['y'][nodedex],'go',markersize=10)
        
    plt.plot(data['x'][0],data['y'][0],'bD',markersize=10)
    plt.xlim(0 - data['bound_length']*0.1,data['bound_length']*1.1)
    plt.ylim(0 - data['bound_length']*0.1,data['bound_length']*1.1)
    cmap = plt.get_cmap('gist_rainbow')
    colors = [cmap(i) for i in np.linspace(0,1,data['num_vehicles'])]
    legendlines = []
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        route_distance = 0
        while not routing.IsEnd(index):
            previous_index = index
            prevnodenum = manager.IndexToNode(previous_index)
            index = solution.Value(routing.NextVar(index))
            nodenum = manager.IndexToNode(index)
            route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
            newline, = plt.plot([data['x'][prevnodenum],data['x'][nodenum]],[data['y'][prevnodenum],data['y'][nodenum]],
            '-',c=colors[vehicle_id], label='$Vehicle {i}$'.format(i=vehicle_id))
            if(previous_index == routing.Start(vehicle_id)):
                legendlines.append(newline)
            max_route_distance = max(route_distance, max_route_distance)
     
    plt.legend(handles=legendlines, labels=['Vehicle {i}'.format(i=(vehicle_id+1)) for vehicle_id in range(data['num_vehicles'])], loc='best')
    
    plt.show()
    imagename = "Depot_{i}_solution_mu.png".format(i=currdepot)
    #plt.savefig(imagename)

#gets number of total visited nodes, and identities of those nodes to be removed for a particular solution   
def get_visited(data, manager, routing, solution):
    total_visited = 0
    visited_nodes = []
    num_trips = 0
    
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        startdex = manager.IndexToNode(index)
        vehicle_visited = 0
        vehicle_nodes = []
        
        if(not(routing.IsEnd(index))):
            nodex = manager.IndexToNode(index)
            nextdex = manager.IndexToNode(solution.Value(routing.NextVar(index)))
            if(nodex != nextdex):
                num_trips += 1
            
        while not routing.IsEnd(index):
            nodex = manager.IndexToNode(index)
            nextdex = solution.Value(routing.NextVar(index))
            nextnodedex = manager.IndexToNode(nextdex)
            if(nodex != nextnodedex and not(routing.IsEnd(nextdex))):
                vehicle_visited += 1
                vehicle_nodes.append(nextnodedex)
            previous_index = index
            index = solution.Value(routing.NextVar(index))       
            
        total_visited += vehicle_visited
        visited_nodes = visited_nodes + vehicle_nodes
    
    return total_visited, visited_nodes, num_trips

#returns euclidean distance between two points
def getdist(x1, y1, x2, y2):
    return np.sqrt(((abs(x1-x2))**2) + ((abs(y1-y2))**2))
        
def get_dist_mat(data,num_nodes):
    #return euclidean distance matrix
    dist_mat = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        x1 = data['x'][i]
        y1 = data['y'][i]
        for j in range(num_nodes):
            x2 = data['x'][j]
            y2 = data['y'][j]
            dist_mat[i][j] = getdist(x1,y1,x2,y2)
    
    return dist_mat.tolist()

def main():
    """Solve the CVRP problem."""
    if len(sys.argv) != 4:
        print('Should be called as follows: python vrp_multipledepots.py [density of nodes per unit^2] [number of depots to approximate diagonal road] [range of drone on a single charge]')
        return
    
    print("Solving for optimal number of vehicles in 100x100 square area based on marginal utility of next vehicle.")
    
    density = float(sys.argv[1])
    bound_length = 100
    num_nodes = int((bound_length*bound_length)*density) + 1
    num_depots = int(sys.argv[2])
    max_distance = int(sys.argv[3])
    
    #create depot coordinates along the line y=x in the square region
    depotcoords = np.linspace(0, bound_length, num_depots, dtype = 'int32').tolist()
    
    #create data for problem
    data = {}
    data['x'] = np.random.randint(low=0,high=bound_length,size=num_nodes).tolist()
    data['y'] = np.random.randint(low=0,high=bound_length,size=num_nodes).tolist()
    
    #make penalty of each node larger than the maximum possible total sum of distances
    data['penalty'] = int(np.sqrt(2)*(bound_length)*num_nodes)
    #depot will always be the zeroth indexed node
    data['depot'] = 0
    data['max_distance'] = max_distance
    data['bound_length'] = bound_length
    data['num_nodes'] = num_nodes
    
    total_trips = 0
    total_nodes_visited = 0
    total_vehicles = 0
    currdepot = 0
    while(currdepot < num_depots):
        #construct distance matrix based on current depot
        (data['x'])[0] = depotcoords[currdepot]
        (data['y'])[0] = depotcoords[currdepot]
        data['distance_matrix'] = get_dist_mat(data,data['num_nodes'])
        #initial number of vehicles is 1
        data['num_vehicles'] = 1
        
        #another while loop to find the number of vehicles that maximizes the marginal utility of drones
        prev_marginal_utility = 0
        prev_total_visited = 0
        prev_num_trips = 0
        prev_visited_nodes = None
        prevmanager = None
        prevrouting = None
        prevsolution = None
        manager, routing, solution = vrp.solve(data)
        total_visited, visited_nodes, num_trips = get_visited(data, manager, routing, solution)
        #marginal utility of first vehicle is total nodes it finds minus previously found nodes, which is none
        marginal_utility = total_visited - prev_total_visited
                
        while((marginal_utility >= prev_marginal_utility) and (manager != None) and (total_visited != 0) and (marginal_utility != 0)):
            prev_marginal_utility = marginal_utility
            prev_total_visited = total_visited
            prev_visited_nodes = visited_nodes
            prev_num_trips = num_trips
            #print(*prev_visited_nodes)
            prevmanager = manager
            prevrouting = routing
            prevsolution = solution
            #increase number of vehicles by 1
            data['num_vehicles'] = data['num_vehicles'] + 1
            manager, routing, solution = vrp.solve(data)
            total_visited, visited_nodes, num_trips = get_visited(data, manager, routing, solution)
            marginal_utility = total_visited - prev_total_visited
        
        #update total_vehicles if num_vehicles - 1 (since we checked one vehicle above the optimal) > total_vehicles
        data['num_vehicles'] = data['num_vehicles'] - 1
        if(data['num_vehicles'] > total_vehicles):
            total_vehicles = data['num_vehicles']
        
        print("Depot Number:", currdepot, " Number of vehicles:", data['num_vehicles'])
        
        
        graph_solution(data, prevmanager, prevrouting, prevsolution, prev_visited_nodes, currdepot)
        
        #delete visited node coordinates in DESCENDING ORDER and increment currdepot since this depot has been completed
        #sort the visited nodes in descending order, so deletions happen from outermost to inner and don't change inner indices
        print("previous number of nodes:", len(data['x']))

        if(prev_visited_nodes != None):
        
            prev_visited_nodes.sort(reverse = True)
        
            #delete the row and column in the x and y vals corresponding to the visited nodes
            for node_dex in prev_visited_nodes:
                del data['x'][node_dex]
                del data['y'][node_dex]
            
        print("new number of nodes:", len(data['x']))
        

        data['num_nodes'] = data['num_nodes'] - prev_total_visited
        
        total_nodes_visited += prev_total_visited
        total_trips += data['num_vehicles']
        
        currdepot += 1
        
        #plt.plot(data['x'],data['y'],'ko',markersize=10)
        #plt.show()
        #pdb.set_trace()
    
    print("Total Number of Vehicles Needed:", total_vehicles)
    print("Number of nodes visited per vehicle trip:", (float(total_nodes_visited))/total_trips)
        


if __name__ == '__main__':
    main()