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
def graph_solution(data, manager, routing, solution, visited_nodes, marginal_utilities, total_utilities, nodes_per_trip):
    max_route_distance = 0
    
    plt.figure(figsize=(10.0, 5.0))
    
    plt.subplot(1,2,1)
    plt.plot(data['x'],data['y'],'ko',markersize=10)
    
    if(visited_nodes != None):
        for nodedex in visited_nodes:
            plt.plot(data['x'][nodedex],data['y'][nodedex],'go',markersize=10)
    
    #plot the depots as blue diamonds
    for i in range(data['num_depots']):
        plt.plot(data['x'][i],data['y'][i],'bD',markersize=10)
    
    plt.xlim(0 - data['bound_length']*0.1,data['bound_length']*1.1)
    plt.ylim(0 - data['bound_length']*0.1,data['bound_length']*1.1)
    cmap = plt.get_cmap('gist_rainbow')
    colors = [cmap(i) for i in np.linspace(0,1,data['num_vehicles_depot'])]
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
            newline, = plt.plot([data['x'][prevnodenum],data['x'][nodenum]],[data['y'][prevnodenum],data['y'][nodenum]],'-',c=colors[vehicle_id % data['num_vehicles_depot']], label='$Vehicle {i}$'.format(i=((vehicle_id%data['num_vehicles_depot']) + 1)))
            
            if(previous_index == routing.Start(vehicle_id) and vehicle_id < data['num_vehicles_depot']):
                legendlines.append(newline)
            
            max_route_distance = max(route_distance, max_route_distance)
     
    plt.legend(handles=legendlines, labels=['Vehicle {i}'.format(i=(vehicle_id+1)) for vehicle_id in range(data['num_vehicles_depot'])], loc='best', prop={'size': 8})
    
    plt.subplot(1,2,2)
    plt.plot([i+1 for i in range(data['num_vehicles_depot'])], marginal_utilities, 'ro-', markersize = 5)
    plt.plot([i+1 for i in range(data['num_vehicles_depot'])], total_utilities, 'ko-', markersize = 5)
    plt.plot([i+1 for i in range(data['num_vehicles_depot'])], nodes_per_trip, 'bo-', markersize = 5)
    plt.xlim(0,data['num_vehicles_depot'] + 1)
    plt.ylim(0, max(total_utilities)*1.1)
    plt.legend(labels=['Marginal Visited Nodes', 'Total Visited Nodes', 'Visited Nodes Per Trip'], loc='best', prop={'size': 8})

    #plt.show()
    imagename = "{i}_vehicle_solution_tu.png".format(i=data['num_vehicles_depot'])
    plt.savefig(imagename)

#gets number of total visited nodes, and identities of those nodes to be removed for a particular solution
def get_visited(data, manager, routing, solution):
    total_visited = 0
    visited_nodes = []
    num_trips = 0
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        vehicle_visited = 0
        vehicle_nodes = []
        
        if(not(routing.IsEnd(index))):
            nodex = manager.IndexToNode(index)
            nextdex = manager.IndexToNode(solution.Value(routing.NextVar(index)))
            if(nodex != nextdex):
                num_trips += 1
            
        while not routing.IsEnd(index):
            nodex = manager.IndexToNode(index)
            nextdex = manager.IndexToNode(solution.Value(routing.NextVar(index)))
            if(nodex != nextdex):
                vehicle_visited += 1
                vehicle_nodes.append(nodex)
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
    
    print("Solving for optimal number of vehicles in 100x100 square area based on marginal utility of next vehicle.\n")
    
    density = float(sys.argv[1])
    bound_length = 100
    num_depots = int(sys.argv[2])
    num_nodes = int((bound_length*bound_length)*density) + num_depots
    max_distance = int(sys.argv[3])
    
    #create depot coordinates along the line y=x in the square region
    depotcoords = np.linspace(0, bound_length, num_depots, dtype = 'int32').tolist()
    
    #create data for problem
    data = {}
    data['x'] = np.random.randint(low=0,high=bound_length,size=num_nodes).tolist()
    data['y'] = np.random.randint(low=0,high=bound_length,size=num_nodes).tolist()
    
    #put params in data
    data['max_distance'] = max_distance
    data['bound_length'] = bound_length
    data['num_nodes'] = num_nodes
    data['num_depots'] = num_depots
    
    #depots will always be the first k=num_depots nodes
    for i in range(num_depots):
        data['x'][i] = depotcoords[i]
        data['y'][i] = depotcoords[i]
        
    #construct distance matrix based on current depot
    data['distance_matrix'] = get_dist_mat(data,data['num_nodes'])
    
    #make penalty of each node larger than the maximum possible total sum of distances
    data['penalty'] = int(np.sqrt(2)*(bound_length)*num_nodes)
    
    #assign vehicles starts and ends num_vehicles at a time, start with 1 vehicle per node
    data['starts'] = [i for i in range(num_depots)]
    data['ends'] = [i for i in range(num_depots)]
    
    #initial number of vehicles is the same as number of depots (1 vehicle per depot)
    data['num_vehicles'] = num_depots
    #num_vehicles_depot is number of vehicles per depot
    data['num_vehicles_depot'] = 1
    
    marginal_utilities = []
    total_utilities = []
    nodes_per_trip = []
            
    #while loop to find the number of vehicles that maximizes the total utility of drones (nodes visited)
    prev_total_visited = 0
    prev_visited_nodes = None
    prevmanager = None
    prevrouting = None
    prevsolution = None
    manager, routing, solution = vrp.solvemulti(data)
    total_visited, visited_nodes, total_trips = get_visited(data, manager, routing, solution)
    total_utilities.append(total_visited)
    marginal_utilities.append(total_visited - prev_total_visited)
    nodes_per_trip.append((float(total_visited))/total_trips)   
            
    while((total_visited > prev_total_visited) and (manager != None)):
        prev_total_visited = total_visited
        prev_visited_nodes = visited_nodes
        prev_total_trips = total_trips
        #print(*prev_visited_nodes)
        prevmanager = manager
        prevrouting = routing
        prevsolution = solution
        
        print("Total Number of Vehicles Needed:", data['num_vehicles_depot'])
        print("Max Possible Number of Vehicle Trips:", data['num_vehicles'])
        print("Total Number of Trips", total_trips)
        print("Number of nodes visited per trip:", nodes_per_trip[-1], "\n")
        
        graph_solution(data, prevmanager, prevrouting, prevsolution, prev_visited_nodes, marginal_utilities, total_utilities, nodes_per_trip)
 
        #increase number of vehicles PER DEPOT by 1, reflect change in total number of vehicles
        data['num_vehicles'] = data['num_vehicles'] + num_depots
        data['num_vehicles_depot'] += 1
        
        #modify starts and ends to reflect the larger number of vehicles
        newstarts = []
        newends = []
        
        for i in range(num_depots):
            for j in range(data['num_vehicles_depot']):
                newstarts.append(i)
                newends.append(i)
        
        data['starts'] = newstarts
        data['ends'] = newends
        
        del newstarts
        del newends 
        
        manager, routing, solution = vrp.solvemulti(data)
        total_visited, visited_nodes, total_trips = get_visited(data, manager, routing, solution) 
        total_utilities.append(total_visited)
        marginal_utilities.append(total_visited - prev_total_visited)
        nodes_per_trip.append((float(total_visited))/total_trips)         
         
    #since we checked one vehicle above the optimal, best solution # of vehicles is 1 less than current.
    data['num_vehicles_depot'] = data['num_vehicles_depot'] - 1
    data['num_vehicles'] = data['num_vehicles'] - num_depots
        
    total_nodes_visited = prev_total_visited
    total_trips = prev_total_trips
    total_vehicles = data['num_vehicles_depot']
                
    #plt.plot(data['x'],data['y'],'ko',markersize=10)
    #plt.show()
    #pdb.set_trace()
      
    print("Overall Total Number of Vehicles Needed:", total_vehicles)
    print("Overall Max Possible Number of Vehicle Trips:", data['num_vehicles'])
    print("Overall Total Number of Trips", total_trips)
    print("Overall number of nodes visited per vehicle trip:", (float(total_nodes_visited))/total_trips)
        


if __name__ == '__main__':
    main()