"""Vehicles Routing Problem (VRP)."""

from __future__ import print_function
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
import math



def create_data_model(num_nodes, num_vehicles, num_depots):
    """Stores the data for the problem."""
    
    bound_length = 1000
    num_vehicles = int(num_vehicles)
    num_depots = int(num_depots)
    #one set of depot nodes (split the original depot nodes) for every vehicle, which allows for overlaps
    #zeroth depot node is NOT split
    num_depots_total = 2*(num_depots-1)*num_vehicles + 1
    #num_nodes_total includes total number of dummy depots
    num_nodes_total = int(num_nodes) + num_depots_total
    num_nodes = int(num_nodes) + num_depots
    
    
    data = {}
    x,y = np.random.randint(low=0,high=bound_length,size=(2,(num_nodes)))
    depotcoords = np.linspace(0, bound_length, num_depots+2, dtype = 'int32').tolist()
    depotcoords.pop(0)
    depotcoords.pop(-1)
    xdeps = np.repeat(depotcoords[1:num_depots], 2*num_vehicles)
    ydeps = np.repeat(depotcoords[1:num_depots], 2*num_vehicles)
    data['x'] = np.concatenate([[depotcoords[0]], xdeps, x[1:]])
    data['y'] = np.concatenate([[depotcoords[0]], ydeps, y[1:]])
    #print(*list(x))
    #print(*list(xdeps))
    #print(*list(data['x']))
    #print("length of coords:", len(data['y']))
    #print("num_depots_total:", num_depots_total)
    #print("num_nodes_total:", num_nodes_total)
    data['penalty'] = int(np.sqrt(2)*(bound_length)*(num_nodes_total**3))
    
    print("PENALTY:", data['penalty'])
    
    def getdist(x1, y1, x2, y2):
        return np.sqrt((abs(x1-x2)**2) + (abs(y1-y2)**2))
        
    #return euclidean distance matrix
    dist_mat = np.zeros((num_nodes_total, num_nodes_total))
    for i in range(num_nodes_total):
        x1 = data['x'][i]
        y1 = data['y'][i]
        for j in range(num_nodes_total):
            x2 = data['x'][j]
            y2 = data['y'][j]
            dist_mat[i][j] = getdist(x1,y1,x2,y2)
      
    
    #distance between depots set to 0
    #dist_mat[0:num_vehicles][0:num_vehicles] = 0
    data['distance_matrix'] = dist_mat.tolist()
    data['num_vehicles'] = num_vehicles
    data['num_depots'] = num_depots
    data['num_nodes_total'] = num_nodes_total
    data['num_depots_total'] = num_depots_total
    
    min_from_node = []
    for i in range(num_depots_total, num_nodes_total):
        mindist = 2*bound_length
        #want to find smallest distance from this node to a depot
        for j in range(0,num_depots_total,(2*num_vehicles)):
            distance = data['distance_matrix'][i][j]
            if(distance < mindist):
                mindist = distance
        
        min_from_node.append(mindist)
    
    #max_distance is set to twice the maximum of the minimum distances from any node to a depot
    data['max_distance'] = int(2.4*math.ceil(max(min_from_node)))
    #data['max_distance'] = 1500
    
    print("\nMAX DISTANCE:", data['max_distance'], "\n")
    
    for i in range(num_depots_total):
        for j in range(num_depots_total):
            if(i != 0 and j != 0 and i != j):
                data['distance_matrix'][i][j] = data['max_distance'] + 1
                #data['distance_matrix'][i][j] = 0
            else:
                data['distance_matrix'][i][j] = 0
    
    
    li = data['distance_matrix']
    for i in range(num_nodes):
        for j in range(num_nodes):
            li[i][j] = int(li[i][j])
            
    print(pd.DataFrame(li).round(0))
    
    #if setting start and end depots
    #data['starts'] = np.random.randint(low=0,high=num_nodes,size=num_vehicles).tolist()
    #data['ends'] = [0 for i in range(num_vehicles)]
    data['depot'] = 0
    
    del dist_mat
    
    return data


def print_solution(data, manager, routing, solution):
    """Prints solution on console."""
    max_route_distance = 0
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {}:\n'.format(vehicle_id+1)
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
def graph_solution(data, manager, routing, solution):
    dpi = 192
    plt.figure(figsize=(1000/dpi, 1000/dpi))
    max_route_distance = 0
    num_depots_total = data['num_depots_total']
    num_nodes_total = data['num_nodes_total']
    plt.plot(data['x'][list(range(num_depots_total, num_nodes_total))],data['y'][list(range(num_depots_total, num_nodes_total))],'ko',markersize=10)
    #if setting start and end points
    #plt.plot(data['x'][data['starts']],data['y'][data['starts']],'kD',markersize=10)
    li = list(range(0,data['num_depots_total'],data['num_vehicles']))
    plt.plot(data['x'][li],data['y'][li],'bD',markersize=10)
    #imagename = "{i}_nodemap_connecteddepot.png".format(i=data['num_vehicles'])
    #plt.savefig(imagename, dpi=dpi)
    #plt.show()
    plt.plot(data['x'][0],data['y'][0],'gD',markersize=10)
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
     
    plt.legend(handles=legendlines, labels=['Vehicle {i}'.format(i=(vehicle_id+1)) for vehicle_id in range(data['num_vehicles'])], loc='best', prop={'size': 5})
    imagename = "{i}_vehicle_solution_connecteddepot.png".format(i=data['num_vehicles'])
    plt.savefig(imagename, dpi=dpi)
    plt.show()
    plt.clf()


def main():
    """Solve the CVRP problem."""
    if len(sys.argv) != 4:
        print('Should be called as follows: python vrp_multipledepots.py [number of nodes] [number of vehicles] [number of depots]')
        return
        
    # Instantiate the data problem.
    data = create_data_model(sys.argv[1], sys.argv[2], sys.argv[3])

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),data['num_vehicles'],data['depot'])
    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)


    # Create and register a transit callback.
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Distance constraint.
    dimension_name = 'Distance'
    routing.AddDimension(
        transit_callback_index,
        0,  # no slack
        data['max_distance'],  # vehicle maximum travel distance
        #data['penalty'],
        True,  # start cumul to zero
        dimension_name)
    distance_dimension = routing.GetDimensionOrDie(dimension_name)
    #distance_dimension.SetGlobalSpanCostCoefficient(100)
    
    #penalty for not visiting non-depot nodes
    penalty = data['penalty']
    #penalty = data['max_distance']
    for node in range(1, data['num_nodes_total']):
        if (node < data['num_depots_total']):
            routing.AddDisjunction([manager.NodeToIndex(node)], 0)
        else:
            routing.AddDisjunction([manager.NodeToIndex(node)], penalty)


    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    #search_parameters.local_search_metaheuristic = (routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    #search_parameters.local_search_metaheuristic = (routing_enums_pb2.LocalSearchMetaheuristic.TABU_SEARCH)
    search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    #search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_MOST_CONSTRAINED_ARC)
    #search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.SAVINGS)
    #search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.SWEEP)
    #search_parameters.time_limit.seconds = 20
    #search_parameters.solution_limit = 200

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if solution:
        print_solution(data, manager, routing, solution)
        graph_solution(data, manager, routing, solution)
    
    else:
        print("No Solution Found.")


if __name__ == '__main__':
    main()