"""Vehicles Routing Problem (VRP)."""

from __future__ import print_function
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
import math


def getdist(x1, y1, x2, y2):
        return np.sqrt((abs(x1-x2)**2) + (abs(y1-y2)**2))

def create_data_model(num_nodes, num_vehicles):
    """Stores the data for the problem."""
    
    bound_length = 1000
    num_vehicles = int(num_vehicles)
    #only have one depot
    num_depots = 1
    num_nodes = int(num_nodes) + 1
    
    
    data = {}
    x,y = np.random.randint(low=0,high=bound_length,size=(2,(num_nodes)))
    #initial "depot" coordinates is just the start of the road.
    x[0] = 0
    y[0] = 0
    data['x'] = x
    data['y'] = y

    data['penalty'] = int(np.sqrt(2)*(bound_length)*(num_nodes**3))
    
    print("PENALTY:", data['penalty'])
    
        
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
    data['num_depots'] = num_depots
    data['num_nodes'] = num_nodes
    data['bound_length'] = bound_length
    
    #distance between depot and nodes set to shortest distance to road (y=x in this case)
    node_roaddists = []
    roadnodecoords = []
    for i in range(0, num_nodes):
        xlen = data['x'][i]
        ylen = data['y'][i]
        sidelen = (xlen + ylen) / 2.0
        roaddist = getdist(sidelen, sidelen, xlen, ylen)
        node_roaddists.append(roaddist)
        roadnodecoords.append(sidelen)
        data['distance_matrix'][0][i] = roaddist
        data['distance_matrix'][i][0] = roaddist
            
    data['roadnodecoords'] = roadnodecoords
    #max_distance is set to twice the maximum of the minimum distances from any node to a depot
    data['max_distance'] = math.ceil(2*max(node_roaddists))
    #data['max_distance'] = 1500
    
    print("\nMAX DISTANCE:", data['max_distance'], "\n")  
    
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
    num_nodes = data['num_nodes']
    plt.plot(data['x'][list(range(1, num_nodes))],data['y'][list(range(1, num_nodes))],'ko',markersize=7)
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
            linecreated = False
            if (prevnodenum != 0 and nodenum != 0):
                newline, = plt.plot([data['x'][prevnodenum],data['x'][nodenum]],[data['y'][prevnodenum],data['y'][nodenum]], '-',c=colors[vehicle_id], label='$Vehicle {i}$'.format(i=vehicle_id+1))
            elif (prevnodenum == 0 and nodenum != 0):
                plt.plot(data['roadnodecoords'][nodenum],data['roadnodecoords'][nodenum],'bD',markersize=7)
                newline, = plt.plot([data['roadnodecoords'][nodenum],data['x'][nodenum]],[data['roadnodecoords'][nodenum],data['y'][nodenum]], '-',c=colors[vehicle_id],label='$Vehicle {i}$'.format(i=vehicle_id+1))
                linecreated = True
            elif (prevnodenum != 0 and nodenum == 0):
                plt.plot(data['roadnodecoords'][prevnodenum],data['roadnodecoords'][prevnodenum],'bD',markersize=7)
                newline, = plt.plot([data['x'][prevnodenum],data['roadnodecoords'][prevnodenum]],[data['y'][prevnodenum],data['roadnodecoords'][prevnodenum]], '-',c=colors[vehicle_id], label='$Vehicle {i}$'.format(i=vehicle_id+1))
            
            if(linecreated):
                legendlines.append(newline)
            
            max_route_distance = max(route_distance, max_route_distance)
     
    plt.legend(handles=legendlines, loc='best', prop={'size': 5})
    imagename = "{i}_vehicle_solution_singledepotroad.png".format(i=data['num_vehicles'])
    plt.xlim(0,data['bound_length'])
    plt.ylim(0, data['bound_length'])
    plt.savefig(imagename, dpi=dpi)
    plt.show()
    #plt.clf()
    
    #graphs solution using matplotlib
def extract_roadnode_data(data, manager, routing, solution):
    routedistances = []
    usednodecoords = []
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        route_distance = 0
        while not routing.IsEnd(index):
            previous_index = index
            prevnodenum = manager.IndexToNode(previous_index)
            index = solution.Value(routing.NextVar(index))
            nodenum = manager.IndexToNode(index)
            route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
            linecreated = False
            if (prevnodenum == 0 and nodenum != 0):
                usednodecoords.append(data['roadnodecoords'][nodenum])
            elif (prevnodenum != 0 and nodenum == 0):
                usednodecoords.append(data['roadnodecoords'][prevnodenum])
            
        if(route_distance != 0):
            routedistances.append(route_distance)
     
    #numroadnodes = len(usednodecoords)
    
    for i in range(len(routedistances)):
        print("Route", i, " distance", routedistances[i], " endpoints:", usednodecoords[2*i], usednodecoords[2*i + 1])
    
    """
    dist_mat = np.zeros((numroadnodes, numroadnodes))
    
    for i in range(numroadnodes):
        for j in range(numroadnodes):
            dist_mat[i][j] = getdist(usednodecoords[i],usednodecoords[i],usednodecoords[j],usednodecoords[j])
    """
            
    return routedistances, usednodecoords

def solve_singledepotroad(numnodes, numvehicles):
        
    # Instantiate the data problem.
    data = create_data_model(numnodes, numvehicles)

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
    for node in range(1, data['num_nodes']):
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
        vrpdata = {}
        vrpdata['data'] = data
        vrpdata['manager'] = manager
        vrpdata['routing'] = routing
        vrpdata['solution'] = solution
        
        return vrpdata, truckdata
    
    else:
        print("No Solution Found.") 
        return 

def main():
    """Solve the CVRP problem."""
    if len(sys.argv) != 3:
        print('Should be called as follows: python vrp_multipledepots.py [number of nodes] [number of vehicles]')
        return
        
    # Instantiate the data problem.
    data = create_data_model(sys.argv[1], sys.argv[2])

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
    for node in range(1, data['num_nodes']):
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
        routedistances, usednodecoords = extract_roadnode_data(data, manager, routing, solution)
        graph_solution(data, manager, routing, solution)
    
    else:
        print("No Solution Found.")


if __name__ == '__main__':
    main()