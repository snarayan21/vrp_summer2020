"""Vehicles Routing Problem (VRP)."""

from __future__ import print_function
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import numpy as np
import matplotlib.pyplot as plt
import sys


"""
def create_data_model(num_nodes, num_vehicles):
    #Stores the data for the problem.
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
"""
def print_solution(data, manager, routing, solution):
    #Prints solution on console.
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
def graph_solution(data, manager, routing, solution):
    max_route_distance = 0
    plt.plot(data['x'],data['y'],'ko',markersize=10)
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
"""

def solve(data):
    """Solve the CVRP problem."""

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
        #set to data['max_distance'] if restricting distance of vehicle
        data['max_distance'],  # vehicle maximum travel distance
        True,  # start cumul to zero
        dimension_name)
    distance_dimension = routing.GetDimensionOrDie(dimension_name)
    
    #uncomment the below line to make the solver prioritize minimizing the maximum distance of any vehicle (we usually want to prioritize # nodes visited)
    #distance_dimension.SetGlobalSpanCostCoefficient(100)
    
    #allow drops --> high penalty prioritizes visiting more nodes since the cost of dropping another node is the penalty, which is extremely high
    penalty = data['penalty']
    for node in range(1, len(data['distance_matrix'])):
        routing.AddDisjunction([manager.NodeToIndex(node)], penalty)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if solution:
        return manager, routing, solution
    
    else:
        return None, None, None
        
def solvemulti(data):
    """Solve the CVRP problem."""    
    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),data['num_vehicles'],data['starts'],data['ends'])
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
        #set to data['max_distance'] if restricting distance of vehicle
        data['max_distance'],  # vehicle maximum travel distance
        True,  # start cumul to zero
        dimension_name)
    distance_dimension = routing.GetDimensionOrDie(dimension_name)
    
    #uncomment the below line to make the solver prioritize minimizing the maximum distance of any vehicle (we usually want to prioritize # nodes visited)
    #distance_dimension.SetGlobalSpanCostCoefficient(100)
    
    #allow drops --> high penalty prioritizes visiting more nodes since the cost of dropping another node is the penalty, which is extremely high
    penalty = data['penalty']
    for node in range(1, len(data['distance_matrix'])):
        routing.AddDisjunction([manager.NodeToIndex(node)], penalty)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.time_limit.seconds = 45

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if solution:
        return manager, routing, solution
    
    else:
        return None, None, None
        
def solvemulti_nodistlim(data):
    """Solve the CVRP problem."""

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),data['num_vehicles'],data['starts'],data['ends'])
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
        #set to data['penalty'] since we do not want there to be an effective distance constraint
        data['penalty'],  # vehicle maximum travel distance
        True,  # start cumul to zero
        dimension_name)
    distance_dimension = routing.GetDimensionOrDie(dimension_name)
    
    #uncomment the below line to make the solver prioritize minimizing the maximum distance of any vehicle (we usually want to prioritize # nodes visited)
    #we want the max path length of any vehicle to be minimized in mapping applications where we must visit every node,
    #and assuming infinite battery swaps.
    distance_dimension.SetGlobalSpanCostCoefficient(100)
    
    #do NOT allow drops --> there should be no situation in which nodes are dropped.
    """
    penalty = data['penalty']
    for node in range(1, len(data['distance_matrix'])):
        routing.AddDisjunction([manager.NodeToIndex(node)], penalty)
    """
    
    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.local_search_metaheuristic = (routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    #search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    #search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_MOST_CONSTRAINED_ARC)
    #search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.SAVINGS)
    #search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.SWEEP)
    search_parameters.time_limit.seconds = 20
    search_parameters.solution_limit = 200

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if solution:
        return manager, routing, solution
    
    else:
        return None, None, None
     
def solveconnecteddepots(data):
    """Solve the CVRP problem."""    
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
        #set to data['max_distance'] if restricting distance of vehicle
        data['max_distance'],  # vehicle maximum travel distance
        True,  # start cumul to zero
        dimension_name)
    distance_dimension = routing.GetDimensionOrDie(dimension_name)
    
    #uncomment the below line to make the solver prioritize minimizing the maximum distance of any vehicle (we usually want to prioritize # nodes visited)
    #distance_dimension.SetGlobalSpanCostCoefficient(100)
    
    #allow drops --> penalty of dropping a node should be high EXCEPT FOR DEPOTS
    penalty = data['penalty']
    for node in range(data['num_depots'], len(data['distance_matrix'])):
        routing.AddDisjunction([manager.NodeToIndex(node)], penalty)
    

    # Setting first solution heuristic.
    #search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    #search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    #search_parameters.time_limit.seconds = 45
    
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    #search_parameters.local_search_metaheuristic = (routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    #search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_MOST_CONSTRAINED_ARC)
    #search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.SAVINGS)
    #search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.SWEEP)
    search_parameters.time_limit.seconds = 10
    #search_parameters.solution_limit = 200

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if solution:
        return manager, routing, solution
    
    else:
        return None, None, None
        
           
def get_max_dist(data, manager, routing, solution):
    """Prints solution on console."""
    max_route_distance = 0.0
    cost = 0.0
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        #plan_output = 'Route for vehicle {}:\n'.format(vehicle_id + 1)
        route_distance = 0.0
        while not routing.IsEnd(index):
            #plan_output += ' {} -> '.format(manager.IndexToNode(index))
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id)
        #plan_output += '{}\n'.format(manager.IndexToNode(index))
        #plan_output += 'Distance of the route: {}m\n'.format(route_distance)
        #print(plan_output)
        max_route_distance = max(route_distance, max_route_distance)
        cost += route_distance
    #print('Maximum of the route distances: {}m'.format(max_route_distance))
    return max_route_distance, cost

#gets number of total visited nodes, and identities of those nodes to be removed for a particular solution
def get_visited(data, manager, routing, solution):
    total_visited = 0
    visited_nodes = []
    num_trips = 0
    used_vehicles = []
    used_indicator = 0
    
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
                used_indicator += 1
                                
            #here, adds used vehicles per depot to the used_vehicles list, resets used_indicator to 0 for new depot count
            if(vehicle_id % data['num_vehicles_depot'] == data['num_vehicles_depot'] - 1):
                used_vehicles.append(used_indicator)    
                used_indicator = 0        
            
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
        
    return total_visited, visited_nodes, num_trips, max(used_vehicles)

def getdist(x1, y1, x2, y2):
    return int(np.ceil(np.sqrt(((abs(x1-x2))**2) + ((abs(y1-y2))**2))))
        
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

