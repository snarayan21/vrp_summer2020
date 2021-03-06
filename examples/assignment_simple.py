"""Vehicles Routing Problem (VRP)."""

from __future__ import print_function
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import numpy as np
import matplotlib.pyplot as plt
import sys
import matplotlib


def circle_points(r, n):
    t = np.linspace(0, 2*np.pi, n)
    x = r * np.cos(t)
    y = r * np.sin(t)
    return x.tolist(), y.tolist()

def create_data_model():
    """Stores the data for the problem."""
    data = {}
    
    data['x'], data['y'] = circle_points(10, 6)
    
    """
    data['distance_matrix'] = [
    [0, 1, 0, 3, 4, 5],
    [1, 0, 1, 2, 3, 0],
    [0, 1, 0, 1, 2, 3],
    [3, 2, 1, 0, 0, 2],
    [4, 3, 2, 0, 0, 1],
    [5, 0, 3, 2, 1, 0]
    ]
    """
    
    data['distance_matrix'] = [
    [0, 10, 20, 30, 40, 0],
    [10, 0, 10, 20, 0, 40],
    [20, 10, 0, 0, 20, 30],
    [30, 20, 0, 0, 10, 20],
    [40, 0, 20, 10, 0, 10],
    [0, 40, 30, 20, 10, 0]
    ]
    
    data['num_vehicles'] = 1
    data['depot'] = 0
    
    return data


def print_solution(data, manager, routing, solution):
    """Prints solution on console."""
    max_route_distance = 0
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
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
    plt.plot(data['x'],data['y'],'ko',markersize=10)
    plt.plot(data['x'][0],data['y'][0],'bD',markersize=10)
    #imagename = "simple_vrp_nodemap.png"
    #plt.savefig(imagename, dpi=dpi)
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
    #imagename = "simple_vrp_solution.png"
    #plt.savefig(imagename, dpi=dpi)
    plt.show()
    plt.clf()

def graph_nodemap(data):
    dpi = 192
    plt.figure(figsize=(1000/dpi, 1000/dpi))
    max_route_distance = 0
    plt.plot(data['x'],data['y'],'ko',markersize=10)
    plt.plot(data['x'][0],data['y'][0],'bD',markersize=10)
    imagename = "simple_vrp_nodemap.png"
    plt.savefig(imagename, dpi=dpi)
    plt.show()

def main():
    """Solve the CVRP problem."""
    # Instantiate the data problem.
    data = create_data_model()
    
    #graph_nodemap(data)

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                           data['num_vehicles'], data['depot'])

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
        30000,  # vehicle maximum travel distance
        True,  # start cumul to zero
        dimension_name)
    distance_dimension = routing.GetDimensionOrDie(dimension_name)
    #distance_dimension.SetGlobalSpanCostCoefficient(100)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if solution:
        print_solution(data, manager, routing, solution)
        #graph_solution(data, manager, routing, solution)


if __name__ == '__main__':
    main()