"""Vehicles Routing Problem (VRP)."""

from __future__ import print_function
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
import math
import shapely.geometry as geom


def getdist(x1, y1, x2, y2):
        return int(np.ceil(np.sqrt((abs(x1-x2)**2) + (abs(y1-y2)**2))))
        
#gets number of total visited nodes, and identities of those nodes to be removed for a particular solution
def get_visited(data, manager, routing, solution):
    total_visited = 0
    num_trips = 0
    
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        startdex = manager.IndexToNode(index)
        vehicle_visited = 0
        
        if(not(routing.IsEnd(index))):
            nextdex = manager.IndexToNode(solution.Value(routing.NextVar(index)))
            if(nextdex != 0):
                num_trips += 1     
            
        while not routing.IsEnd(index):
            nodex = manager.IndexToNode(index)
            nextdex = solution.Value(routing.NextVar(index))
            nextnodedex = manager.IndexToNode(nextdex)
            if(nodex != nextnodedex and not(routing.IsEnd(nextdex))):
                vehicle_visited += 1
            previous_index = index
            index = solution.Value(routing.NextVar(index))       
            
        total_visited += vehicle_visited
        
    return total_visited, num_trips

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
    data['vehiclepenalty'] = math.ceil(np.sqrt(data['penalty']))
    
    print("\nNODE PENALTY:", data['penalty'])
    print("VEHICLE PENALTY:", data['vehiclepenalty'])
    
        
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
    #node_roaddists = []
    roadnodecoords = []
    for i in range(0, num_nodes):
        xlen = data['x'][i]
        ylen = data['y'][i]
        sidelen = (xlen + ylen) / 2.0
        roaddist = getdist(sidelen, sidelen, xlen, ylen)
        #node_roaddists.append(roaddist)
        roadnodecoords.append(sidelen)
        data['distance_matrix'][0][i] = roaddist
        data['distance_matrix'][i][0] = roaddist
            
    data['roadnodecoords'] = roadnodecoords
    #max distance is given as a fixed input
    #data['max_distance'] = math.ceil(2*max(node_roaddists))
    #data['max_distance'] = 1500
    
    #print("\nMAX DISTANCE:", data['max_distance'], "\n")  
    
    li = data['distance_matrix']
    for i in range(num_nodes):
        for j in range(num_nodes):
            li[i][j] = int(li[i][j])
            
    #print(pd.DataFrame(li).round(0))
    
    #if setting start and end depots
    #data['starts'] = np.random.randint(low=0,high=num_nodes,size=num_vehicles).tolist()
    #data['ends'] = [0 for i in range(num_vehicles)]
    data['depot'] = 0
    
    del dist_mat
    
    return data

# given xcoords and ycoords of waypoints, create data model.
def create_data_model_fromdata(num_nodes, num_vehicles, xcoords, ycoords):
    """Stores the data for the problem."""
    
    bound_length = 1000
    num_vehicles = int(num_vehicles)
    #only have one depot
    num_depots = 1
    num_nodes = int(num_nodes) + 1
    
    
    data = {}

    #initial "depot" is node 0, at (0,0)
    x = [0] + xcoords
    y = [0] + ycoords

    data['x'] = x
    data['y'] = y

    data['penalty'] = int(np.sqrt(2)*(bound_length)*(num_nodes**3))
    data['vehiclepenalty'] = math.ceil(np.sqrt(data['penalty']))
    
    print("\nNODE PENALTY:", data['penalty'])
    print("VEHICLE PENALTY:", data['vehiclepenalty'])
    
        
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
    #node_roaddists = []
    roadnodecoords = []
    for i in range(0, num_nodes):
        xlen = data['x'][i]
        ylen = data['y'][i]
        sidelen = (xlen + ylen) / 2.0
        roaddist = getdist(sidelen, sidelen, xlen, ylen)
        #node_roaddists.append(roaddist)
        roadnodecoords.append(sidelen)
        data['distance_matrix'][0][i] = roaddist
        data['distance_matrix'][i][0] = roaddist
            
    data['roadnodecoords'] = roadnodecoords
    #max distance is given as a fixed input
    #data['max_distance'] = math.ceil(2*max(node_roaddists))
    #data['max_distance'] = 1500
    
    #print("\nMAX DISTANCE:", data['max_distance'], "\n")  
    
    li = data['distance_matrix']
    for i in range(num_nodes):
        for j in range(num_nodes):
            li[i][j] = int(li[i][j])
            
    #print(pd.DataFrame(li).round(0))
    
    #if setting start and end depots
    #data['starts'] = np.random.randint(low=0,high=num_nodes,size=num_vehicles).tolist()
    #data['ends'] = [0 for i in range(num_vehicles)]
    data['depot'] = 0
    
    del dist_mat
    
    return data

# given xcoords and ycoords of waypoints, create data model.
def create_data_model_fromdata_curved(num_nodes, num_vehicles, xcoords, ycoords, line, depot, dim):
    """Stores the data for the problem."""
    
    bound_length = dim
    num_vehicles = int(num_vehicles)
    #only have one depot
    num_depots = 1
    num_nodes = int(num_nodes) + 1

    data = {}

    #initial "depot" is determined by the depot point given as arg to function
    x = [depot.x] + xcoords
    y = [depot.y] + ycoords

    print(x)
    print(y)

    data['x'] = x
    data['y'] = y

    data['penalty'] = int(np.sqrt(2)*(bound_length)*(num_nodes**3))
    data['vehiclepenalty'] = math.ceil(np.sqrt(data['penalty']))
    
    print("\nNODE PENALTY:", data['penalty'])
    print("VEHICLE PENALTY:", data['vehiclepenalty'])
    
        
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

    #distance between depot and nodes set to shortest distance to road (a general curved road in this case)
    #node_roaddists = []
    roadnodecoordsx = []
    roadnodecoordsy = []
    for i in range(0, num_nodes):
        xlen = data['x'][i]
        ylen = data['y'][i]
        point = geom.Point(xlen, ylen)
        roaddist = point.distance(line)
        point_on_line = line.interpolate(line.project(point))
        roadnodecoordsx.append(point_on_line.x)
        roadnodecoordsy.append(point_on_line.y)
        data['distance_matrix'][0][i] = roaddist
        data['distance_matrix'][i][0] = roaddist
            
    data['roadnodecoordsx'] = roadnodecoordsx
    data['roadnodecoordsy'] = roadnodecoordsy
    #max distance is given as a fixed input
    #data['max_distance'] = math.ceil(2*max(node_roaddists))
    #data['max_distance'] = 1500
    
    #print("\nMAX DISTANCE:", data['max_distance'], "\n")  
    
    li = data['distance_matrix']
    for i in range(num_nodes):
        for j in range(num_nodes):
            li[i][j] = int(li[i][j])
            
    #print(pd.DataFrame(li).round(0))
    
    #if setting start and end depots
    #data['starts'] = np.random.randint(low=0,high=num_nodes,size=num_vehicles).tolist()
    #data['ends'] = [0 for i in range(num_vehicles)]
    data['depot'] = 0
    
    del dist_mat
    
    return data


def print_solution(data, manager, routing, solution):
    """Prints solution on console."""
    print("\n---------\n")
    max_route_distance = 0
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {}:\n'.format(vehicle_id+1)
        #route_distance = 0 - data['vehiclepenalty']
        route_distance = 0
        while not routing.IsEnd(index):
            plan_output += ' {} -> '.format(manager.IndexToNode(index))
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id)
        plan_output += '{}\n'.format(manager.IndexToNode(index))
        #if(route_distance != 0 - data['vehiclepenalty']):
        if(route_distance != 0):
            plan_output += 'Distance of the route: {}\n'.format(route_distance)
        else:
            plan_output += 'Distance of the route: {}\n'.format(0)
        
        if not("0 -> 0" in plan_output):
            print(plan_output)
        max_route_distance = max(route_distance, max_route_distance)
    print('Maximum of the route distances: {}m'.format(max_route_distance))

#graphs solution using matplotlib
def graph_solution(data, manager, routing, solution):
    dpi = 192
    plt.figure(figsize=(1000/dpi, 1000/dpi))
    max_route_distance = 0
    num_nodes = data['num_nodes']
    plt.plot([data['x'][k] for k in list(range(1, num_nodes))],[data['y'][k] for k in list(range(1, num_nodes))],'ko',markersize=7)
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
    plt.clf()
    plt.close()

#graphs solution using matplotlib
def graph_solution_curved(data, manager, routing, solution, line):
    dpi = 192
    plt.figure(figsize=(1000/dpi, 1000/dpi))
    max_route_distance = 0
    num_nodes = data['num_nodes']
    plt.plot([data['x'][k] for k in list(range(1, num_nodes))],[data['y'][k] for k in list(range(1, num_nodes))],'ko',markersize=7)
    xLine, yLine = line.xy
    plt.plot(xLine,yLine,'b', lw=2)
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
                plt.plot(data['roadnodecoordsx'][nodenum],data['roadnodecoordsy'][nodenum],'bD',markersize=7)
                newline, = plt.plot([data['roadnodecoordsx'][nodenum],data['x'][nodenum]],[data['roadnodecoordsy'][nodenum],data['y'][nodenum]], '-',c=colors[vehicle_id],label='$Vehicle {i}$'.format(i=vehicle_id+1))
                linecreated = True
            elif (prevnodenum != 0 and nodenum == 0):
                plt.plot(data['roadnodecoordsx'][prevnodenum],data['roadnodecoordsy'][prevnodenum],'bD',markersize=7)
                newline, = plt.plot([data['x'][prevnodenum],data['roadnodecoordsx'][prevnodenum]],[data['y'][prevnodenum],data['roadnodecoordsy'][prevnodenum]], '-',c=colors[vehicle_id], label='$Vehicle {i}$'.format(i=vehicle_id+1))
            
            if(linecreated):
                legendlines.append(newline)
            
            max_route_distance = max(route_distance, max_route_distance)
     
    plt.legend(handles=legendlines, loc='best', prop={'size': 5})
    imagename = "{i}_vehicle_solution_singledepotroad_curved.png".format(i=data['num_vehicles'])
    plt.xlim(0,data['bound_length'])
    plt.ylim(0, data['bound_length'])
    plt.savefig(imagename, dpi=dpi)
    plt.show()
    plt.clf()
    plt.close()
    
    #graphs solution using matplotlib
def extract_roadnode_data(data, manager, routing, solution):
    routedistances = []
    usednodecoords = []
    print("\n---------\n")
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
        print("Route", i+1, " distance", routedistances[i], " endpoints:", usednodecoords[2*i], usednodecoords[2*i + 1])
    
    """
    dist_mat = np.zeros((numroadnodes, numroadnodes))
    
    for i in range(numroadnodes):
        for j in range(numroadnodes):
            dist_mat[i][j] = getdist(usednodecoords[i],usednodecoords[i],usednodecoords[j],usednodecoords[j])
    """
    
    print("\n---------\n")
            
    return routedistances, usednodecoords

def extract_roadnode_data_curved(data, manager, routing, solution):
    routedistances = []
    usednodecoords = [[],[]]
    print("\n---------\n")
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
                usednodecoords[0].append(data['roadnodecoordsx'][nodenum])
                usednodecoords[1].append(data['roadnodecoordsy'][nodenum])
            elif (prevnodenum != 0 and nodenum == 0):
                usednodecoords[0].append(data['roadnodecoordsx'][prevnodenum])
                usednodecoords[1].append(data['roadnodecoordsy'][prevnodenum])
            
        if(route_distance != 0):
            routedistances.append(route_distance)
     
    #numroadnodes = len(usednodecoords)
    
    for i in range(len(routedistances)):
        print("Route", i+1, " distance", routedistances[i], " endpoints: ", "(", usednodecoords[0][2*i], ", ", usednodecoords[0][2*i + 1], ") ", "(", usednodecoords[1][2*i], ", ", usednodecoords[1][2*i + 1], ") ")
    
    print("\n---------\n")
            
    return routedistances, usednodecoords

def solve_singledepotroad(numnodes, numvehicles, dronerange):
        
    # Instantiate the data problem.
    data = create_data_model(numnodes, numvehicles)
    
    max_distance = math.ceil(dronerange)

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
        max_distance,  # vehicle maximum travel distance
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
    
    #IMPORTANT: SETS FIXED COST FOR EACH VEHICLE USED!!
    #should be less than node penalty to prioritize visiting all nodes before dropping vehicles
    #routing.SetFixedCostOfAllVehicles(data['vehiclepenalty'])

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
        total_visited, num_trips = get_visited(data, manager, routing, solution)
        if(total_visited != numnodes):
            print("\nDid not visit", numnodes - total_visited, "nodes!\n")
            print("Not all nodes were visited. Drone range is likely too low.")
        print_solution(data, manager, routing, solution)
        print("\n------\nNumber of routes:", num_trips)
        graph_solution(data, manager, routing, solution)
        routedistances, usednodecoords = extract_roadnode_data(data, manager, routing, solution)
        vrpdata = {}
        truckdata = {}
        vrpdata['data'] = data
        vrpdata['manager'] = manager
        vrpdata['routing'] = routing
        vrpdata['solution'] = solution
        truckdata['routedistances'] = routedistances
        truckdata['usednodecoords'] = usednodecoords
        return vrpdata, truckdata
    
    else:
        print("No Solution Found.")
        return None

def solve_singledepotroad_fromdata(numnodes, numvehicles, dronerange, xcoords, ycoords):
        
    # Instantiate the data problem.
    data = create_data_model_fromdata(numnodes, numvehicles, xcoords, ycoords)
    
    max_distance = math.ceil(dronerange)

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
        max_distance,  # vehicle maximum travel distance
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
    
    #IMPORTANT: SETS FIXED COST FOR EACH VEHICLE USED!!
    #should be less than node penalty to prioritize visiting all nodes before dropping vehicles
    #routing.SetFixedCostOfAllVehicles(data['vehiclepenalty'])

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
        total_visited, num_trips = get_visited(data, manager, routing, solution)
        if(total_visited != numnodes):
            print("\nDid not visit", numnodes - total_visited, "nodes!\n")
            print("Not all nodes were visited. Drone range is likely too low.")
        print_solution(data, manager, routing, solution)
        print("\n------\nNumber of routes:", num_trips)
        graph_solution(data, manager, routing, solution)
        routedistances, usednodecoords = extract_roadnode_data(data, manager, routing, solution)
        vrpdata = {}
        truckdata = {}
        vrpdata['data'] = data
        vrpdata['manager'] = manager
        vrpdata['routing'] = routing
        vrpdata['solution'] = solution
        truckdata['routedistances'] = routedistances
        truckdata['usednodecoords'] = usednodecoords
        return vrpdata, truckdata
    
    else:
        print("No Solution Found.")
        return None

def solve_singledepotroad_fromdata_curved(numnodes, numvehicles, dronerange, xcoords, ycoords, line, depot, dim):
        
    # Instantiate the data problem.
    data = create_data_model_fromdata_curved(numnodes, numvehicles, xcoords, ycoords, line, depot, dim)
    
    max_distance = math.ceil(dronerange)

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
        max_distance,  # vehicle maximum travel distance
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
    
    #IMPORTANT: SETS FIXED COST FOR EACH VEHICLE USED!!
    #should be less than node penalty to prioritize visiting all nodes before dropping vehicles
    #routing.SetFixedCostOfAllVehicles(data['vehiclepenalty'])

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
        total_visited, num_trips = get_visited(data, manager, routing, solution)
        if(total_visited != numnodes):
            print("\nDid not visit", numnodes - total_visited, "nodes!\n")
            print("Not all nodes were visited. Drone range is likely too low.")
        print_solution(data, manager, routing, solution)
        print("\n------\nNumber of routes:", num_trips)
        graph_solution_curved(data, manager, routing, solution, line)
        routedistances, usednodecoords = extract_roadnode_data_curved(data, manager, routing, solution)
        vrpdata = {}
        truckdata = {}
        vrpdata['data'] = data
        vrpdata['manager'] = manager
        vrpdata['routing'] = routing
        vrpdata['solution'] = solution
        truckdata['routedistances'] = routedistances
        truckdata['usednodecoords'] = usednodecoords
        return vrpdata, truckdata
    
    else:
        print("No Solution Found.")
        return None

def main():
    """Solve the CVRP problem."""
    if len(sys.argv) != 4:
        print('Should be called as follows: python vrp_multipledepots.py [number of nodes] [number of vehicles] [max drone range]')
        return
        
    # Instantiate the data problem.
    data = create_data_model(sys.argv[1], sys.argv[2])
    
    max_distance = math.ceil(float(sys.argv[3]))

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
        max_distance,  # vehicle maximum travel distance
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
    
    #IMPORTANT: SETS FIXED COST FOR EACH VEHICLE USED!!
    #should be less than node penalty to prioritize visiting all nodes before dropping vehicles
    #routing.SetFixedCostOfAllVehicles(data['vehiclepenalty'])

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
        total_visited, num_trips = get_visited(data, manager, routing, solution)
        print("\nNumber of routes:", num_trips)
        if(total_visited != data['num_nodes']-1):
            print("Did not visit", data['num_nodes'] - 1 - total_visited, "nodes")
            print("Not all nodes were visited. Drone range is likely too low.")
        print_solution(data, manager, routing, solution)
        routedistances, usednodecoords = extract_roadnode_data(data, manager, routing, solution)
        graph_solution(data, manager, routing, solution)
    
    else:
        print("No Solution Found.")


if __name__ == '__main__':
    main()