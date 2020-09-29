from __future__ import print_function
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
import vrp_multidepot_solvefuncs as vrp
import pdb

def get_max_dist(data, manager, routing, solution):
    max_route_distance = 0.0
    cost = 0.0
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        #plan_output = 'Route for vehicle {}:\n'.format(vehicle_id + 1)
        route_distance = 0
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

#gets k evenly spaced points along defined segments
def get_segments_points(xs, ys, k):

    # Linear length on the line
    distance = np.cumsum(np.sqrt( np.ediff1d(xs, to_begin=0)**2 + np.ediff1d(ys, to_begin=0)**2 ))
    distance = distance/distance[-1]

    fx, fy = interp1d( distance, xs ), interp1d( distance, ys )

    alpha = np.linspace(0, 1, k)
    x_regular, y_regular = fx(alpha), fy(alpha)

    return x_regular, y_regular

#graphs solution using matplotlib
def graph_solution(data, manager, routing, solution, route_times, vTruck, depotcoords):
    dpi = 192
    fig, ax = plt.subplots(figsize=(1000/dpi, 1000/dpi))
    ax.set_xlim(0,data['bound_length'])
    ax.set_ylim(0, data['bound_length'])
    #ax.axis("off")
    num_nodes = data['num_nodes']
    num_drones = data['num_vehicles_depot']
    num_depots = data['num_depots']
    #ax.plot(data['x'][list(range(1, num_nodes))],data['y'][list(range(1, num_nodes))],'ko',markersize=7, zorder = 0)
    all_route_points = []
    
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        route_points_x = []
        route_points_y = []
        depotcoord = vehicle_id // num_drones
        while not routing.IsEnd(index):
            previous_index = index
            prevnodenum = manager.IndexToNode(previous_index)
            index = solution.Value(routing.NextVar(index))
            nodenum = manager.IndexToNode(index)
            if (prevnodenum != depotcoord and nodenum != depotcoord):
                route_points_x.append(data['x'][nodenum])
                route_points_y.append(data['y'][nodenum])
            elif (prevnodenum == depotcoord and nodenum != depotcoord):
                route_points_x.append(data['x'][prevnodenum])
                route_points_x.append(data['x'][nodenum])
                route_points_y.append(data['y'][prevnodenum])
                route_points_y.append(data['y'][nodenum])
            elif (prevnodenum != depotcoord and nodenum == depotcoord):
                route_points_x.append(data['x'][nodenum])
                route_points_y.append(data['y'][nodenum])
        
        all_route_points.append(route_points_x)
        all_route_points.append(route_points_y)

    last_route_dex = 0
    for i in range(len(route_times)):
        if(route_times[i] > 0):
            last_route_dex = i

    last_node = last_route_dex//num_drones + 1

    print("LAST NODE", last_node)

    truck_leg_times = []
    segment_time = int(((data["bound_length"]*np.sqrt(2))/float(num_depots + 1))/float(vTruck))
    for i in range(last_node):
        truck_leg_times.append(segment_time)
    
    truck_leg_times.append(int(last_node*segment_time))

    drone_leg_times = []
    drone_leg_times.append(0)
    for i in range(last_node):
        maxdronetime = 0
        for j in range(num_drones):
            dronetime = route_times[(i*num_drones)+j]
            if(dronetime > maxdronetime):
                maxdronetime = dronetime
        drone_leg_times.append(int(maxdronetime))

    truck_endtime_legs = []
    truck_endtime_legs.append(truck_leg_times[0] + drone_leg_times[0])
    for i in range(1,last_node+1):
        truck_endtime_legs.append(truck_leg_times[i] + drone_leg_times[i] + truck_endtime_legs[i-1])

    objective = truck_endtime_legs[-1]

    truck_coords = [[],[]]
    currdex = 0
    depotcoords = [0] + depotcoords[0:last_node] + [0]
    for i in range(last_node+1):
        truck_leg_time = truck_leg_times[i]
        truck_wait_time = drone_leg_times[i]

        for j in range(truck_wait_time):
            truck_coords[0].append(depotcoords[currdex])
            truck_coords[1].append(depotcoords[currdex])

        xmove, ymove = get_segments_points([depotcoords[currdex], depotcoords[currdex+1]], [depotcoords[currdex], depotcoords[currdex+1]], truck_leg_time)
        truck_coords[0] = truck_coords[0] + list(xmove)
        truck_coords[1] = truck_coords[1] + list(ymove)

        currdex += 1
    
    #now calculate drones coordinates for each route, then combine to get all drones over time.
    routes_coords = [[],[]]
    route_finish_times = []

    for i in range(len(route_times)):
        route_pts_x = all_route_points[2*i]
        route_pts_y = all_route_points[(2*i)+1]

        if(route_times[i] == 0):
            li = [-1]*objective
            routes_coords[0].append(li)
            routes_coords[1].append(li)
            route_finish_times.append(0)
        else:
            depotnum = i//num_drones
            routestarttime = truck_endtime_legs[depotnum]
            routetime = route_times[i]
            routeendtime = routestarttime + routetime

            route_finish_times.append(routeendtime)

            route_i_coord_x = []
            route_i_coord_y = []

            for j in range(routestarttime):
                route_i_coord_x.append(-1)
                route_i_coord_y.append(-1)

            xmove, ymove = get_segments_points(route_pts_x, route_pts_y, routetime)
            route_i_coord_x = route_i_coord_x + list(xmove)
            route_i_coord_y = route_i_coord_y + list(ymove)

            for j in range(objective - routeendtime):
                route_i_coord_x.append(-1)
                route_i_coord_y.append(-1)
            
            routes_coords[0].append(route_i_coord_x)
            routes_coords[1].append(route_i_coord_y)

    drones_combined_coords_x = []
    drones_combined_coords_y = []

    for i in range(objective):
        #these are lists of the coords of active drones at frame i
        active_drone_coords_x = []
        active_drone_coords_y = []
        for j in range(len(route_times)):
            #each route has its x coordinates list and y coordinates list in routes_coords.
            #route j has its x coords as list at index 2*(j-1)
            #route j has its y coords as list at index 2*(j-1) + 1
            #ex: route 3 has x coords at index 4, y coords at index 5
            route_x = routes_coords[0][j]
            route_y = routes_coords[1][j]

            if(route_x[i] != -1):
                active_drone_coords_x.append(route_x[i])

            if(route_y[i] != -1):
                active_drone_coords_y.append(route_y[i])
            
        drones_combined_coords_x.append(active_drone_coords_x)
        drones_combined_coords_y.append(active_drone_coords_y)

    #now handle routes changing colors as specified times in route_finish_times
    route_finished_indicators = []
    for i in range(len(route_finish_times)):
        #0 means route not done. 1 means route done.
        route_i_indicator = []
        route_i_finish_time = route_finish_times[i]
        unfinished_list = [0]*(route_i_finish_time-1)
        finished_list = [1]*(objective-route_i_finish_time+1)
        route_i_indicator = route_i_indicator + unfinished_list + finished_list
        route_finished_indicators.append(route_i_indicator)
    
    print("\nlength of truck coords x:", len(truck_coords[0]))
    print("length of truck coords y:", len(truck_coords[1]))
    print("length of drone coords x:", len(drones_combined_coords_x))
    print("length of drone coords y:", len(drones_combined_coords_y))
    print("OBJECTIVE:", objective)
    print("\n")

    #for i, j in active_arcs:
    ax.plot([0, data['bound_length']], [0, data['bound_length']], color='r', zorder=0)
    #plt.scatter(xc[1:], yc[1:], c = 'w', linewidth = 6, zorder = 3)
        
    ax.plot([0], [0], c='r', markerSize='10')

    ims = []
    for i in range(objective):
        artists = []
        for j in range(len(route_finished_indicators)):
            route_indicator = route_finished_indicators[j]
            route_i_xpts = all_route_points[2*j]
            route_i_ypts = all_route_points[(2*j)+1]
            if(route_indicator[i] == 0):
                route_i, = ax.plot(route_i_xpts,route_i_ypts, 'o-k', zorder = 0)
            else:
                route_i, = ax.plot(route_i_xpts,route_i_ypts, 'o-', c="0.5", zorder = 0)
            artists.append(route_i)

        truck, = ax.plot([truck_coords[0][i]], [truck_coords[1][i]], color = "blue", marker="s", markerSize='10', animated=True)
        drones = ax.scatter(drones_combined_coords_x[i], drones_combined_coords_y[i], s=165, color = "#1AF527", marker="o", animated=True)

        artists.append(truck)
        artists.append(drones)

        ims.append(artists)
    
    #truck node visit order
    for i in range(1,len(depotcoords)-1):
        ax.plot([depotcoords[i]],[depotcoords[i]],'wo',markersize=10,zorder=1)
        ax.text(depotcoords[i],depotcoords[i],i, ha="center", va="center", fontsize = 10, zorder=2)

    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,repeat_delay=1000)

    writer = animation.FFMpegWriter(fps=20, metadata=dict(artist='Me'), bitrate=1800)
    ani.save("fixed_depot_comparison_solution.mp4", writer=writer)
         
    #plt.legend(handles=legendlines, loc='best', prop={'size': 5})
    #imagename = "solution_singledepotroad.png"
    plt.show()
    #plt.clf()

    return objective


def print_solution(data, manager, routing, solution):
    route_distances = []
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
        
        print(plan_output)
        route_distances.append(route_distance)

        max_route_distance = max(route_distance, max_route_distance)
    print('Maximum of the route distances: {}m'.format(max_route_distance))
    return route_distances


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


def multi_fixed_depot_comparison(numnodes, numdrones, num_dpts, distlim, xcoords, ycoords, vDrone, vTruck):
    """Solve the CVRP problem."""
    
    print("\n\n------------------")
    print("Solving for optimal number of vehicles in 1000x1000 square area for comparison to optimal drone deployment..\n")
    
    num_nodes = int(numnodes) + int(num_dpts)
    numdrones = int(numdrones)
    bound_length = 1000
    num_depots = int(num_dpts)
    max_distance = int(distlim)
    
    #create depot coordinates along the line y=x in the square region
    depotcoords = np.linspace(0, bound_length, num_depots+2, dtype = 'int32').tolist()
    depotcoords.pop(0)
    depotcoords.pop(-1)
    
    #create data for problem
    data = {}

    x = xcoords
    y = ycoords

    data['x'] = depotcoords + x
    data['y'] = depotcoords + y
    
    #put params in data
    data['max_distance'] = max_distance
    data['bound_length'] = bound_length
    data['num_nodes'] = num_nodes
    data['num_depots'] = num_depots
        
    #construct distance matrix based on current depot
    data['distance_matrix'] = vrp.get_dist_mat(data,data['num_nodes'])

    #make penalty of each node larger than the maximum possible total sum of distances
    data['penalty'] = int(np.sqrt(2)*(bound_length)*num_nodes)
    
    starts = []
    ends = []

    #number of vehicles is same as number of drones * number of depots
    data['num_vehicles'] = num_depots*numdrones
    #num_vehicles_depot is number of vehicles per depot, is equal to number of drones available.
    data['num_vehicles_depot'] = numdrones
        
    for i in range(num_depots):
         for j in range(data['num_vehicles_depot']):
             starts.append(i)
             ends.append(i)
        
    data['starts'] = starts
    data['ends'] = ends

    routedistances = []
    used_vehicles = 0
    
    manager, routing, solution = vrp.solvemulti(data)
    if(routing == None):
        print("solution not found...")
        return None

    total_visited, visited_nodes, total_trips, used_vehicles = vrp.get_visited(data, manager, routing, solution)

    if(total_visited < num_nodes - num_depots):
        print("\nWARNING: Could not visit all waypoints with current distance limit. Please make distance limit larger.")

    print("Total Number of Vehicles Available Per Depot:", data['num_vehicles_depot'])
        
    routedistances = print_solution(data, manager, routing, solution)
    routetimes = [int(d/float(vDrone)) for d in routedistances]

    print("\nRoute Distances:", *routedistances)
    print("Route Times:", *routetimes)

    objective = graph_solution(data, manager, routing, solution, routetimes, vTruck, depotcoords)

    print("\nFIXED DEPOT SOLUTION TIME:", objective)

def main():
    """Solve the CVRP problem."""
    if len(sys.argv) != 7:
        print('Should be called as follows: python complete_solver_gurobi.py [number of waypoints] [number of drones] [number of depots] [range of drone] [velocity of drone] [velocity of truck]')
        return
    
    numnodes = int(sys.argv[1])
    numdrones = int(sys.argv[2])
    numdpts = int(sys.argv[3])
    dronerange = np.ceil(float(sys.argv[4]))
    dronevel = float(sys.argv[5])
    truckvel = float(sys.argv[6])
    bound_length = 1000

    xcoords = np.random.randint(low=0,high=bound_length,size=numnodes).tolist()
    ycoords = np.random.randint(low=0,high=bound_length,size=numnodes).tolist()

    multi_fixed_depot_comparison(numnodes, numdrones, numdpts, dronerange, xcoords, ycoords, dronevel, truckvel)
    

if __name__ == '__main__':
    main()