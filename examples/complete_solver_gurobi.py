from __future__ import print_function
from distutils.spawn import find_executable
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
import shapely.geometry as geom
import postprocessing as postprocessing
import copy
from matplotlib.lines import Line2D

#gets k evenly spaced points along defined segments
def get_segments_points(xs, ys, k):

    # Linear length on the line
    distance = np.cumsum(np.sqrt( np.ediff1d(xs, to_begin=0)**2 + np.ediff1d(ys, to_begin=0)**2 ))
    distance = distance/distance[-1]

    fx, fy = interp1d( distance, xs ), interp1d( distance, ys )

    alpha = np.linspace(0, 1, k)
    x_regular, y_regular = fx(alpha), fy(alpha)

    return x_regular, y_regular

#gets k evenly spaced points along defined segments
def get_road_points_curved(pt1, pt2, line, k):
    dist1 = line.project(pt1)
    dist2 = line.project(pt2)
    diff = 0
    if(dist2 >= dist1):
        if(k > 1):
            diff = (dist2-dist1)/(k)
        else:
            diff = dist2-dist1
        new_points = [line.interpolate(dist1 + (i+1)*(diff)) for i in range(k)]
        xs = []
        ys = []
        for pt in new_points:
            xs.append(pt.x)
            ys.append(pt.y)
    else:
        if(k > 1):
            diff = (dist1-dist2)/(k)
        else:
            diff = dist1-dist2
        new_points = [line.interpolate(dist1 - (i+1)*(diff)) for i in range(k)]
        xs = []
        ys = []
        for pt in new_points:
            xs.append(pt.x)
            ys.append(pt.y)

    return xs, ys

#graphs solution using matplotlib
def graph_solution(data, manager, routing, solution, depart_times, wait_times, deployments, route_times, objective, xc, yc, visit_order, usednodecoords, filename, graph_indicator, postprocessing_indicator):
    if(graph_indicator):
        dpi = 192
        fig, ax = plt.subplots(figsize=(1000/dpi, 1000/dpi))
        ax.set_xlim(0,data['bound_length'])
        ax.set_ylim(0, data['bound_length'])
        #ax.axis("off")
        num_nodes = data['num_nodes']
        #ax.plot(data['x'][list(range(1, num_nodes))],data['y'][list(range(1, num_nodes))],'ko',markersize=7, zorder = 0)
        all_route_points = []
        
        for vehicle_id in range(data['num_vehicles']):
            index = routing.Start(vehicle_id)
            route_points_x = []
            route_points_y = []
            while not routing.IsEnd(index):
                previous_index = index
                prevnodenum = manager.IndexToNode(previous_index)
                index = solution.Value(routing.NextVar(index))
                nodenum = manager.IndexToNode(index)
                if (prevnodenum != 0 and nodenum != 0):
                    #ax.plot([data['x'][prevnodenum],data['x'][nodenum]],[data['y'][prevnodenum],data['y'][nodenum]], '-',c='k', label='$Vehicle {i}$'.format(i=vehicle_id+1), zorder = 1)
                    route_points_x.append(data['x'][nodenum])
                    route_points_y.append(data['y'][nodenum])
                elif (prevnodenum == 0 and nodenum != 0):
                    #plt.plot(data['roadnodecoords'][nodenum],data['roadnodecoords'][nodenum],'bD',markersize=7)
                    #ax.plot([data['roadnodecoords'][nodenum],data['x'][nodenum]],[data['roadnodecoords'][nodenum],data['y'][nodenum]], '-',c='k',label='$Vehicle {i}$'.format(i=vehicle_id+1), zorder = 1)
                    route_points_x.append(data['roadnodecoords'][nodenum])
                    route_points_x.append(data['x'][nodenum])
                    route_points_y.append(data['roadnodecoords'][nodenum])
                    route_points_y.append(data['y'][nodenum])
                elif (prevnodenum != 0 and nodenum == 0):
                    #plt.plot(data['roadnodecoords'][prevnodenum],data['roadnodecoords'][prevnodenum],'bD',markersize=7)
                    #ax.plot([data['x'][prevnodenum],data['roadnodecoords'][prevnodenum]],[data['y'][prevnodenum],data['roadnodecoords'][prevnodenum]], '-',c='k', label='$Vehicle {i}$'.format(i=vehicle_id+1), zorder = 1)
                    route_points_x.append(data['roadnodecoords'][prevnodenum])
                    route_points_y.append(data['roadnodecoords'][prevnodenum])
            
            if(len(route_points_x) > 0 and len(route_points_y) > 0):
                all_route_points.append(route_points_x)
                all_route_points.append(route_points_y)
        
        if(postprocessing_indicator):
            for i in range(len(all_route_points)):
                if(i % 2 == 0):
                    all_route_points[i][0] = usednodecoords[i+1]
                    all_route_points[i][-1] = usednodecoords[i+2]
                else:
                    all_route_points[i][0] = usednodecoords[i]
                    all_route_points[i][-1] = usednodecoords[i+1]

        """ #plot the routes initally all color black (have not been completed)
        for i in range(1,len(route_times)+1):
            #these are the x and y points of route i
            route_i_xpts = all_route_points[2*(i-1)]
            route_i_ypts = all_route_points[2*(i-1)+1]
            ax.plot(route_i_xpts,route_i_ypts, fmt="o-k",label='$Vehicle {i}$'.format(i=vehicle_id+1), zorder = 1) """

        
        #calculate truck coordinates over time
        truck_coords = [[],[]]
        for i in range(len(visit_order)-1):
            startnode = visit_order[i]
            endnode = visit_order[i+1]
            startcoord = usednodecoords[startnode]
            endcoord = usednodecoords[endnode]

            #number of frames where truck is waiting
            truckwaittime = wait_times[endnode]
            startdeparttime = depart_times[startnode]
            enddeparttime = depart_times[endnode]

            #number of frames where truck is moving, not waiting
            truckmovetime = enddeparttime - startdeparttime - truckwaittime

            #generate the coordinates for the frames in which truck is moving
            if(startcoord != endcoord):
                if(truckmovetime < 0):
                    truckmovetime = 0
                xmove, ymove = get_segments_points([startcoord, endcoord], [startcoord, endcoord], truckmovetime)
                truck_coords[0] = truck_coords[0] + list(xmove)
                truck_coords[1] = truck_coords[1] + list(ymove)

            #add in the positions for frames where truck is waiting
            for i in range(truckwaittime):
                truck_coords[0].append(endcoord)
                truck_coords[1].append(endcoord)
        
        #add in last edge from last collection node back to depot
        lastcollectnode = visit_order[-1]
        finalnode = visit_order[0]
        lastcollectcoord = usednodecoords[lastcollectnode]
        finalcoord = usednodecoords[finalnode]
        lastcollectdeparttime = depart_times[lastcollectnode]
        finaltime = objective

        truckmovetime = finaltime - lastcollectdeparttime

        if(lastcollectcoord != finalcoord):
            xmove, ymove = get_segments_points([lastcollectcoord, finalcoord], [lastcollectcoord, finalcoord], truckmovetime)
            truck_coords[0] = truck_coords[0] + list(xmove)
            truck_coords[1] = truck_coords[1] + list(ymove)

        
        #now calculate drones coordinates for each route, then combine to get all drones over time.
        routes_coords = []
        route_finish_times = []
        route_start_times = []

        for i in range(1,len(route_times)+1):
            #endpoints of route i
            p1 = 2*i - 1
            p2 = 2*i

            p1coord = usednodecoords[p1]
            p2coord = usednodecoords[p2]

            #get waypoints of the route from earlier
            #x waypoints of route i are at index 2*(i-1)
            #y waypoints of route i are at index 2*(i-1) + 1
            #ex: route 3 has waypoints xcoord at index 4 and waypoints ycoord at index 5
            route_i_waypoints_x = all_route_points[2*(i-1)]
            route_i_waypoints_y = all_route_points[2*(i-1)+1]
            
            #depart times of truck from endpoints
            p1truckdepart = depart_times[p1]
            p2truckdepart = depart_times[p2]

            #truck departure time from route start point is equal to route start,
            #since deploy points have no waiting time for either truck or drone.
            #this must be the smaller of the two depart times.
            routestarttime = min(p1truckdepart, p2truckdepart)

            #get the starting point of the route.
            startpoint = p1
            startpointcoord = p1coord
            endpoint = p2
            endpointcoord = p2coord
            if routestarttime != p1truckdepart:
                startpoint = p2
                startpointcoord = p2coord
                endpoint = p1
                endpointcoord = p1coord

            #make sure the route waypoints are ordered in the correct direction
            #depending on the coordinate of the starting point of the route
            if route_i_waypoints_x[0] != startpointcoord:
                route_i_waypoints_x = route_i_waypoints_x[::-1]
                route_i_waypoints_y = route_i_waypoints_y[::-1]

            #time at which truck picks up drone.
            truckrouteendtime = max(p1truckdepart, p2truckdepart)

            #time between truck deployment and departure from pickup node
            deploypickuptime = truckrouteendtime - routestarttime

            #time the drone was in the air, moving
            routemovetime = route_times[i]

            #drone ends route at routestarttime + routemovetime
            route_finish_times.append(routestarttime + routemovetime)
            route_start_times.append(routestarttime)

            route_i_coords_x = []
            route_i_coords_y = []

            for i in range(routestarttime):
                #-1 signifies that drone was not in the air or waiting for pickup at all. drone was with the truck.
                route_i_coords_x.append(-1)
                route_i_coords_y.append(-1)

            xmove, ymove = get_segments_points(route_i_waypoints_x, route_i_waypoints_y, routemovetime)
            route_i_coords_x = route_i_coords_x + list(xmove)
            route_i_coords_y = route_i_coords_y + list(ymove)

            #handle case where drone has to wait to be picked up
            routewaittime = 0
            if routemovetime < deploypickuptime:
                routewaittime = deploypickuptime - routemovetime
                for i in range(routewaittime):
                    route_i_coords_x.append(endpointcoord)
                    route_i_coords_y.append(endpointcoord)
            
            routefinishedtime = routestarttime + routemovetime + routewaittime

            #time between route being done and entire process being completed
            routedonetime = objective - routefinishedtime

            for i in range(routedonetime):
                #drone is with the truck again.
                route_i_coords_x.append(-1)
                route_i_coords_y.append(-1)

            routes_coords.append(route_i_coords_x)
            routes_coords.append(route_i_coords_y)

        drones_combined_coords_x = []
        drones_combined_coords_y = []

        for i in range(objective):
            #these are lists of the coords of active drones at frame i
            active_drone_coords_x = []
            active_drone_coords_y = []
            for j in range(1,len(route_times)+1):
                #each route has its x coordinates list and y coordinates list in routes_coords.
                #route j has its x coords as list at index 2*(j-1)
                #route j has its y coords as list at index 2*(j-1) + 1
                #ex: route 3 has x coords at index 4, y coords at index 5
                route_x = routes_coords[2*(j-1)]
                route_y = routes_coords[2*(j-1)+1]

                if(route_x[i] != -1):
                    active_drone_coords_x.append(route_x[i])

                if(route_y[i] != -1):
                    active_drone_coords_y.append(route_y[i])
                
            drones_combined_coords_x.append(active_drone_coords_x)
            drones_combined_coords_y.append(active_drone_coords_y)

        """ print("length of truck coords x:", len(truck_coords[0]))
        print("length of truck coords y:", len(truck_coords[1]))
        print("length of drone coords x:", len(drones_combined_coords_x))
        print("length of drone coords y:", len(drones_combined_coords_y)) """

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

        #for i, j in active_arcs:
        ax.plot([xc[0], data['bound_length']], [yc[0], data['bound_length']], color='r', zorder=0)
        #plt.scatter(xc[1:], yc[1:], c = 'w', linewidth = 6, zorder = 3)
            
        ax.plot([xc[0]], [yc[0]], c='r', markerSize='10')

        ims = []
        for i in range(min(len(truck_coords[0]), len(drones_combined_coords_x), objective)):
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
        for i in range(1,len(visit_order)):
            dex = visit_order[i]
            ax.plot(xc[dex],yc[dex],'wo',markersize=10,zorder=1)
            ax.text(xc[dex],yc[dex],i, ha="center", va="center", fontsize = 10, zorder=2)

        ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,repeat_delay=1000)

        writer = animation.FFMpegWriter(fps=20, metadata=dict(artist='Me'), bitrate=1800)
        complete_filename = filename + ".mp4"
        ani.save(complete_filename, writer=writer)
            
        #plt.legend(handles=legendlines, loc='best', prop={'size': 5})
        #imagename = "solution_singledepotroad.png"
        #plt.show()
        plt.clf()
        plt.close()
    else:
        print("\n not graphing this one lol.")

#graphs solution using matplotlib
"""
def graph_solution(data, manager, routing, solution, depart_times, wait_times, deployments, route_times, objective, xc, yc, visit_order, usednodecoords, filename, graph_indicator):
    if(graph_indicator):
        dpi = 192
        fig, ax = plt.subplots(figsize=(1000/dpi, 1000/dpi))
        ax.set_xlim(0,data['bound_length'])
        ax.set_ylim(0, data['bound_length'])
        #ax.axis("off")
        num_nodes = data['num_nodes']
        #ax.plot(data['x'][list(range(1, num_nodes))],data['y'][list(range(1, num_nodes))],'ko',markersize=7, zorder = 0)
        all_route_points = []
        
        for vehicle_id in range(data['num_vehicles']):
            index = routing.Start(vehicle_id)
            route_points_x = []
            route_points_y = []
            while not routing.IsEnd(index):
                previous_index = index
                prevnodenum = manager.IndexToNode(previous_index)
                index = solution.Value(routing.NextVar(index))
                nodenum = manager.IndexToNode(index)
                if (prevnodenum != 0 and nodenum != 0):
                    #ax.plot([data['x'][prevnodenum],data['x'][nodenum]],[data['y'][prevnodenum],data['y'][nodenum]], '-',c='k', label='$Vehicle {i}$'.format(i=vehicle_id+1), zorder = 1)
                    route_points_x.append(data['x'][nodenum])
                    route_points_y.append(data['y'][nodenum])
                elif (prevnodenum == 0 and nodenum != 0):
                    #plt.plot(data['roadnodecoords'][nodenum],data['roadnodecoords'][nodenum],'bD',markersize=7)
                    #ax.plot([data['roadnodecoords'][nodenum],data['x'][nodenum]],[data['roadnodecoords'][nodenum],data['y'][nodenum]], '-',c='k',label='$Vehicle {i}$'.format(i=vehicle_id+1), zorder = 1)
                    route_points_x.append(data['roadnodecoords'][nodenum])
                    route_points_x.append(data['x'][nodenum])
                    route_points_y.append(data['roadnodecoords'][nodenum])
                    route_points_y.append(data['y'][nodenum])
                elif (prevnodenum != 0 and nodenum == 0):
                    #plt.plot(data['roadnodecoords'][prevnodenum],data['roadnodecoords'][prevnodenum],'bD',markersize=7)
                    #ax.plot([data['x'][prevnodenum],data['roadnodecoords'][prevnodenum]],[data['y'][prevnodenum],data['roadnodecoords'][prevnodenum]], '-',c='k', label='$Vehicle {i}$'.format(i=vehicle_id+1), zorder = 1)
                    route_points_x.append(data['roadnodecoords'][prevnodenum])
                    route_points_y.append(data['roadnodecoords'][prevnodenum])
            
            if(len(route_points_x) > 0 and len(route_points_y) > 0):
                all_route_points.append(route_points_x)
                all_route_points.append(route_points_y)
        
        #calculate truck coordinates over time
        truck_coords = [[],[]]
        for i in range(len(visit_order)-1):
            startnode = visit_order[i]
            endnode = visit_order[i+1]
            startcoord = usednodecoords[startnode]
            endcoord = usednodecoords[endnode]

            #number of frames where truck is waiting
            truckwaittime = wait_times[endnode]
            startdeparttime = depart_times[startnode]
            enddeparttime = depart_times[endnode]

            #number of frames where truck is moving, not waiting
            truckmovetime = enddeparttime - startdeparttime - truckwaittime

            #generate the coordinates for the frames in which truck is moving
            if(startcoord != endcoord):
                if(truckmovetime < 0):
                    truckmovetime = 0
                xmove, ymove = get_segments_points([startcoord, endcoord], [startcoord, endcoord], truckmovetime)
                truck_coords[0] = truck_coords[0] + list(xmove)
                truck_coords[1] = truck_coords[1] + list(ymove)

            #add in the positions for frames where truck is waiting
            for i in range(truckwaittime):
                truck_coords[0].append(endcoord)
                truck_coords[1].append(endcoord)
        
        #add in last edge from last collection node back to depot
        lastcollectnode = visit_order[-1]
        finalnode = visit_order[0]
        lastcollectcoord = usednodecoords[lastcollectnode]
        finalcoord = usednodecoords[finalnode]
        lastcollectdeparttime = depart_times[lastcollectnode]
        finaltime = objective

        truckmovetime = finaltime - lastcollectdeparttime

        if(lastcollectcoord != finalcoord):
            xmove, ymove = get_segments_points([lastcollectcoord, finalcoord], [lastcollectcoord, finalcoord], truckmovetime)
            truck_coords[0] = truck_coords[0] + list(xmove)
            truck_coords[1] = truck_coords[1] + list(ymove)

        
        #now calculate drones coordinates for each route, then combine to get all drones over time.
        routes_coords = []
        route_finish_times = []
        route_start_times = []

        for i in range(1,len(route_times)+1):
            #endpoints of route i
            p1 = 2*i - 1
            p2 = 2*i

            p1coord = usednodecoords[p1]
            p2coord = usednodecoords[p2]

            #get waypoints of the route from earlier
            #x waypoints of route i are at index 2*(i-1)
            #y waypoints of route i are at index 2*(i-1) + 1
            #ex: route 3 has waypoints xcoord at index 4 and waypoints ycoord at index 5
            route_i_waypoints_x = all_route_points[2*(i-1)]
            route_i_waypoints_y = all_route_points[2*(i-1)+1]
            
            #depart times of truck from endpoints
            p1truckdepart = depart_times[p1]
            p2truckdepart = depart_times[p2]

            #truck departure time from route start point is equal to route start,
            #since deploy points have no waiting time for either truck or drone.
            #this must be the smaller of the two depart times.
            routestarttime = min(p1truckdepart, p2truckdepart)

            #get the starting point of the route.
            startpoint = p1
            startpointcoord = p1coord
            endpoint = p2
            endpointcoord = p2coord
            if routestarttime != p1truckdepart:
                startpoint = p2
                startpointcoord = p2coord
                endpoint = p1
                endpointcoord = p1coord

            #make sure the route waypoints are ordered in the correct direction
            #depending on the coordinate of the starting point of the route
            if route_i_waypoints_x[0] != startpointcoord:
                route_i_waypoints_x = route_i_waypoints_x[::-1]
                route_i_waypoints_y = route_i_waypoints_y[::-1]

            #time at which truck picks up drone.
            truckrouteendtime = max(p1truckdepart, p2truckdepart)

            #time between truck deployment and departure from pickup node
            deploypickuptime = truckrouteendtime - routestarttime

            #time the drone was in the air, moving
            routemovetime = route_times[i]

            #drone ends route at routestarttime + routemovetime
            route_finish_times.append(routestarttime + routemovetime)
            route_start_times.append(routestarttime)

            route_i_coords_x = []
            route_i_coords_y = []

            for i in range(routestarttime):
                #-1 signifies that drone was not in the air or waiting for pickup at all. drone was with the truck.
                route_i_coords_x.append(-1)
                route_i_coords_y.append(-1)

            xmove, ymove = get_segments_points(route_i_waypoints_x, route_i_waypoints_y, routemovetime)
            route_i_coords_x = route_i_coords_x + list(xmove)
            route_i_coords_y = route_i_coords_y + list(ymove)

            #handle case where drone has to wait to be picked up
            routewaittime = 0
            if routemovetime < deploypickuptime:
                routewaittime = deploypickuptime - routemovetime
                for i in range(routewaittime):
                    route_i_coords_x.append(endpointcoord)
                    route_i_coords_y.append(endpointcoord)
            
            routefinishedtime = routestarttime + routemovetime + routewaittime

            #time between route being done and entire process being completed
            routedonetime = objective - routefinishedtime

            for i in range(routedonetime):
                #drone is with the truck again.
                route_i_coords_x.append(-1)
                route_i_coords_y.append(-1)

            routes_coords.append(route_i_coords_x)
            routes_coords.append(route_i_coords_y)

        drones_combined_coords_x = []
        drones_combined_coords_y = []

        for i in range(objective):
            #these are lists of the coords of active drones at frame i
            active_drone_coords_x = []
            active_drone_coords_y = []
            for j in range(1,len(route_times)+1):
                #each route has its x coordinates list and y coordinates list in routes_coords.
                #route j has its x coords as list at index 2*(j-1)
                #route j has its y coords as list at index 2*(j-1) + 1
                #ex: route 3 has x coords at index 4, y coords at index 5
                route_x = routes_coords[2*(j-1)]
                route_y = routes_coords[2*(j-1)+1]

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

        #for i, j in active_arcs:
        ax.plot([xc[0], data['bound_length']], [yc[0], data['bound_length']], color='r', zorder=0)
        #plt.scatter(xc[1:], yc[1:], c = 'w', linewidth = 6, zorder = 3)
            
        ax.plot([xc[0]], [yc[0]], c='r', markerSize='10')

        ims = []
        for i in range(min(len(truck_coords[0]), len(drones_combined_coords_x), objective)):
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
        for i in range(1,len(visit_order)):
            dex = visit_order[i]
            ax.plot(xc[dex],yc[dex],'wo',markersize=10,zorder=1)
            ax.text(xc[dex],yc[dex],i, ha="center", va="center", fontsize = 10, zorder=2)

        ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,repeat_delay=1000)

        writer = animation.FFMpegWriter(fps=20, metadata=dict(artist='Me'), bitrate=1800)
        complete_filename = filename + ".mp4"
        ani.save(complete_filename, writer=writer)
            
        #plt.legend(handles=legendlines, loc='best', prop={'size': 5})
        #imagename = "solution_singledepotroad.png"
        #plt.show()
        plt.clf()
        plt.close()
    else:
        print("\n not graphing this one lol.")
"""

#graphs solution using matplotlib
def graph_solution_curved(data, manager, routing, solution, depart_times, wait_times, deployments, route_times, objective, xc, yc, visit_order, usednodecoords, filename, graph_indicator, line):
    if(graph_indicator):
        dpi = 192
        fig, ax = plt.subplots(figsize=(1000/dpi, 1000/dpi))
        ax.set_xlim(0,data['bound_length'])
        ax.set_ylim(0, data['bound_length'])
        #ax.axis("off")
        num_nodes = data['num_nodes']
        #ax.plot(data['x'][list(range(1, num_nodes))],data['y'][list(range(1, num_nodes))],'ko',markersize=7, zorder = 0)
        all_route_points = []
        
        for vehicle_id in range(data['num_vehicles']):
            index = routing.Start(vehicle_id)
            route_points_x = []
            route_points_y = []
            while not routing.IsEnd(index):
                previous_index = index
                prevnodenum = manager.IndexToNode(previous_index)
                index = solution.Value(routing.NextVar(index))
                nodenum = manager.IndexToNode(index)
                if (prevnodenum != 0 and nodenum != 0):
                    #ax.plot([data['x'][prevnodenum],data['x'][nodenum]],[data['y'][prevnodenum],data['y'][nodenum]], '-',c='k', label='$Vehicle {i}$'.format(i=vehicle_id+1), zorder = 1)
                    route_points_x.append(data['x'][nodenum])
                    route_points_y.append(data['y'][nodenum])
                elif (prevnodenum == 0 and nodenum != 0):
                    #plt.plot(data['roadnodecoords'][nodenum],data['roadnodecoords'][nodenum],'bD',markersize=7)
                    #ax.plot([data['roadnodecoords'][nodenum],data['x'][nodenum]],[data['roadnodecoords'][nodenum],data['y'][nodenum]], '-',c='k',label='$Vehicle {i}$'.format(i=vehicle_id+1), zorder = 1)
                    route_points_x.append(data['roadnodecoordsx'][nodenum])
                    route_points_x.append(data['x'][nodenum])
                    route_points_y.append(data['roadnodecoordsy'][nodenum])
                    route_points_y.append(data['y'][nodenum])
                elif (prevnodenum != 0 and nodenum == 0):
                    #plt.plot(data['roadnodecoords'][prevnodenum],data['roadnodecoords'][prevnodenum],'bD',markersize=7)
                    #ax.plot([data['x'][prevnodenum],data['roadnodecoords'][prevnodenum]],[data['y'][prevnodenum],data['roadnodecoords'][prevnodenum]], '-',c='k', label='$Vehicle {i}$'.format(i=vehicle_id+1), zorder = 1)
                    route_points_x.append(data['roadnodecoordsx'][prevnodenum])
                    route_points_y.append(data['roadnodecoordsy'][prevnodenum])
            
            if(len(route_points_x) > 0 and len(route_points_y) > 0):
                all_route_points.append(route_points_x)
                all_route_points.append(route_points_y)

        """ #plot the routes initally all color black (have not been completed)
        for i in range(1,len(route_times)+1):
            #these are the x and y points of route i
            route_i_xpts = all_route_points[2*(i-1)]
            route_i_ypts = all_route_points[2*(i-1)+1]
            ax.plot(route_i_xpts,route_i_ypts, fmt="o-k",label='$Vehicle {i}$'.format(i=vehicle_id+1), zorder = 1) """

        
        #calculate truck coordinates over time
        truck_coords = [[],[]]
        for i in range(len(visit_order)-1):
            startnode = visit_order[i]
            endnode = visit_order[i+1]
            startcoordx = usednodecoords[0][startnode]
            startcoordy = usednodecoords[1][startnode]
            endcoordx = usednodecoords[0][endnode]
            endcoordy = usednodecoords[1][endnode]

            #number of frames where truck is waiting
            truckwaittime = wait_times[endnode]
            startdeparttime = depart_times[startnode]
            enddeparttime = depart_times[endnode]

            #number of frames where truck is moving, not waiting
            truckmovetime = enddeparttime - startdeparttime - truckwaittime

            #generate the coordinates for the frames in which truck is moving
            if(startcoordx != endcoordx or startcoordy != endcoordy):
                if(truckmovetime < 0):
                    truckmovetime = 0
                xmove, ymove = get_road_points_curved(geom.Point(startcoordx, startcoordy), geom.Point(endcoordx, endcoordy), line, truckmovetime)
                truck_coords[0] = truck_coords[0] + list(xmove)
                truck_coords[1] = truck_coords[1] + list(ymove)

            #add in the positions for frames where truck is waiting
            for i in range(truckwaittime):
                truck_coords[0].append(endcoordx)
                truck_coords[1].append(endcoordy)
        
        #add in last edge from last collection node back to depot
        lastcollectnode = visit_order[-1]
        finalnode = visit_order[0]
        lastcollectcoordx = usednodecoords[0][lastcollectnode]
        lastcollectcoordy = usednodecoords[1][lastcollectnode]
        finalcoordx = usednodecoords[0][finalnode]
        finalcoordy = usednodecoords[1][finalnode]
        lastcollectdeparttime = depart_times[lastcollectnode]
        finaltime = objective

        truckmovetime = finaltime - lastcollectdeparttime

        if(lastcollectcoordx != finalcoordx or lastcollectcoordy != finalcoordy):
            xmove, ymove = get_road_points_curved(geom.Point(lastcollectcoordx, lastcollectcoordy), geom.Point(finalcoordx, finalcoordy), line, truckmovetime)
            truck_coords[0] = truck_coords[0] + list(xmove)
            truck_coords[1] = truck_coords[1] + list(ymove)

        
        #now calculate drones coordinates for each route, then combine to get all drones over time.
        routes_coords = []
        route_finish_times = []
        route_start_times = []

        for i in range(1,len(route_times)+1):
            #endpoints of route i
            p1 = 2*i - 1
            p2 = 2*i

            p1coordx = usednodecoords[0][p1]
            p1coordy = usednodecoords[1][p1]
            p2coordx = usednodecoords[0][p2]
            p2coordy = usednodecoords[1][p2]

            #get waypoints of the route from earlier
            #x waypoints of route i are at index 2*(i-1)
            #y waypoints of route i are at index 2*(i-1) + 1
            #ex: route 3 has waypoints xcoord at index 4 and waypoints ycoord at index 5
            route_i_waypoints_x = all_route_points[2*(i-1)]
            route_i_waypoints_y = all_route_points[2*(i-1)+1]
            
            #depart times of truck from endpoints
            p1truckdepart = depart_times[p1]
            p2truckdepart = depart_times[p2]

            #truck departure time from route start point is equal to route start,
            #since deploy points have no waiting time for either truck or drone.
            #this must be the smaller of the two depart times.
            routestarttime = min(p1truckdepart, p2truckdepart)

            #get the starting point of the route.
            startpoint = p1
            startpointcoordx = p1coordx
            startpointcoordy = p1coordy
            endpoint = p2
            endpointcoordx = p2coordx
            endpointcoordy = p2coordy
            if routestarttime != p1truckdepart:
                startpoint = p2
                startpointcoordx = p2coordx
                startpointcoordy = p2coordy
                endpoint = p1
                endpointcoordx = p1coordx
                endpointcoordy = p1coordy

            #make sure the route waypoints are ordered in the correct direction
            #depending on the coordinate of the starting point of the route
            if route_i_waypoints_x[0] != startpointcoordx:
                route_i_waypoints_x = route_i_waypoints_x[::-1]
                route_i_waypoints_y = route_i_waypoints_y[::-1]

            #time at which truck picks up drone.
            truckrouteendtime = max(p1truckdepart, p2truckdepart)

            #time between truck deployment and departure from pickup node
            deploypickuptime = truckrouteendtime - routestarttime

            #time the drone was in the air, moving
            routemovetime = route_times[i]

            #drone ends route at routestarttime + routemovetime
            route_start_times.append(routestarttime)
            route_finish_times.append(routestarttime + routemovetime)

            route_i_coords_x = []
            route_i_coords_y = []

            for i in range(routestarttime):
                #-1 signifies that drone was not in the air or waiting for pickup at all. drone was with the truck.
                route_i_coords_x.append(-1)
                route_i_coords_y.append(-1)

            xmove, ymove = get_segments_points(route_i_waypoints_x, route_i_waypoints_y, routemovetime)
            route_i_coords_x = route_i_coords_x + list(xmove)
            route_i_coords_y = route_i_coords_y + list(ymove)

            #handle case where drone has to wait to be picked up
            routewaittime = 0
            if routemovetime < deploypickuptime:
                routewaittime = deploypickuptime - routemovetime
                for i in range(routewaittime):
                    route_i_coords_x.append(endpointcoordx)
                    route_i_coords_y.append(endpointcoordy)
            
            routefinishedtime = routestarttime + routemovetime + routewaittime

            #time between route being done and entire process being completed
            routedonetime = objective - routefinishedtime

            for i in range(routedonetime):
                #drone is with the truck again.
                route_i_coords_x.append(-1)
                route_i_coords_y.append(-1)

            routes_coords.append(route_i_coords_x)
            routes_coords.append(route_i_coords_y)

        drones_combined_coords_x = []
        drones_combined_coords_y = []

        for i in range(objective):
            #these are lists of the coords of active drones at frame i
            active_drone_coords_x = []
            active_drone_coords_y = []
            for j in range(1,len(route_times)+1):
                #each route has its x coordinates list and y coordinates list in routes_coords.
                #route j has its x coords as list at index 2*(j-1)
                #route j has its y coords as list at index 2*(j-1) + 1
                #ex: route 3 has x coords at index 4, y coords at index 5
                route_x = routes_coords[2*(j-1)]
                route_y = routes_coords[2*(j-1)+1]

                if(route_x[i] != -1):
                    active_drone_coords_x.append(route_x[i])

                if(route_y[i] != -1):
                    active_drone_coords_y.append(route_y[i])
                
            drones_combined_coords_x.append(active_drone_coords_x)
            drones_combined_coords_y.append(active_drone_coords_y)

        """ print("length of truck coords x:", len(truck_coords[0]))
        print("length of truck coords y:", len(truck_coords[1]))
        print("length of drone coords x:", len(drones_combined_coords_x))
        print("length of drone coords y:", len(drones_combined_coords_y)) """

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

        #plot road
        xLine, yLine = line.xy
        plt.plot(xLine,yLine,'r', lw=2, zorder=0)

        ims = []
        for i in range(min(len(truck_coords[0]), len(drones_combined_coords_x), objective)):
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
        for i in range(1,len(visit_order)):
            dex = visit_order[i]
            ax.plot(xc[dex],yc[dex],'wo',markersize=10,zorder=1)
            ax.text(xc[dex],yc[dex],i, ha="center", va="center", fontsize = 10, zorder=2)

        ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,repeat_delay=1000)

        writer = animation.FFMpegWriter(fps=20, metadata=dict(artist='Me'), bitrate=1800)
        complete_filename = filename + ".mp4"
        ani.save(complete_filename, writer=writer)
            
        #plt.legend(handles=legendlines, loc='best', prop={'size': 5})
        #imagename = "solution_singledepotroad.png"
        plt.show()
        plt.clf()
        plt.close()
    else:
        print("\n not graphing this one lol.")

#graphs solution using matplotlib
def graph_solution_curved_strip(data, manager, routing, solution, depart_times, wait_times, deployments, route_times, objective, xc, yc, visit_order, usednodecoords, filename, graph_indicator, line, numdrones, depot):
    if(graph_indicator):
        #ax.axis("off")
        num_nodes = data['num_nodes']
        #ax.plot(data['x'][list(range(1, num_nodes))],data['y'][list(range(1, num_nodes))],'ko',markersize=7, zorder = 0)
        all_route_points = []
        
        for vehicle_id in range(data['num_vehicles']):
            index = routing.Start(vehicle_id)
            route_points_x = []
            route_points_y = []
            while not routing.IsEnd(index):
                previous_index = index
                prevnodenum = manager.IndexToNode(previous_index)
                index = solution.Value(routing.NextVar(index))
                nodenum = manager.IndexToNode(index)
                if (prevnodenum != 0 and nodenum != 0):
                    #ax.plot([data['x'][prevnodenum],data['x'][nodenum]],[data['y'][prevnodenum],data['y'][nodenum]], '-',c='k', label='$Vehicle {i}$'.format(i=vehicle_id+1), zorder = 1)
                    route_points_x.append(data['x'][nodenum])
                    route_points_y.append(data['y'][nodenum])
                elif (prevnodenum == 0 and nodenum != 0):
                    #plt.plot(data['roadnodecoords'][nodenum],data['roadnodecoords'][nodenum],'bD',markersize=7)
                    #ax.plot([data['roadnodecoords'][nodenum],data['x'][nodenum]],[data['roadnodecoords'][nodenum],data['y'][nodenum]], '-',c='k',label='$Vehicle {i}$'.format(i=vehicle_id+1), zorder = 1)
                    route_points_x.append(data['roadnodecoordsx'][nodenum])
                    route_points_x.append(data['x'][nodenum])
                    route_points_y.append(data['roadnodecoordsy'][nodenum])
                    route_points_y.append(data['y'][nodenum])
                elif (prevnodenum != 0 and nodenum == 0):
                    #plt.plot(data['roadnodecoords'][prevnodenum],data['roadnodecoords'][prevnodenum],'bD',markersize=7)
                    #ax.plot([data['x'][prevnodenum],data['roadnodecoords'][prevnodenum]],[data['y'][prevnodenum],data['roadnodecoords'][prevnodenum]], '-',c='k', label='$Vehicle {i}$'.format(i=vehicle_id+1), zorder = 1)
                    route_points_x.append(data['roadnodecoordsx'][prevnodenum])
                    route_points_y.append(data['roadnodecoordsy'][prevnodenum])
            
            if(len(route_points_x) > 0 and len(route_points_y) > 0):
                all_route_points.append(route_points_x)
                all_route_points.append(route_points_y)

        """ #plot the routes initally all color black (have not been completed)
        for i in range(1,len(route_times)+1):
            #these are the x and y points of route i
            route_i_xpts = all_route_points[2*(i-1)]
            route_i_ypts = all_route_points[2*(i-1)+1]
            ax.plot(route_i_xpts,route_i_ypts, fmt="o-k",label='$Vehicle {i}$'.format(i=vehicle_id+1), zorder = 1) """

        
        #calculate truck coordinates over time
        truck_coords = [[],[]]
        for i in range(len(visit_order)-1):
            startnode = visit_order[i]
            endnode = visit_order[i+1]
            startcoordx = usednodecoords[0][startnode]
            startcoordy = usednodecoords[1][startnode]
            endcoordx = usednodecoords[0][endnode]
            endcoordy = usednodecoords[1][endnode]

            #number of frames where truck is waiting
            truckwaittime = wait_times[endnode]
            startdeparttime = depart_times[startnode]
            enddeparttime = depart_times[endnode]

            #number of frames where truck is moving, not waiting
            truckmovetime = enddeparttime - startdeparttime - truckwaittime

            #generate the coordinates for the frames in which truck is moving
            if(startcoordx != endcoordx or startcoordy != endcoordy):
                if(truckmovetime < 0):
                    truckmovetime = 0
                xmove, ymove = get_road_points_curved(geom.Point(startcoordx, startcoordy), geom.Point(endcoordx, endcoordy), line, truckmovetime)
                truck_coords[0] = truck_coords[0] + list(xmove)
                truck_coords[1] = truck_coords[1] + list(ymove)

            #add in the positions for frames where truck is waiting
            for i in range(truckwaittime):
                truck_coords[0].append(endcoordx)
                truck_coords[1].append(endcoordy)
        
        #add in last edge from last collection node back to depot
        lastcollectnode = visit_order[-1]
        finalnode = visit_order[0]
        lastcollectcoordx = usednodecoords[0][lastcollectnode]
        lastcollectcoordy = usednodecoords[1][lastcollectnode]
        finalcoordx = usednodecoords[0][finalnode]
        finalcoordy = usednodecoords[1][finalnode]
        lastcollectdeparttime = depart_times[lastcollectnode]
        finaltime = objective

        truckmovetime = finaltime - lastcollectdeparttime

        if(lastcollectcoordx != finalcoordx or lastcollectcoordy != finalcoordy):
            xmove, ymove = get_road_points_curved(geom.Point(lastcollectcoordx, lastcollectcoordy), geom.Point(finalcoordx, finalcoordy), line, truckmovetime)
            truck_coords[0] = truck_coords[0] + list(xmove)
            truck_coords[1] = truck_coords[1] + list(ymove)

        
        #now calculate drones coordinates for each route, then combine to get all drones over time.
        routes_coords = []
        route_finish_times = []
        route_start_times = []
        route_wait_times = []
        flipped = []

        for i in range(1,len(route_times)+1):
            #endpoints of route i
            p1 = 2*i - 1
            p2 = 2*i

            p1coordx = usednodecoords[0][p1]
            p1coordy = usednodecoords[1][p1]
            p2coordx = usednodecoords[0][p2]
            p2coordy = usednodecoords[1][p2]

            #get waypoints of the route from earlier
            #x waypoints of route i are at index 2*(i-1)
            #y waypoints of route i are at index 2*(i-1) + 1
            #ex: route 3 has waypoints xcoord at index 4 and waypoints ycoord at index 5
            route_i_waypoints_x = all_route_points[2*(i-1)]
            route_i_waypoints_y = all_route_points[2*(i-1)+1]
            
            #depart times of truck from endpoints
            p1truckdepart = depart_times[p1]
            p2truckdepart = depart_times[p2]

            #truck departure time from route start point is equal to route start,
            #since deploy points have no waiting time for either truck or drone.
            #this must be the smaller of the two depart times.
            routestarttime = min(p1truckdepart, p2truckdepart)

            #get the starting point of the route.
            startpoint = p1
            startpointcoordx = p1coordx
            startpointcoordy = p1coordy
            endpoint = p2
            endpointcoordx = p2coordx
            endpointcoordy = p2coordy
            if routestarttime != p1truckdepart:
                startpoint = p2
                startpointcoordx = p2coordx
                startpointcoordy = p2coordy
                endpoint = p1
                endpointcoordx = p1coordx
                endpointcoordy = p1coordy

            #make sure the route waypoints are ordered in the correct direction
            #depending on the coordinate of the starting point of the route
            if route_i_waypoints_x[0] != startpointcoordx:
                route_i_waypoints_x = route_i_waypoints_x[::-1]
                route_i_waypoints_y = route_i_waypoints_y[::-1]
                flipped.append(True)
            else:
                flipped.append(False)

            #time at which truck picks up drone.
            truckrouteendtime = max(p1truckdepart, p2truckdepart)

            #time between truck deployment and departure from pickup node
            deploypickuptime = truckrouteendtime - routestarttime

            #time the drone was in the air, moving
            routemovetime = route_times[i]

            #drone ends route at routestarttime + routemovetime
            route_start_times.append(routestarttime-1)
            route_finish_times.append(routestarttime + routemovetime-1)

            route_i_coords_x = []
            route_i_coords_y = []

            for i in range(routestarttime):
                #-1 signifies that drone was not in the air or waiting for pickup at all. drone was with the truck.
                route_i_coords_x.append(-1)
                route_i_coords_y.append(-1)

            xmove, ymove = get_segments_points(route_i_waypoints_x, route_i_waypoints_y, routemovetime)
            route_i_coords_x = route_i_coords_x + list(xmove)
            route_i_coords_y = route_i_coords_y + list(ymove)

            #handle case where drone has to wait to be picked up
            routewaittime = 0
            if routemovetime < deploypickuptime:
                routewaittime = deploypickuptime - routemovetime
                for i in range(routewaittime):
                    route_i_coords_x.append(endpointcoordx)
                    route_i_coords_y.append(endpointcoordy)
            
            route_wait_times.append(routestarttime + routemovetime + routewaittime - 1)
            routefinishedtime = routestarttime + routemovetime + routewaittime

            #time between route being done and entire process being completed
            routedonetime = objective - routefinishedtime

            for i in range(routedonetime):
                #drone is with the truck again.
                route_i_coords_x.append(-1)
                route_i_coords_y.append(-1)

            routes_coords.append(route_i_coords_x)
            routes_coords.append(route_i_coords_y)

        drones_combined_coords_x = []
        drones_combined_coords_y = []

        for i in range(objective):
            #these are lists of the coords of active drones at frame i
            active_drone_coords_x = []
            active_drone_coords_y = []
            for j in range(1,len(route_times)+1):
                #each route has its x coordinates list and y coordinates list in routes_coords.
                #route j has its x coords as list at index 2*(j-1)
                #route j has its y coords as list at index 2*(j-1) + 1
                #ex: route 3 has x coords at index 4, y coords at index 5
                route_x = routes_coords[2*(j-1)]
                route_y = routes_coords[2*(j-1)+1]

                if(route_x[i] != -1):
                    active_drone_coords_x.append(route_x[i])

                if(route_y[i] != -1):
                    active_drone_coords_y.append(route_y[i])
                
            drones_combined_coords_x.append(active_drone_coords_x)
            drones_combined_coords_y.append(active_drone_coords_y)
        
        active_drones = []
        for li in drones_combined_coords_x:
            active_drones.append(len(li))
        
        dronesused = max(active_drones)

        #0 means waiting, 1 means moving
        truck_move_wait = []
        for i in range(len(truck_coords[0])-1):
            if(truck_coords[0][i] == truck_coords[0][i+1] and truck_coords[1][i] == truck_coords[1][i+1]):
                truck_move_wait.append(0)
            else:
                truck_move_wait.append(1)
        
        truck_move_wait.append(1)

        truck_move = []
        truck_wait = []
        wait = False
        start = -1
        end = -1
        for i in range(len(truck_move_wait)):
            if(truck_move_wait[i] == 0):
                wait = True
            
            if(truck_move_wait[i] == 1):
                wait = False
            
            if(i == len(truck_move_wait) - 1):
                end = len(truck_move_wait) - 1
                if(wait):
                    truck_wait.append((start, end))
                else:
                    truck_move.append((start, end))
            else:
                if(i == 0):
                    start = i
                
                if(truck_move_wait[i] != truck_move_wait[i+1]):
                    if(start == -1):
                        start = i
                    else:
                        end = i
                        if(wait):
                            truck_wait.append((start, end))
                        else:
                            truck_move.append((start, end))
                        start = end
                        end = -1

        route_times = list(zip(list(range(1,len(route_start_times)+1)), route_start_times, route_finish_times, route_wait_times))
        route_times = sorted(route_times, key=lambda x: x[1])
        route_order = []
        for route in route_times:
            route_order.append(route[0])
        drones_times = {}
        route_drones = {}

        for i in range(1, numdrones+1):
            drones_times[i] = []

        for i in range(1, len(route_times)+1):
            route_drones[i] = 0
            
        dronelist = list(range(1,dronesused+1))
        drone_endtimes = []
        for route in route_times:
            for det in drone_endtimes:
                if(det[1] <= route[1]):
                    dronelist.append(det[0])
                    drone_endtimes.remove(det)
            
            drone_assigned = dronelist.pop(0)
            drone_endtimes.append((drone_assigned, route[3]))
            drones_times[drone_assigned].append(route)
            route_drones[route[0]] = drone_assigned
            
        plt.style.use('ieee_ras_conf.mplstyle')
        if not find_executable('latex'):
            plt.rc('text', usetex=False)
            print('Latex not installed; not using Latex text.')
        
        fig_w = 3.4
        fig_h = 5.6
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(fig_w, fig_h), gridspec_kw={'height_ratios': [7, 3]})


        xLine, yLine = line.xy
        ax.plot(xLine,yLine,'k', lw=2, ms=0, zorder=0)

        cmap = plt.get_cmap('gist_rainbow')
        colors = [cmap(i) for i in np.linspace(0,1,numdrones)]
        for i in range(len(route_times)):
            route_i_xpts = all_route_points[2*i]
            route_i_ypts = all_route_points[(2*i)+1]
            maxcoords = [-1, -1, -1, -1]
            maxdist = -1
            for j in range(len(route_i_xpts)-1):
                x1 = route_i_xpts[j]
                y1 = route_i_ypts[j]
                x2 = route_i_xpts[j+1]
                y2 = route_i_ypts[j+1]
                distance = np.hypot(abs(x1-x2), abs(y1-y2))
                if(distance > maxdist):
                    maxdist = distance
                    maxcoords = [x1, x2, y1, y2]

            route_i, = ax.plot(route_i_xpts,route_i_ypts, color=colors[route_drones[i+1]-1], lw=1.5, markersize=0, zorder = 0)
            ax.scatter(route_i_xpts[1:-1],route_i_ypts[1:-1], color='k', s=12, zorder = 0)
            #ax.plot(maxcoords[0],maxcoords[1],'wo',markersize=10,zorder=1)
            recislope = -1*(maxcoords[1]-maxcoords[0])/(maxcoords[3]-maxcoords[2])
            midpoint = [(maxcoords[0]+maxcoords[1])/2, (maxcoords[2]+maxcoords[3])/2]
            d = 30
            xdiff = d/(np.sqrt(1+(recislope**2)))
            ydiff = recislope*xdiff
            #ax.plot(midpoint[0]+xdiff,midpoint[1]+ydiff,'wo',markersize=10,zorder=1)
            if(len(route_i_xpts) < 4):
                slope = (route_i_ypts[0]-route_i_ypts[1])/(route_i_xpts[0]-route_i_xpts[1])
                xdiff = abs(30/(np.sqrt(1+(slope**2))))
                ydiff = abs(slope*xdiff)
                if(route_i_xpts[0] > route_i_xpts[1]):
                    xdiff = -1*xdiff
                if(route_i_ypts[0] > route_i_ypts[1]):
                    ydiff = -1*ydiff
                
                ax.text(route_i_xpts[1]+xdiff, route_i_ypts[1]+ydiff, str(route_order.index(i+1)+1), ha='center', va='center', color='k', zorder=2)
            else:
                ax.text(midpoint[0]+xdiff, midpoint[1]+ydiff, str(route_order.index(i+1)+1), ha='center', va='center', color='k', zorder=2)
                startpoint = [(maxcoords[0] + midpoint[0])/2, (maxcoords[2] + midpoint[1])/2]
                endpoint = [(maxcoords[1] + midpoint[0])/2, (maxcoords[3] + midpoint[1])/2]
                dx = endpoint[0] - startpoint[0]
                dy = endpoint[1] - startpoint[1]
                if(flipped[i]):
                    ax.arrow(endpoint[0], endpoint[1], -1*dx, -1*dy, color=colors[route_drones[i+1]-1], width=4, head_width=60, head_length=40, length_includes_head=True, zorder=1)
                else:
                    ax.arrow(startpoint[0], startpoint[1], dx, dy, color=colors[route_drones[i+1]-1], width=4, head_width=60, head_length=40, length_includes_head=True, zorder=1)
        
        ax.scatter([depot.x + 6], [depot.y + 4], marker='*', c='y', s=60)
        ax.text(depot.x + 60, depot.y-30, "Start\n\& End", ha='center', va='center', color='k', zorder=2)
            
        ax.set_xlim(0,data['bound_length'])
        ax.set_ylim(0, data['bound_length'])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Route Plan Map')
        
        custom_lines = [Line2D([0], [0], lw=1, color='k')]
        names = ['Road']
        letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
        for i in range(numdrones):
            custom_lines.append(Line2D([0], [0], lw=1, color=colors[i]))
            names.append("Drone " + str(letters[i]))
        
        ax.legend((custom_lines), names, loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, ncol=3)

        """ for i,evt in enumerate(x):
            ax2.barh(i,width=evt[1]-evt[0],left=evt[0]) """

        ax2.set_yticks(range(numdrones+1))
        labels = copy.copy(names)
        labels[0] = "Truck"
        ax2.invert_yaxis()
        ax2.set_yticklabels(labels)
        ax2.set_xlim(0, objective-1)
        ax2.set_xlabel('Time')
        ax2.set_title('Truck and Multidrone Schedule')
        
        truck_move_diffs = []
        truck_move_starts = []
        for interval in truck_move:
            truck_move_diffs.append(interval[1]-interval[0])
            truck_move_starts.append(interval[0])
        
        ax2.barh(y=0, width=truck_move_diffs, height=0.8, left=truck_move_starts, color='k')

        drone_move_diffs = {}
        drone_wait_diffs = {}
        drone_move_starts = {}
        drone_wait_starts = {}
        drone_route_nums = {}

        for i in drones_times.keys():
            drone_move_diffs[i] = []
            drone_wait_diffs[i] = []
            drone_move_starts[i] = []
            drone_wait_starts[i] = []
            drone_route_nums[i] = []

        for i in drones_times.keys():
            for info in drones_times[i]:
                drone_move_diffs[i].append(info[2]-info[1])
                drone_wait_diffs[i].append(info[3]-info[2])
                drone_move_starts[i].append(info[1])
                drone_wait_starts[i].append(info[2])
                drone_route_nums[i].append(info[0])
        
        """ for i in drone_move_diffs.keys():
            ax2.barh(y=i, width=drone_move_diffs[i], height=0.9, left=drone_move_starts[i], color=colors[i-1])
            ax2.barh(y=i, width=drone_wait_diffs[i], height=0.9, left=drone_wait_starts[i], color=(0.8, 0.8, 0.8), zorder=0)
            for j in range(len(drone_move_diffs[i])):
                ax2.plot(drone_move_starts[i][j] + (drone_move_diffs[i][j]/2),i,'wo',markersize=8,zorder=1)
                ax2.text(drone_move_starts[i][j] + (drone_move_diffs[i][j]/2), i, str(route_order.index(drone_route_nums[i][j])+1), ha='center', va='center', color='k', fontsize=8, zorder=2) """
            
        for i in range(1,numdrones+1):
            if(i <= dronesused):
                ax2.barh(y=i, width=drone_move_diffs[i], height=0.9, left=drone_move_starts[i], color=colors[i-1])
                ax2.barh(y=i, width=drone_wait_diffs[i], height=0.9, left=drone_wait_starts[i], color=(0.8, 0.8, 0.8), zorder=0)
                for j in range(len(drone_move_diffs[i])):
                    ax2.plot(drone_move_starts[i][j] + (drone_move_diffs[i][j]/2),i,'wo',markersize=8,zorder=1)
                    ax2.text(drone_move_starts[i][j] + (drone_move_diffs[i][j]/2), i, str(route_order.index(drone_route_nums[i][j])+1), ha='center', va='center', color='k', fontsize=8, zorder=2)
            else:
                ax2.barh(y=i, width=[10, 20], height=0.9, left=[10, 60], color='w')
                ax2.text(objective/2, i, "Not Deployed", ha='center', va='center', color='k', fontsize=8, zorder=2)
            
        ax.set_title("(a)")
        ax2.set_title("(b)")

        plt.tight_layout(pad=0.1, h_pad=1.2)

        string = '/Users/saaketh/Desktop/publication_plots/pub_plots_python/ieee_ras_conf/figures/' + filename + '.pdf'
        fig.savefig(string)

        plt.show()
        plt.clf()
        plt.close()
        """ print("length of truck coords x:", len(truck_coords[0]))
        print("length of truck coords y:", len(truck_coords[1]))
        print("length of drone coords x:", len(drones_combined_coords_x))
        print("length of drone coords y:", len(drones_combined_coords_y)) """

        """ #now handle routes changing colors as specified times in route_finish_times
        route_finished_indicators = []
        for i in range(len(route_finish_times)):
            #0 means route not done. 1 means route done.
            route_i_indicator = []
            route_i_finish_time = route_finish_times[i]
            unfinished_list = [0]*(route_i_finish_time-1)
            finished_list = [1]*(objective-route_i_finish_time+1)
            route_i_indicator = route_i_indicator + unfinished_list + finished_list
            route_finished_indicators.append(route_i_indicator)

        #plot road
        xLine, yLine = line.xy
        plt.plot(xLine,yLine,'r', lw=2, zorder=0)

        ims = []
        for i in range(min(len(truck_coords[0]), len(drones_combined_coords_x), objective)):
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
        for i in range(1,len(visit_order)):
            dex = visit_order[i]
            ax.plot(xc[dex],yc[dex],'wo',markersize=10,zorder=1)
            ax.text(xc[dex],yc[dex],i, ha="center", va="center", fontsize = 10, zorder=2)

        ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,repeat_delay=1000)

        writer = animation.FFMpegWriter(fps=20, metadata=dict(artist='Me'), bitrate=1800)
        complete_filename = filename + ".mp4"
        ani.save(complete_filename, writer=writer)
            
        #plt.legend(handles=legendlines, loc='best', prop={'size': 5})
        #imagename = "solution_singledepotroad.png"
        plt.show()
        plt.clf()
        plt.close() """
    else:
        print("\n not graphing this one lol.")
    

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
    routepointcoords = truckdata['routepointcoords']
    partroutedistances = truckdata['partroutedistances']

    print("\nROUTEDISTANCES LENGTH:", len(routedistances))
    print("\nUSEDNODECOORDS LENGTH:", len(usednodecoords))
    print("\nROUTEPOINTCOORDSX LENGTH:", len(routepointcoords[0]))
    print("\nROUTEPOINTCOORDSY LENGTH:", len(routepointcoords[1]))
    
    depart_times, wait_times, deployments, route_times, objective, xc, yc, visit_order, exact_depart_times, exact_wait_times, exact_route_times, exact_objective = truck_gurobi.tsp_truck(routedistances, usednodecoords, numdrones, dronevel, truckvel, data, manager, routing, solution, True)

    usednodecoords = [0] + usednodecoords

    combined_route_times = sum(route_times.values())
    print("\n--------------------")
    print("\nOPTIMIZED DRONE DEPLOYMENT TIME:", objective)
    print("LOWER BOUND TIME:", combined_route_times/float(numdrones))

    graph_solution(data, manager, routing, solution, depart_times, wait_times, deployments, route_times, objective, xc, yc, visit_order, usednodecoords, "postprocessing_control", True, False)

    usednodecoords.pop(0)

    post_depart_times, post_wait_times, post_deployments, post_route_times, post_objective, post_xc, post_yc, post_visit_order = postprocessing.postprocess(routedistances, dronevel, truckvel, usednodecoords, routepointcoords, visit_order, deployments, data, manager, routing, solution, exact_objective, partroutedistances, dronerange, exact_depart_times, exact_wait_times, exact_route_times, xc, yc)

    print("\n--------------------")
    print("\nPOSTPROCESSING DRONE DEPLOYMENT TIME:", post_objective)

    graph_solution(data, manager, routing, solution, post_depart_times, post_wait_times, post_deployments, post_route_times, post_objective, post_xc, post_yc, post_visit_order, post_xc, "postprocessing_test", True, True)


if __name__ == '__main__':
    main()