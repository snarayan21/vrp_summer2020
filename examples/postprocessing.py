from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
import math
import gurobipy
from heapq import nlargest
from gurobipy import Model, GRB, quicksum, abs_, and_
import shapely.geometry as geom
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from ortools.linear_solver import pywraplp
from ortools.sat.python import cp_model
import complete_solver_gurobi

def postprocess(dRs, dronevel, truckvel, usednodecoords, routepointcoords, visit_order, deployments, data, manager, routing, solution, objective, partroutedistances, dronerange, exact_depart_times, exact_wait_times, exact_route_times, xc, yc):

    print("\nPostprocessing to optimize route endpoints...\n")

    n = len(usednodecoords)
    vT = float(truckvel)
    vD = float(dronevel)

    bound_length = 1000

    V = [i for i in range(1, n+1)]
    eV = [i for i in V if i % 2 == 0]
    oV = [i for i in V if i % 2 == 1]
    N = [0] + V
    R = [i for i in range(1,(n//2) + 1)]

    tRs = (np.asarray(dRs)/vD).tolist()
    tR = {i: tRs[i-1] for i in R}

    ptRs = (np.asarray(partroutedistances)/vD).tolist()
    ptR = {i: ptRs[i-1] for i in R}

    slacks = []
    for i in range(len(partroutedistances)):
        slacks.append(dronerange - partroutedistances[i])
    
    sR = {i: slacks[i-1] for i in R}

    A = []
    curr = 0
    for i in range(1,len(visit_order)):
        A.append((curr, visit_order[i]))
        curr = visit_order[i]
    
    A.append((curr, 0))

    z = {(i, j): 1 for i,j in A}

    g = deployments

    first = visit_order[1]
    last = visit_order[-1]

    #x and y coordinates of first and last route waypoints
    xp = routepointcoords[0]
    yp = routepointcoords[1]

    #this is for a straight line road through diagonal y = x
    dx = [1/np.sqrt(2),1/np.sqrt(2)]

    mdl = Model('Postprocessing Step')

    c = mdl.addVars(A, ub = np.sqrt(2)*bound_length, vtype=GRB.CONTINUOUS, name = "c")
    d = mdl.addVars(N, ub = objective, vtype=GRB.CONTINUOUS, name = "d")
    w = mdl.addVars(V, ub = max(ptRs) + ((2*np.sqrt(2)*bound_length)/vD), vtype=GRB.CONTINUOUS, name = "w")
    s = mdl.addVars(V, ub = np.sqrt(2)*bound_length, vtype=GRB.CONTINUOUS, name = "s")
    x = mdl.addVars(N, ub = bound_length, vtype=GRB.CONTINUOUS, name="x")
    y = mdl.addVars(N, ub = bound_length, vtype=GRB.CONTINUOUS, name="y")
    rds = mdl.addVars(V, ub = 2*(bound_length**2), vtype=GRB.CONTINUOUS, name="xd")
    rd = mdl.addVars(V, ub = np.sqrt(2)*bound_length, vtype=GRB.CONTINUOUS, name="xd")
    l = mdl.addVars(R, ub = (max(ptRs)*vD) + 2*np.sqrt(2)*bound_length, vtype=GRB.CONTINUOUS, name = "l")

    print("exact depart times: ", exact_depart_times)
    print("exact wait times: ", exact_wait_times)
    print("exact route times: ", exact_route_times)
    print("xc: ", *xc)

    for i, j in A:
        c[i,j].start = np.sqrt(((xc[i] - xc[j])*(xc[i] - xc[j])) + ((yc[i] - yc[j])*(yc[i] - yc[j])))
    
    for i in N:
        d[i].start = exact_depart_times[i]
        x[i].start = xc[i]
        y[i].start = yc[i]
    
    for i in V:
        w[i].start = exact_wait_times[i]
        s[i].start = np.sqrt(2)*xc[i]
        rds[i].start = ((xc[i] - xp[i-1])*(xc[i] - xp[i-1])) + ((yc[i] - yp[i-1])*(yc[i] - yp[i-1]))
        rd[i].start = np.sqrt(((xc[i] - xp[i-1])*(xc[i] - xp[i-1])) + ((yc[i] - yp[i-1])*(yc[i] - yp[i-1])))
    
    for i in eV:
        rdi = np.sqrt(((xc[i] - xp[i-1])*(xc[i] - xp[i-1])) + ((yc[i] - yp[i-1])*(yc[i] - yp[i-1])))
        rdim1 = np.sqrt(((xc[i-1] - xp[i-2])*(xc[i-1] - xp[i-2])) + ((yc[i-1] - yp[i-2])*(yc[i-1] - yp[i-2])))
        l[i/2].start = rdi + rdim1 + (ptR[i/2]*vD)
            

    #objective is sum of truck move time plus truck wait time
    mdl.setObjective(
        quicksum(c[i,j]/vT for i, j in A) +
        quicksum(w[i] for i in V)
    )

    #departure time of depot is 0
    d[0].ub = 0

    #departure time of next node j must be departure time of node i plus truck time plus waiting time
    mdl.addConstrs((d[j] == d[i] + (c[i ,j]/vT) + w[j] for i, j in A if j != 0), name="first")

    #Big M constraint for enforcing that difference in departure times of route endpoints has to be
    #greater than the drone time for that route, M = init soln objective + maximum possible new route length
    mdl.addConstrs((d[j] - d[j-1] - (l[j/2]/vD) >= -1*(objective + max(ptRs) + (2*np.sqrt(2)*bound_length)/vD)*(1-g[j]) for j in eV), name="second")
    #just as above, for odd numbered nodes.
    mdl.addConstrs((d[j] - d[j+1] - (l[(j+1)/2]/vD) >= -1*(objective + max(ptRs) + (2*np.sqrt(2)*bound_length)/vD)*(1-g[j]) for j in oV), name="third")

    #set x coord and y coord constraints for points on road
    mdl.addConstrs((x[i] == s[i]*dx[0] for i in V), name="fourth")
    mdl.addConstrs((y[i] == s[i]*dx[1] for i in V), name="fifth")
    #x and y coordinate of depot are 0 for straight line road y = x
    x[0].ub = 0
    y[0].ub = 0

    #the cost c_ij equals the distance between node i and node j (euclidean distance in straight line case)
    mdl.addConstrs((c[i, j]*c[i, j] == ((x[i] - x[j])*(x[i] - x[j])) + ((y[i] - y[j])*(y[i] - y[j])) for i, j in A), name="sixth")

    #define helper variables for constraints on start and end segments of routes
    mdl.addConstrs((rds[i] == ((x[i] - xp[i-1])*(x[i] - xp[i-1])) + ((y[i] - yp[i-1])*(y[i] - yp[i-1])) for i in V), name="seventh")

    #rd equals square root of rds, where rd is the edge distance between road point of route and corresponding waypoint
    for i in V:
        mdl.addGenConstrPow(rds[i], rd[i], 0.5, "edgedist")
    
    #sum of the two ending edge lengths for any route must be less than the slack distance left on that route.
    mdl.addConstrs((rd[i] + rd[i-1] <= sR[i/2] for i in eV), name="eighth")

    #full length of the route, l_i, equals the waypoint-only distance plus the two ending edge lengths
    mdl.addConstrs((l[i/2] == rd[i] + rd[i-1] + (ptR[i/2]*vD) for i in eV), name="ninth")

    mdl.params.FeasibilityTol = 0.005
    mdl.params.NonConvex = 2
    mdl.params.MIPGap = 0.0001
    mdl.params.Method = 5
    mdl.params.TimeLimit = 600  # seconds
    #mdl.params.ImproveStartTime = 30 #seconds
    mdl.params.ImproveStartNodes = 1 #nodes
    mdl.params.MIPFocus = 1
    mdl.params.Cuts = 2
    mdl.optimize()

    new_usednodecoords = [0] + [(v.X)/np.sqrt(2) for k,v in s.items()]
    depart_times = { k : round(v.X) for k,v in d.items() }
    wait_times = { k : round(v.X) for k,v in w.items() }
    rounded_route_times = {k: round((v.X)/vD) for k,v in l.items()}
    new_objective = round(mdl.objVal)

    ds = []
    ws = []
    xs = []

    for v in mdl.getVars():
        if("d" in v.varName):
            ds.append((v.varName, v.x))
        elif ("w" in v.varName):
            ws.append((v.varName, v.x))
        elif ("x" in v.varName):
            xs.append((v.varName, v.x))   

    print("\n")

    for i in range(len(visit_order)):
        node = visit_order[i]
        print("d[", str(i), "]", ": ", ds[node][1])
    
    print("\n")
    
    for i in range(1,len(visit_order)):
        node = visit_order[i]
        print("w[", str(i), "]", ": ", ws[node-1][1])

    print("\n")
    
    for i in range(len(visit_order)):
        node = visit_order[i]
        print("x[", str(i), "]", ": ", xs[node][1])

    return depart_times, wait_times, deployments, rounded_route_times, new_objective, new_usednodecoords, new_usednodecoords, visit_order

    












