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

def main():

    if len(sys.argv) != 5:
        print('Should be called as follows: python tsp_truck_gurobi.py [number of nodes] [number of drones] [truck velocity] [drone velocity]')
        return
        
    n = int(sys.argv[1])
    D = int(sys.argv[2])
    vT = float(sys.argv[3])
    vD = float(sys.argv[4])
    
    if(n % 2 != 0):
        print("Number of nodes must be even.")
        return
        
    bound_length = 1000
    
    dRs = np.random.uniform(low = np.sqrt(2)*0.5*bound_length, high = np.sqrt(2)*1.5*bound_length, size = n//2)
    tRs = (dRs/vD).tolist()
    
    xc = np.random.randint(low=0,high=bound_length,size=n+1)
    #xc = np.linspace(0, np.sqrt(2)*bound_length, n+2, dtype = 'int32').tolist()
    #xc.pop(-1)
    yc = xc
    
    xc[0] = 0
    yc[0] = 0
    
    #plt.plot(xc[0], yc[0], c='r', marker='s')
    #plt.scatter(xc[1:], yc[1:], c='b')
    
    #sets needed
    V = [i for i in range(1, n+1)]
    eV = [i for i in V if i % 2 == 0]
    oV = [i for i in V if i % 2 == 1]
    N = [0] + V
    A = [(i, j) for i in N for j in N if i != j]
    R = [i for i in range(1,(n//2) + 1)]
    c = {(i, j): np.hypot(xc[i]-xc[j], yc[i]-yc[j])/vT for i, j in A}
    #add a very small cost to every 0 cost edge so the solver never has identical departure times for two given nodes
    for i,j in A:
        if c[i,j] == 0:
            c[i,j] = 0.01
    tR = {i: tRs[i-1] for i in R}
    
    #computed values for better variable bounds
    M = (np.sqrt(2)*bound_length*2)
    costs = list(c.values())
    ckmax = sum(nlargest(n+1, costs))
    cmax = max(costs)
    maxdiff = 0
    for i in N:
        for j in N:
            diff = costs[i] - costs[j]
            if diff > maxdiff:
                maxdiff = diff
    
    mdl = Model('Truck Problem')

    #variables needed, most are intermediate helpers lol
    x = mdl.addVars(A, vtype=GRB.BINARY, name = "x")
    a = mdl.addVars(N, ub = ckmax + sum(tRs), vtype=GRB.CONTINUOUS, name = "a")
    w = mdl.addVars(V, ub = max(maxdiff,max(tRs)), vtype=GRB.CONTINUOUS, name = "w")
    h = mdl.addVars(V, vtype=GRB.BINARY, name = "h")
    b = mdl.addVars(R, lb = -1*(ckmax + sum(tRs)), ub = ckmax + sum(tRs), vtype=GRB.CONTINUOUS, name = "b")
    s = mdl.addVars(R, ub = ckmax + sum(tRs), vtype=GRB.CONTINUOUS, name = "s")
    z = mdl.addVars(R, ub = ckmax + sum(tRs), vtype=GRB.CONTINUOUS, name = "z")
    #d = mdl.addVars(R, ub = max(maxdiff,max(tRs)), vtype=GRB.CONTINUOUS, name = "d")
    e = mdl.addVars(R, vtype=GRB.BINARY, name = "e")
    f = mdl.addVars(V, lb = -1, ub = 1, vtype=GRB.INTEGER, name = "f")
    g = mdl.addVars(V, vtype=GRB.BINARY, name = "g")
    y = mdl.addVars(N, lb = 0, ub = 1*D, vtype=GRB.INTEGER, name = "y")
    #l = mdl.addVar(lb = 0, ub = ckmax + sum(tRs), vtype=GRB.CONTINUOUS, name = "l")    
    mdl.modelSense = GRB.MINIMIZE

    #objective is sum of truck time costs plus truck waiting times
    #mdl.setObjective(l)
    mdl.setObjective(
        quicksum(x[i, j]*c[i, j] for i, j in A) + 
        quicksum(w[i] for i in V))
    
    #sum of x[i,j]'s  for all fixed values of i is 1 (exactly one outgoing edge from every node)
    mdl.addConstrs(quicksum(x[i, j] for j in N if j != i) == 1 for i in N)
    
    #sum of x[i,j]'s  for all fixed values of j is 1 (exactly one incoming edge from every node)
    mdl.addConstrs(quicksum(x[i, j] for i in N if i != j) == 1 for j in N)

    #set l to be the cost of the entire tour (last d[i] + cost of node i to node 0).
    #mdl.addConstrs((x[i, 0] == 1) >> (l == a[i]+c[i,0]) for i in V)
    #mdl.addConstr(l == max_(a[i] for i in N))

    #set arrival time at node 0 to be 0
    a[0].ub = 0.0
    
    #if edge (i,j) is in solution, then a[j] = a[i] + c[i,j] + w[j] (waiting time at node j)
    #also functions as Miller-Tucker-Zemlin constraint since it prevents subcycles
    mdl.addConstrs((x[i, j] == 1) >> (a[i]+c[i,j]+w[j] == a[j]) for i, j in A if j != 0)
    
    #b[i] = a[2i] - a[2i-1] (truck travel time between route endpoints), is a helper
    mdl.addConstrs(b[i] == a[2*i] - a[(2*i)-1] for i in R)
    
    #s[i] = |b[i]|, is another helper variable
    for i in R:
        mdl.addGenConstrAbs(s[i],b[i])

    #z[i] = s[i] - waiting times at both endpoints. 
    #One of waiting times is guaranteed to be 0, other one can be 0 or positive.
    #this gives the true truck-only travel time between route endpoints
    mdl.addConstrs(z[i] == s[i] - w[2*i] - w[(2*i)-1] for i in R)
    
    #if z[i] <= tR[i] then e[i] = 1
    #mdl.addConstrs(tR[i] - z[i] <= (M/vD)*e[i] for i in R)
    #if e[i] = 1 then z[i] <= tR[i], completes the iff statement
    #means e[i] = 1 when truck travel time is less than drone travel time
    #means e[i] = 0 when truck travel time is greater than drone travel time
    mdl.addConstrs((e[i] == 1) >> (z[i] <= tR[i]) for i in R)
    mdl.addConstrs((e[i] == 0) >> (z[i] >= tR[i]) for i in R)
    
    #if a[j] < a[j-1] then g[j] = 0, otherwise g[j] = 1 for even j
    #g[j] = 0 means starting new route, otherwise completing route
    #mdl.addConstrs(a[j-1] - a[j] <= (M/vT)*g[j] for j in eV)
    mdl.addConstrs((g[j] == 0) >> (a[j] <= a[j-1]) for j in eV)
    mdl.addConstrs((g[j] == 1) >> (a[j] >= a[j-1]) for j in eV)
    
    #if a[j] > a[j-1] then g[j] = 0, otherwise g[j] = 1 for odd j
    #g[j] = 0 means starting new route, otherwise completing route
    #mdl.addConstrs(a[j+1] - a[j] <= (M/vT)*g[j] for j in oV)
    mdl.addConstrs((g[j] == 0) >> (a[j] <= a[j+1]) for j in oV)
    mdl.addConstrs((g[j] == 1) >> (a[j] >= a[j+1]) for j in oV)

    #if g[j] = 0 (ending pending route) and 
    #e[j/2] = 1 (even j), or e[(j+1)/2] = 1 (odd j),
    #which means (truck time < drone time),
    #then w[j] = tR[j/2] - z[j/2] for even j
    #then w[j] = tR[(j+1)/2] - z[(j+1)/2 + 1] for odd j
    #otherwise w[j] = 0
    
    #h[i] is 1 when both are true, false otherwise
    mdl.addConstrs(h[i] == and_(g[i], e[i/2]) for i in eV)
    mdl.addConstrs(h[i] == and_(g[i], e[(i+1)/2]) for i in oV)

    #if h[i] = 0, then w[i] = 0
    #else, when h[i] = 1, then w[i] = drone time - truck time
    mdl.addConstrs((h[i] == 0) >> (w[i] == 0) for i in eV)
    mdl.addConstrs((h[i] == 0) >> (w[i] == 0) for i in oV)
    mdl.addConstrs((h[i] == 1) >> (w[i] == tR[i/2] - z[i/2]) for i in eV)
    mdl.addConstrs((h[i] == 1) >> (w[i] == tR[(i+1)/2] - z[(i+1)/2]) for i in oV)
    
    #if g[j] = 0, then f[j] = -1 since we are starting a new route, so capacity decreases
    mdl.addConstrs((g[j] == 0) >> (f[j] == -1) for j in V)
    #if g[j] = 1, then f[j] = 1 since we are completing old route, so capacity increases
    mdl.addConstrs((g[j] == 1) >> (f[j] == 1) for j in V)
    
    #if edge (i,j) is in solution, then capacity y[j] = y[i] + f[j]
    mdl.addConstrs((x[i, j] == 1) >> (y[i]+f[j] == y[j]) for i, j in A if j != 0)
    
    #0 <= y[j] <= D (# of drones) for all j
    mdl.addConstrs(y[j] >= 0 for j in V)
    mdl.addConstrs(y[j] <= D for j in V)
    
    #set drone availability at node 0 (depot) to be D
    y[0].lb = D
    y[0].ub = D
    
    mdl.params.MIPGap = 0.05
    mdl.params.Method = 5
    mdl.params.TimeLimit = 60  # seconds
    #mdl.params.ImproveStartTime = 1 #seconds
    #mdl.params.ImproveStartNodes = 1 #nodes
    #mdl.params.MIPFocus = 1
    mdl.optimize()
    
    print("\nVARIABLE MEANINGS:")
    print("x[i,j] is presence of edge in solution")
    print("a[i] is DEPARTURE time at node i")
    print("w[i] is waiting time at node i")
    print("h[i] is 1 if node i is collection node and if truck time < drone time")
    print("b[i] is signed truck travel time between endpoints of route i")
    print("s[i] is absolute truck travel time, including waiting time after collection, between endpoints of route i")
    print("z[i] is absolute truck travel time between endpoints of route i")
    print("e[i] is 1 if truck travel time z[i] on route i is less than drone travel time on route i, 0 otherwise")
    print("d[i] equals z[i] if e[i] is 1, tR[i] otherwise")
    print("f[i] equals -1 if starting new route at node i, 1 if completing old route at node i")
    print("g[i] equals 0 if truck arrives at node i before other endpoint of route, 1 otherwise.")
    print("y[i] is the number of available vehicles after node i")

    print("--------\n")
    print("coords:", *xc, "\n")
    print("drone route times:", tR, "\n")
    #print(c)

    for i in range(0, n+1):
        for j in range(0, n+1):
            if(j != n):
                if(i != j):
                    print(c[i,j], end = ' ')
                else:
                    print(0, end = ' ')
            else:
                if(i != j):
                    print(c[i,j])
                else:
                    print(0)
            
    
    for v in mdl.getVars():
        if(not("x" in v.varName)):
            print('%s %g' % (v.varName, v.x))

    active_arcs = [a for a in A if x[a].x > 0.98]
    
    active_arcs = sorted(active_arcs, key=lambda x: x[0])
    
    print(active_arcs)
    
    visit_order = []
    route = "0 -> "
    i = 0
    curr = 0
    while i < n:
        curr = active_arcs[curr][1]
        if(curr != 0):
            visit_order.append(curr)
        route = route + str(curr) + " -> "
        i += 1
    route = route + "0"
    
    print(route)

    for i in R:
        e1 = xc[(2*i)-1]
        e2 = xc[(2*i)]
        c1x = e1 + (30*i)
        c1y = e1 - (30*i)
        c2x = e2 + (30*i)
        c2y = e2 - (30*i)
        plt.plot([e1, c1x], [e1, c1y], c='b', zorder=0)
        plt.plot([c1x, c2x], [c1y, c2y], c='b', zorder=0)
        plt.plot([c2x, e2], [c2y, e2], c='b', zorder=0)
        
    for i, j in active_arcs:
        plt.plot([xc[i], xc[j]], [yc[i], yc[j]], c='y', zorder=0)
        plt.plot(xc[0], yc[0], c='r', marker='s')
        plt.scatter(xc[1:], yc[1:], c = 'w', linewidth = 7)
        
    for i in range(len(visit_order)):
        dex = visit_order[i]
        plt.text(xc[dex],yc[dex],i+1, ha="center", va="center")
    
    plt.show()

"""  
def tsp_truck(routedistances, usednodecoords, numdrones, dronevel, truckvel):

    print("Finding optimal truck routing...")
        
    n = len(usednodecoords)
    D = int(numdrones)
    vT = float(truckvel)
    vD = float(dronevel)
    
    if(n % 2 != 0):
        print("Number of nodes must be even.")
        return
        
    bound_length = 1000
    
    dRs = np.asarray(routedistances)
    tRs = (dRs/vD).tolist()
    
    xc = [0] + usednodecoords
    #xc = np.linspace(0, np.sqrt(2)*bound_length, n+2, dtype = 'int32').tolist()
    #xc.pop(-1)
    yc = xc
    
    #plt.plot(xc[0], yc[0], c='r', marker='s')
    #plt.scatter(xc[1:], yc[1:], c='b')
    
    #sets needed
    V = [i for i in range(1, n+1)]
    eV = [i for i in V if i % 2 == 0]
    oV = [i for i in V if i % 2 == 1]
    N = [0] + V
    A = [(i, j) for i in N for j in N if i != j]
    R = [i for i in range(1,(n//2) + 1)]
    c = {(i, j): np.hypot(xc[i]-xc[j], yc[i]-yc[j])/vT for i, j in A}
    #add a very small cost to every 0 cost edge so the solver never has identical departure times for two given nodes
    for i,j in A:
        if c[i,j] == 0:
            c[i,j] = 0.01

    tR = {i: tRs[i-1] for i in R}
    
    #computed values for better variable bounds
    M = (np.sqrt(2)*bound_length*2)
    costs = list(c.values())
    ckmax = sum(nlargest(n+1, costs))
    cmax = max(costs)
    maxdiff = 0
    for i in N:
        for j in N:
            diff = costs[i] - costs[j]
            if diff > maxdiff:
                maxdiff = diff
    
    mdl = Model('Truck Problem')

    #variables needed, most are intermediate helpers lol
    x = mdl.addVars(A, vtype=GRB.BINARY, name = "x")
    a = mdl.addVars(N, ub = ckmax + sum(tRs), vtype=GRB.CONTINUOUS, name = "a")
    w = mdl.addVars(V, ub = max(maxdiff,max(tRs)), vtype=GRB.CONTINUOUS, name = "w")
    h = mdl.addVars(V, vtype=GRB.BINARY, name = "h")
    b = mdl.addVars(R, lb = -1*(ckmax + sum(tRs)), ub = ckmax + sum(tRs), vtype=GRB.CONTINUOUS, name = "b")
    s = mdl.addVars(R, ub = ckmax + sum(tRs), vtype=GRB.CONTINUOUS, name = "s")
    z = mdl.addVars(R, ub = ckmax + sum(tRs), vtype=GRB.CONTINUOUS, name = "z")
    #d = mdl.addVars(R, ub = max(maxdiff,max(tRs)), vtype=GRB.CONTINUOUS, name = "d")
    e = mdl.addVars(R, vtype=GRB.BINARY, name = "e")
    f = mdl.addVars(V, lb = -1, ub = 1, vtype=GRB.INTEGER, name = "f")
    g = mdl.addVars(V, vtype=GRB.BINARY, name = "g")
    y = mdl.addVars(N, lb = 0, ub = 1*D, vtype=GRB.INTEGER, name = "y")
    #l = mdl.addVar(lb = 0, ub = ckmax + sum(tRs), vtype=GRB.CONTINUOUS, name = "l")    
    mdl.modelSense = GRB.MINIMIZE

    #objective is sum of truck time costs plus truck waiting times
    #mdl.setObjective(l)
    mdl.setObjective(
        quicksum(x[i, j]*c[i, j] for i, j in A) + 
        quicksum(w[i] for i in V))
    
    #sum of x[i,j]'s  for all fixed values of i is 1 (exactly one outgoing edge from every node)
    mdl.addConstrs(quicksum(x[i, j] for j in N if j != i) == 1 for i in N)
    
    #sum of x[i,j]'s  for all fixed values of j is 1 (exactly one incoming edge from every node)
    mdl.addConstrs(quicksum(x[i, j] for i in N if i != j) == 1 for j in N)

    #set l to be the cost of the entire tour (last d[i] + cost of node i to node 0).
    #mdl.addConstrs((x[i, 0] == 1) >> (l == a[i]+c[i,0]) for i in V)
    #mdl.addConstr(l == max_(a[i] for i in N))

    #set arrival time at node 0 to be 0
    a[0].ub = 0.0
    
    #if edge (i,j) is in solution, then a[j] = a[i] + c[i,j] + w[j] (waiting time at node j)
    #also functions as Miller-Tucker-Zemlin constraint since it prevents subcycles
    mdl.addConstrs((x[i, j] == 1) >> (a[i]+c[i,j]+w[j] == a[j]) for i, j in A if j != 0)
    
    #b[i] = a[2i] - a[2i-1] (truck travel time between route endpoints, including waiting time), is a helper
    mdl.addConstrs(b[i] == a[2*i] - a[(2*i)-1] for i in R)
    
    #s[i] = |b[i]|, is another helper variable
    for i in R:
        mdl.addGenConstrAbs(s[i],b[i])

    #z[i] = s[i] - waiting times at both endpoints. 
    #One of waiting times is guaranteed to be 0, other one can be 0 or positive.
    #this gives the true truck-only travel time between route endpoints
    mdl.addConstrs(z[i] == s[i] - w[2*i] - w[(2*i)-1] for i in R)
    
    #if z[i] <= tR[i] then e[i] = 1
    #mdl.addConstrs(tR[i] - z[i] <= (M/vD)*e[i] for i in R)
    #if e[i] = 1 then z[i] <= tR[i], completes the iff statement
    #means e[i] = 1 when truck travel time is less than drone travel time
    #means e[i] = 0 when truck travel time is greater than drone travel time
    mdl.addConstrs((e[i] == 1) >> (z[i] <= tR[i]) for i in R)
    mdl.addConstrs((e[i] == 0) >> (z[i] >= tR[i]) for i in R)
    
    #if a[j] < a[j-1] then g[j] = 0, otherwise g[j] = 1 for even j
    #g[j] = 0 means starting new route, otherwise completing route
    #mdl.addConstrs(a[j-1] - a[j] <= (M/vT)*g[j] for j in eV)
    mdl.addConstrs((g[j] == 0) >> (a[j] <= a[j-1]) for j in eV)
    mdl.addConstrs((g[j] == 1) >> (a[j] >= a[j-1]) for j in eV)
    
    #if a[j] > a[j-1] then g[j] = 0, otherwise g[j] = 1 for odd j
    #g[j] = 0 means starting new route, otherwise completing route
    #mdl.addConstrs(a[j+1] - a[j] <= (M/vT)*g[j] for j in oV)
    mdl.addConstrs((g[j] == 0) >> (a[j] <= a[j+1]) for j in oV)
    mdl.addConstrs((g[j] == 1) >> (a[j] >= a[j+1]) for j in oV)

    #if g[j] = 0 (ending pending route) and 
    #e[j/2] = 1 (even j), or e[(j+1)/2] = 1 (odd j),
    #which means (truck time < drone time),
    #then w[j] = tR[j/2] - z[j/2] for even j
    #then w[j] = tR[(j+1)/2] - z[(j+1)/2 + 1] for odd j
    #otherwise w[j] = 0
    
    #h[i] is 1 when both are true, false otherwise
    mdl.addConstrs(h[i] == and_(g[i], e[i/2]) for i in eV)
    mdl.addConstrs(h[i] == and_(g[i], e[(i+1)/2]) for i in oV)

    #if h[i] = 0, then w[i] = 0
    #else, when h[i] = 1, then w[i] = drone time - truck time
    mdl.addConstrs((h[i] == 0) >> (w[i] == 0) for i in eV)
    mdl.addConstrs((h[i] == 0) >> (w[i] == 0) for i in oV)
    mdl.addConstrs((h[i] == 1) >> (w[i] == tR[i/2] - z[i/2]) for i in eV)
    mdl.addConstrs((h[i] == 1) >> (w[i] == tR[(i+1)/2] - z[(i+1)/2]) for i in oV)
    
    #if g[j] = 0, then f[j] = -1 since we are starting a new route, so capacity decreases
    mdl.addConstrs((g[j] == 0) >> (f[j] == -1) for j in V)
    #if g[j] = 1, then f[j] = 1 since we are completing old route, so capacity increases
    mdl.addConstrs((g[j] == 1) >> (f[j] == 1) for j in V)
    
    #if edge (i,j) is in solution, then capacity y[j] = y[i] + f[j]
    mdl.addConstrs((x[i, j] == 1) >> (y[i]+f[j] == y[j]) for i, j in A if j != 0)
    
    #0 <= y[j] <= D (# of drones) for all j
    mdl.addConstrs(y[j] >= 0 for j in V)
    mdl.addConstrs(y[j] <= D for j in V)
    
    #set drone availability at node 0 (depot) to be D
    y[0].lb = D
    y[0].ub = D
    
    mdl.params.MIPGap = 0.0001
    mdl.params.Method = 5
    mdl.params.TimeLimit = 60  # seconds
    #mdl.params.ImproveStartTime = 40 #seconds
    mdl.params.ImproveStartNodes = 1 #nodes
    mdl.params.MIPFocus = 1
    mdl.optimize()
    
    #print("\nVARIABLE MEANINGS:")
    #print("x[i,j] is presence of edge in solution")
    #print("a[i] is DEPARTURE time at node i")
    #print("w[i] is waiting time at node i")
    #print("h[i] is 1 if node i is collection node and if truck time < drone time")
    #print("b[i] is signed truck travel time, including waiting time after collection, between endpoints of route i")
    #print("s[i] is absolute truck travel time, including waiting time after collection, between endpoints of route i")
    #print("z[i] is absolute truck travel time, excluding waiting time after collection, between endpoints of route i")
    #print("e[i] is 1 if truck travel time z[i] on route i is less than drone travel time on route i, 0 otherwise")
    #print("d[i] equals z[i] if e[i] is 1, tR[i] otherwise")
    #print("f[i] equals -1 if starting new route at node i, 1 if completing old route at node i")
    #print("g[i] equals 0 if truck arrives at node i before other endpoint of route, 1 otherwise.")
    #print("y[i] is the number of available vehicles after node i")

    #print("--------\n")
    #print("coords:", *xc, "\n")
    #print("drone route times:", tR, "\n")
    #print(c)
    
    #total_truck_move_time = 0
    #total_truck_wait_time = 0

    #for v in mdl.getVars():
    #    if("z" in v.varName):
    #        total_truck_move_time += v.x
        
    #    if("w" in v.varName):
    #        total_truck_wait_time += v.x

    active_arcs = [a for a in A if x[a].x > 0.98]

    depart_times = { k : round(v.X) for k,v in a.items() }
    wait_times = { k : round(v.X) for k,v in w.items() }
    deployments = { k : round(v.X) for k,v in f.items() }
    rounded_route_times = {i: round(tR[i]) for i in R}
    objective = round(mdl.objVal)

    print("\nOBJECTIVE:", objective)

    active_arcs = sorted(active_arcs, key=lambda x: x[0])
    
    visit_order = [0]
    route = "0 -> "
    i = 0
    curr = 0
    while i < n:
        curr = active_arcs[curr][1]
        if(curr != 0):
            visit_order.append(curr)
        route = route + str(curr) + " -> "
        i += 1
    route = route + "0"
    
    #print(route)
    
    
    return depart_times, wait_times, deployments, rounded_route_times, objective, xc, yc, visit_order
"""

def simple_init_solution(x, d, w, g, y, A, N, V, tR, R, D, c, xc, yc, n):
    vehicles = {}
    for i in N:
        if(i%2 == 0):
            vehicles[i] = D
            y[i].start = float(D)
        else:
            vehicles[i] = D-1
            y[i].start = float(D-1)
    
    deploys = {}
    for i in V:
        if(i%2 == 0):
            deploys[i] = 1
            g[i].start = 1.0
        else:
            deploys[i] = 0
            g[i].start = 0.0

    for i,j in A:
        x[i,j].start = 0.0
    
    active_arcs = []
    for i in N:
        if(i != n):
            active_arcs.append((i,i+1))
            x[i,i+1].start = 1.0
        else:
            active_arcs.append((i,0))
            x[i,0].start = 1.0
    
    waits = {}
    departs = {}

    for i in V:
        if(i%2 == 0):
            if(tR[i/2] > c[i-1, i]):
                waits[i] = math.ceil(tR[i/2] - c[i-1, i])
                w[i].start = float(tR[i/2] - c[i-1, i])
            else:
                waits[i] = 0
                w[i].start = 0.0
        else:
            waits[i] = 0
            w[i].start = 0.0
    
    d[0].start = 0
    departs[0] = 0
    for i in V:
        departs[i] = math.ceil(c[i-1, i] + waits[i] + departs[i-1])
        d[i].start = float(c[i-1, i] + waits[i] + departs[i-1])

    deployments = { k : (2*round(v)) - 1 for k,v in deploys.items() }
    rounded_route_times = {i: round(tR[i]) for i in R}
    objective = math.ceil(departs[V[-1]] + c[V[-1], 0])

    print("\nINITIAL OBJECTIVE:", objective)

    active_arcs = sorted(active_arcs, key=lambda x: x[0])
    
    visit_order = [0]
    route = "0 -> "
    i = 0
    curr = 0
    while i < n:
        curr = active_arcs[curr][1]
        if(curr != 0):
            visit_order.append(curr)
        route = route + str(curr) + " -> "
        i += 1
    route = route + "0"

    print("\n")
    print(route)   
    print("\n")
    print(tR)

    for i in range(len(visit_order)):
        node = visit_order[i]
        print("d[", str(i), "]", ": ", departs[node])
    
    print("\n")
    
    for i in range(1,len(visit_order)):
        node = visit_order[i]
        print("w[", str(i), "]", ": ", waits[node])
    
    print("\n")

    for i in range(1,len(visit_order)):
        node = visit_order[i]
        print("g[", str(i), "]", ": ", deploys[node])
    
    print("\n")
    
    for i in range(len(visit_order)):
        node = visit_order[i]
        print("y[", str(i), "]", ": ", vehicles[node])
    
    return departs, waits, deployments, rounded_route_times, objective, xc, yc, visit_order

def complex_init_solution(x, d, w, g, y, A, N, V, tR, R, D, c, xc, yc, n, bound_length):

    #list of tasks to do. Index is the node of the task,
    #first entry is cost of task, second entry is 0 if deployment and 1 if pickup, -1 if done
    #third entry is the node for the task (becomes important for when pickup needed)
    #max cost (unachievable) is based on bound length
    tasks = [[(bound_length+1)**2, 0, 0]] * n
    #visited tells us which nodes we have visited
    visited = [False] * n
    #current is the current node we are considering
    current = 0
    #drones is the number of drones we have available
    drones = D
    #time is the current time
    time = 0
    #routetimes keeps track of how much time of a route is left. values change over time
    routetimes = tR.copy()
    #activeroutes keeps track of which routes are active at a certain point.
    activeroutes = []
    #visit_order keeps track of the visit order of nodes
    visit_order = [0]
    rt = "0 -> "

    vehicles = {}
    deploys = {}
    waits = {}
    departs = {}
    active_arcs = []

    vehicles[0] = drones
    y[0].start = float(drones)

    departs[0] = 0
    d[0].start = 0

    for i in range(1,n+1):
        tasks[i-1] = [c[0,i], 0, i]
    
    for i,j in A:
        x[i,j].start = 0.0

    while (False in visited):
        if(drones == 0):
            pickups = list(filter(lambda x: x[1] == 1, tasks))
            pickups = sorted(pickups, key = lambda x: x[0])
            chosen = pickups[0]
            addtime = chosen[0]

            active_arcs.append((current, chosen[2]))
            x[current, chosen[2]].start = 1.0

            chosenwait = 0
            if(addtime - c[current, chosen[2]] > 0):
                chosenwait = addtime - c[current, chosen[2]]

            waits[chosen[2]] = math.ceil(chosenwait)
            w[chosen[2]].start = chosenwait

            current = chosen[2]
            time = time + addtime

            departs[chosen[2]] = math.ceil(time)
            d[chosen[2]].start = float(time)

            chosen[0] = (bound_length+1)**2
            chosen[1] = -1

            visit_order.append(chosen[2])
            rt = rt + str(chosen[2]) + " -> "

            visited[chosen[2]-1] = True
            tasks[chosen[2]-1] = chosen
            drones = drones + 1

            vehicles[chosen[2]] = drones
            y[chosen[2]].start = float(drones)

            deploys[chosen[2]] = 1
            g[chosen[2]].start = 1.0

            route = 0
            if(chosen[2] % 2 == 0):
                route = chosen[2] / 2
            else:
                route = (chosen[2] + 1) / 2
            
            for r in activeroutes:
                if(routetimes[r] <= addtime):
                    routetimes[r] = 0
                else:
                    routetimes[r] = routetimes[r] - addtime
            
            activeroutes.remove(route)
            #just a precaution for current route lol
            routetimes[route] = 0

            for i in range(1,n+1):
                #we already adjusted the task time for our current node
                if(i != chosen[2]):
                    #adjusting task times for deployments
                    if(tasks[i-1][1] == 0):
                        tasks[i-1][0] = c[chosen[2], i]
                    #adjusting task times for 
                    elif(tasks[i-1][1] == 1):
                        wait = 0
                        route = 0

                        if(i % 2 == 0):
                            route = i / 2
                        else:
                            route = (i+1) / 2

                        if(routetimes[route] - c[chosen[2], i] > 0):
                            wait = routetimes[route] - c[chosen[2], i]

                        tasks[i-1][0] = c[chosen[2], i] + wait
          
        else:
            todo = list(filter(lambda x: x[1] != -1, tasks))
            todo = sorted(todo, key = lambda x: x[0])
            chosen = todo[0]
            addtime = chosen[0]
            active_arcs.append((current, chosen[2]))
            x[current, chosen[2]].start = 1.0
            time = time + addtime
            departs[chosen[2]] = math.ceil(time)
            d[chosen[2]].start = float(time)
            route = 0
            partner = 0
            if(chosen[2] % 2 == 0):
                partner = chosen[2] - 1
                route = chosen[2] / 2
            else:
                partner = chosen[2] + 1
                route = (chosen[2] + 1) / 2
            
            chosen[0] = (bound_length+1)**2
            if(chosen[1] == 0):
                #this is a deployment.

                waits[chosen[2]] = 0
                w[chosen[2]].start = 0.0

                tasks[partner - 1][1] = 1 

                drones = drones - 1

                deploys[chosen[2]] = 0
                g[chosen[2]].start = 0.0

                for r in activeroutes:
                    if(routetimes[r] <= addtime):
                        routetimes[r] = 0
                    else:
                        routetimes[r] = routetimes[r] - addtime

                activeroutes.append(route)

                for i in range(1,n+1):
                    #we already adjusted the task time for our current node
                    if(i != chosen[2]):
                        #adjusting task times for deployments
                        if(tasks[i-1][1] == 0):
                            tasks[i-1][0] = c[chosen[2], i]
                        #adjusting task times for pickups
                        elif(tasks[i-1][1] == 1):
                            wait = 0
                            route = 0

                            if(i % 2 == 0):
                                route = i / 2
                            else:
                                route = (i+1) / 2

                            if(routetimes[route] - c[chosen[2], i] > 0):
                                wait = routetimes[route] - c[chosen[2], i]

                            tasks[i-1][0] = c[chosen[2], i] + wait

            else:
                #this is a pickup.
                chosenwait = 0
                if(addtime - c[current, chosen[2]] > 0):
                    chosenwait = addtime - c[current, chosen[2]]

                waits[chosen[2]] = math.ceil(chosenwait)
                w[chosen[2]].start = chosenwait

                deploys[chosen[2]] = 1
                g[chosen[2]].start = 1.0

                drones = drones + 1
                
                for r in activeroutes:
                    if(routetimes[r] <= addtime):
                        routetimes[r] = 0
                    else:
                        routetimes[r] = routetimes[r] - addtime
                
                activeroutes.remove(route)
                #just a precaution for current route lol
                routetimes[route] = 0

                for i in range(1,n+1):
                    #we already adjusted the task time for our current node
                    if(i != chosen[2]):
                        #adjusting task times for deployments
                        if(tasks[i-1][1] == 0):
                            tasks[i-1][0] = c[chosen[2], i]
                        #adjusting task times for 
                        elif(tasks[i-1][1] == 1):
                            wait = 0
                            route = 0

                            if(i % 2 == 0):
                                route = i / 2
                            else:
                                route = (i+1) / 2

                            if(routetimes[route] - c[chosen[2], i] > 0):
                                wait = routetimes[route] - c[chosen[2], i]

                            tasks[i-1][0] = c[chosen[2], i] + wait

            chosen[1] = -1
            visit_order.append(chosen[2])
            rt = rt + str(chosen[2]) + " -> "
            visited[chosen[2]-1] = True
            tasks[chosen[2]-1] = chosen
            vehicles[chosen[2]] = drones
            y[chosen[2]].start = float(drones)
            current = chosen[2]
        print(rt)
    
    active_arcs.append((current, 0))
    x[current, 0].start = 1.0
    objective = math.ceil(departs[current] + c[current, 0])

    print("\nINITIAL OBJECTIVE:", objective)
    
    rt = rt + "0"

    print("\n")
    print(rt)
    print("\n")

    for i in range(len(visit_order)):
        node = visit_order[i]
        print("d[", str(i), "]", ": ", departs[node])
    
    print("\n")
    
    for i in range(1,len(visit_order)):
        node = visit_order[i]
        print("w[", str(i), "]", ": ", waits[node])
    
    print("\n")

    for i in range(1,len(visit_order)):
        node = visit_order[i]
        print("g[", str(i), "]", ": ", deploys[node])
    
    print("\n")
    
    for i in range(len(visit_order)):
        node = visit_order[i]
        print("y[", str(i), "]", ": ", vehicles[node])
    
    deployments = { k : (2*round(v)) - 1 for k,v in deploys.items() }
    rounded_route_times = {i: round(tR[i]) for i in R}
    
    return departs, waits, deployments, rounded_route_times, objective, xc, yc, visit_order
    

def tsp_truck(routedistances, usednodecoords, numdrones, dronevel, truckvel, data, manager, routing, solution, bigM):

    print("\nFinding optimal truck routing...")
        
    n = len(usednodecoords)
    D = int(numdrones)
    vT = float(truckvel)
    vD = float(dronevel)
    
    if(n % 2 != 0):
        print("Number of nodes must be even.")
        return
        
    bound_length = 1000
    
    dRs = np.asarray(routedistances)

    #same as paper, these are drone route times
    tRs = (dRs/vD).tolist()
    
    xc = [0] + usednodecoords
    #xc = np.linspace(0, np.sqrt(2)*bound_length, n+2, dtype = 'int32').tolist()
    #xc.pop(-1)
    yc = xc
    
    #plt.plot(xc[0], yc[0], c='r', marker='s')
    #plt.scatter(xc[1:], yc[1:], c='b')
    
    #sets needed
    V = [i for i in range(1, n+1)]
    eV = [i for i in V if i % 2 == 0]
    oV = [i for i in V if i % 2 == 1]
    #N corresponds to M in the paper
    N = [0] + V
    A = [(i, j) for i in N for j in N if i != j]
    R = [i for i in range(1,(n//2) + 1)]
    c = {(i, j): np.hypot(xc[i]-xc[j], yc[i]-yc[j])/vT for i, j in A}
    #add a very small cost to every 0 cost edge so the solver never has identical departure times for two given nodes
    """ for i,j in A:
        if c[i,j] == 0:
            c[i,j] = 0.01 """

    tR = {i: tRs[i-1] for i in R}
    
    #computed values for better variable bounds
    M = (np.sqrt(2)*bound_length*2)
    costs = list(c.values())
    ckmax = sum(nlargest(n+1, costs))
    cmax = max(costs)
    maxdiff = 0
    for i in N:
        for j in N:
            diff = costs[i] - costs[j]
            if diff > maxdiff:
                maxdiff = diff
    
    mdl = Model('Truck Problem')

    #variables needed
    #x[i,j] is presence of edge in solution (same as paper)
    x = mdl.addVars(A, vtype=GRB.BINARY, name = "x")
    #d[i] is DEPARTURE time at node i (same as paper)
    d = mdl.addVars(N, ub = ckmax + sum(tRs), vtype=GRB.CONTINUOUS, name = "d")
    #w[i] is waiting time at node i (same as paper)
    w = mdl.addVars(V, ub = max(tRs), vtype=GRB.CONTINUOUS, name = "w")
    #g[i] is 1 if node i is recovery site, 0 if launch site (corresponds to c_i in paper)
    g = mdl.addVars(V, vtype=GRB.BINARY, name = "g")
    #y[i] is number of available vehicles after node i (same as paper)
    y = mdl.addVars(N, lb = 0, ub = 1*D, vtype=GRB.INTEGER, name = "y")

    """ print("\nVARIABLE MEANINGS:")
    print("x[i,j] is presence of edge in solution (same as paper)")
    print("d[i] is DEPARTURE time at node i (same as paper)")
    print("w[i] is waiting time at node i (same as paper)")
    print("g[i] is 1 if node i is recovery site, 0 if launch site (corresponds to c_i in paper)")
    print("y[i] is number of available vehicles after node i (same as paper)") """

    #construct a simple initial solution where we just sequentially do route 1, then 2, and so on
    #in_departs, in_waits, in_deployments, in_rounded_route_times, in_objective, in_xc, in_yc, in_visit_order = simple_init_solution(x, d, w, g, y, A, N, V, tR, R, D, c, xc, yc, n)
    in_departs, in_waits, in_deployments, in_rounded_route_times, in_objective, in_xc, in_yc, in_visit_order = complex_init_solution(x, d, w, g, y, A, N, V, tR, R, D, c, xc, yc, n, bound_length)

    in_usednodecoords = [0] + usednodecoords

    #complete_solver_gurobi.graph_solution(data, manager, routing, solution, in_departs, in_waits, in_deployments, in_rounded_route_times, in_objective, in_xc, in_yc, in_visit_order, in_usednodecoords, "initial_solution_test", True)

    #modify bounds on departure times to reflect initial solution objective
    for i in N:
        d[i].ub = in_objective

    mdl.modelSense = GRB.MINIMIZE

    #objective is sum of truck time costs plus truck waiting times
    #mdl.setObjective(l)
    #See Eq. 11 in paper
    mdl.setObjective(
        quicksum(x[i, j]*c[i, j] for i, j in A) + 
        quicksum(w[i] for i in V))
    
    #sum of x[i,j]'s  for all fixed values of i is 1 (exactly one outgoing edge from every node)
    mdl.addConstrs(quicksum(x[i, j] for j in N if j != i) == 1 for i in N)
    
    #sum of x[i,j]'s  for all fixed values of j is 1 (exactly one incoming edge from every node)
    mdl.addConstrs(quicksum(x[i, j] for i in N if i != j) == 1 for j in N)
    
    #if edge (i,j) is in solution, then d[j] = d[i] + c[i,j] + w[j] (waiting time at node j)
    #also functions as Miller-Tucker-Zemlin constraint since it prevents subcycles
    #See Eq. 15 in paper
    if(not(bigM)):
        mdl.addConstrs((x[i, j] == 1) >> (d[i]+c[i,j]+w[j] == d[j]) for i, j in A if j != 0)
    else:
        #replace with Big-M constraint: M = initial solution objective
        mdl.addConstrs(d[j] - d[i] - c[i,j] - w[j] <= in_objective*(1-x[i,j]) for i, j in A if j != 0)
        mdl.addConstrs(d[j] - d[i] - c[i,j] - w[j] >= -1*in_objective*(1-x[i,j]) for i, j in A if j != 0)
        pass

    #set departure time at node 0 to be 0
    #See Eq. 15 in paper
    d[0].ub = 0.0
    
    #if a[j] < a[j-1] then g[j] = 0, otherwise g[j] = 1 for even j
    #g[j] = 0 means starting new route, otherwise completing route
    #See Eq. 18 in paper
    if(not(bigM)):
        mdl.addConstrs((g[j] == 1) >> (d[j] - d[j-1] >= tR[j/2]) for j in eV)
    else:
        #replace with Big-M constraint: M = -1*(initial solution objective + largest route time)
        mdl.addConstrs(d[j] - d[j-1] - tR[j/2] >= -1*(in_objective + max(tRs))*(1-g[j]) for j in eV)
    
    #if a[j] < a[j+1] then g[j] = 0, otherwise g[j] = 1 for odd j
    #g[j] = 0 means starting new route, otherwise completing route
    #See Eq. 18 in paper
    if(not(bigM)):
        mdl.addConstrs((g[j] == 1) >> (d[j] - d[j+1] >= tR[(j+1)/2]) for j in oV)
    else:  
        #replace with Big-M constraint: M = -1*(initial solution objective + largest route time)
        mdl.addConstrs(d[j] - d[j+1] - tR[(j+1)/2] >= -1*(in_objective + max(tRs))*(1-g[j]) for j in oV)

    #if g[j] = 1 for some node, then its partner node should have g[j] = 0
    #this is because launch and recovery sites are paired
    #See Eq. 17 in paper
    mdl.addConstrs(g[j] + g[j-1] == 1 for j in eV)
    mdl.addConstrs(g[j] + g[j+1] == 1 for j in oV)
    
    #if edge (i,j) is in solution, then capacity y[j] = y[i] + (2*g[j]-1)
    #this is because g[j] is a binary variable:
    #g[j] = 0 if j is launch site, g[j] = 1 if j is recovery site
    #so 2*g[j] - 1 = -1 when g[j] = 0 and 1 when g[j] = 1
    #this is the change in # drones at node j
    #This covers Eq. 16 and 19 in the paper
    if(not(bigM)):
        mdl.addConstrs((x[i, j] == 1) >> (y[i] + 2*g[j] - 1 == y[j]) for i, j in A if j != 0)
    else:
        #replace with Big-M constraint: M = D+1
        mdl.addConstrs(y[j] - y[i] - (2*g[j]) + 1 <= (D+1)*(1-x[i,j]) for i, j in A if j != 0)
        mdl.addConstrs(y[j] - y[i] - (2*g[j]) + 1 >= -1*(D+1)*(1-x[i,j]) for i, j in A if j != 0)
    
    #0 <= y[j] <= D (# of drones) for all j
    #See Eq. 20 in paper
    #mdl.addConstrs(y[j] >= 0 for j in V)
    #mdl.addConstrs(y[j] <= D for j in V)
    
    #set drone availability at node 0 (depot) to be D
    #see E. 19 in paper
    y[0].lb = D
    y[0].ub = D
    
    mdl.params.MIPGap = 0.0001
    mdl.params.Method = 5
    mdl.params.TimeLimit = 30  # seconds
    #mdl.params.ImproveStartTime = 30 #seconds
    mdl.params.ImproveStartNodes = 1 #nodes
    mdl.params.MIPFocus = 1
    mdl.optimize()

    active_arcs = [a for a in A if x[a].x > 0.98]

    depart_times = { k : round(v.X) for k,v in d.items() }
    exact_depart_times = { k : v.X for k,v in d.items() }
    wait_times = { k : round(v.X) for k,v in w.items() }
    exact_wait_times = { k : v.X for k,v in w.items() }
    deployments = { k : (2*round(v.X)) - 1 for k,v in g.items() }
    rounded_route_times = {i: round(tR[i]) for i in R}
    exact_route_times = {i: tR[i] for i in R}
    objective = round(mdl.objVal)
    exact_objective = mdl.objVal

    print("\nOBJECTIVE:", objective)

    active_arcs = sorted(active_arcs, key=lambda x: x[0])
    
    visit_order = [0]
    route = "0 -> "
    i = 0
    curr = 0
    while i < n:
        curr = active_arcs[curr][1]
        if(curr != 0):
            visit_order.append(curr)
        route = route + str(curr) + " -> "
        i += 1
    route = route + "0"

    print(route)

    ds = []
    ws = []
    gs = []
    ys = []

    for v in mdl.getVars():
        if("d" in v.varName):
            ds.append((v.varName, v.x))
        elif ("w" in v.varName):
            ws.append((v.varName, v.x))
        elif ("g" in v.varName):
            gs.append((v.varName, v.x))
        elif ("y" in v.varName):
            ys.append((v.varName, v.x))
    
    print("\n")

    for i in range(len(visit_order)):
        node = visit_order[i]
        print("d[", str(i), "]", ": ", ds[node][1])
    
    print("\n")
    
    for i in range(1,len(visit_order)):
        node = visit_order[i]
        print("w[", str(i), "]", ": ", ws[node-1][1])
    
    print("\n")

    for i in range(1,len(visit_order)):
        node = visit_order[i]
        print("g[", str(i), "]", ": ", gs[node-1][1])
    
    print("\n")
    
    for i in range(len(visit_order)):
        node = visit_order[i]
        print("y[", str(i), "]", ": ", ys[node][1])
    
    return depart_times, wait_times, deployments, rounded_route_times, objective, xc, yc, visit_order, exact_depart_times, exact_wait_times, exact_route_times, exact_objective

def tsp_truck_curved(routedistances, usednodecoords, numdrones, dronevel, truckvel, depot, line, dim):

    print("Finding optimal truck routing...")
        
    n = len(usednodecoords[0])
    D = int(numdrones)
    vT = float(truckvel)
    vD = float(dronevel)
    
    if(n % 2 != 0):
        print("Number of nodes must be even.")
        return
        
    bound_length = dim
    
    dRs = np.asarray(routedistances)
    tRs = (dRs/vD).tolist()
    
    xc = [depot.x] + usednodecoords[0]
    #xc = np.linspace(0, np.sqrt(2)*bound_length, n+2, dtype = 'int32').tolist()
    #xc.pop(-1)
    yc = [depot.y] + usednodecoords[1]
    
    #plt.plot(xc[0], yc[0], c='r', marker='s')
    #plt.scatter(xc[1:], yc[1:], c='b')
    
    #sets needed
    V = [i for i in range(1, n+1)]
    eV = [i for i in V if i % 2 == 0]
    oV = [i for i in V if i % 2 == 1]
    N = [0] + V
    A = [(i, j) for i in N for j in N if i != j]
    R = [i for i in range(1,(n//2) + 1)]
    c = {(i, j): (abs(line.project(geom.Point(xc[i], yc[i])) - line.project(geom.Point(xc[j], yc[j]))))/vT for i, j in A}
    #add a very small cost to every 0 cost edge so the solver never has identical departure times for two given nodes
    for i,j in A:
        if c[i,j] == 0:
            c[i,j] = 0.01

    tR = {i: tRs[i-1] for i in R}
    
    #computed values for better variable bounds
    M = (np.sqrt(2)*bound_length*2)
    costs = list(c.values())
    ckmax = sum(nlargest(n+1, costs))
    cmax = max(costs)
    maxdiff = 0
    for i in N:
        for j in N:
            diff = costs[i] - costs[j]
            if diff > maxdiff:
                maxdiff = diff
    
    mdl = Model('Truck Problem')

    #variables needed, most are intermediate helpers lol
    x = mdl.addVars(A, vtype=GRB.BINARY, name = "x")
    a = mdl.addVars(N, ub = ckmax + sum(tRs), vtype=GRB.CONTINUOUS, name = "a")
    w = mdl.addVars(V, ub = max(tRs), vtype=GRB.CONTINUOUS, name = "w")
    h = mdl.addVars(V, vtype=GRB.BINARY, name = "h")
    b = mdl.addVars(R, lb = -1*(ckmax + sum(tRs)), ub = ckmax + sum(tRs), vtype=GRB.CONTINUOUS, name = "b")
    s = mdl.addVars(R, ub = ckmax + sum(tRs), vtype=GRB.CONTINUOUS, name = "s")
    z = mdl.addVars(R, ub = ckmax + sum(tRs), vtype=GRB.CONTINUOUS, name = "z")
    #d = mdl.addVars(R, ub = max(maxdiff,max(tRs)), vtype=GRB.CONTINUOUS, name = "d")
    e = mdl.addVars(R, vtype=GRB.BINARY, name = "e")
    f = mdl.addVars(V, lb = -1, ub = 1, vtype=GRB.INTEGER, name = "f")
    g = mdl.addVars(V, vtype=GRB.BINARY, name = "g")
    y = mdl.addVars(N, lb = 0, ub = 1*D, vtype=GRB.INTEGER, name = "y")
    #l = mdl.addVar(lb = 0, ub = ckmax + sum(tRs), vtype=GRB.CONTINUOUS, name = "l")    
    mdl.modelSense = GRB.MINIMIZE

    #objective is sum of truck time costs plus truck waiting times
    #mdl.setObjective(l)
    mdl.setObjective(
        quicksum(x[i, j]*c[i, j] for i, j in A) + 
        quicksum(w[i] for i in V))
    
    #sum of x[i,j]'s  for all fixed values of i is 1 (exactly one outgoing edge from every node)
    mdl.addConstrs(quicksum(x[i, j] for j in N if j != i) == 1 for i in N)
    
    #sum of x[i,j]'s  for all fixed values of j is 1 (exactly one incoming edge from every node)
    mdl.addConstrs(quicksum(x[i, j] for i in N if i != j) == 1 for j in N)

    #set l to be the cost of the entire tour (last d[i] + cost of node i to node 0).
    #mdl.addConstrs((x[i, 0] == 1) >> (l == a[i]+c[i,0]) for i in V)
    #mdl.addConstr(l == max_(a[i] for i in N))

    #set arrival time at node 0 to be 0
    a[0].ub = 0.0
    
    #if edge (i,j) is in solution, then a[j] = a[i] + c[i,j] + w[j] (waiting time at node j)
    #also functions as Miller-Tucker-Zemlin constraint since it prevents subcycles
    mdl.addConstrs((x[i, j] == 1) >> (a[i]+c[i,j]+w[j] == a[j]) for i, j in A if j != 0)
    
    #b[i] = a[2i] - a[2i-1] (truck travel time between route endpoints, including waiting time), is a helper
    mdl.addConstrs(b[i] == a[2*i] - a[(2*i)-1] for i in R)
    
    #s[i] = |b[i]|, is another helper variable
    for i in R:
        mdl.addGenConstrAbs(s[i],b[i])

    #z[i] = s[i] - waiting times at both endpoints. 
    #One of waiting times is guaranteed to be 0, other one can be 0 or positive.
    #this gives the true truck-only travel time between route endpoints
    mdl.addConstrs(z[i] == s[i] - w[2*i] - w[(2*i)-1] for i in R)
    
    #if z[i] <= tR[i] then e[i] = 1
    #mdl.addConstrs(tR[i] - z[i] <= (M/vD)*e[i] for i in R)
    #if e[i] = 1 then z[i] <= tR[i], completes the iff statement
    #means e[i] = 1 when truck travel time is less than drone travel time
    #means e[i] = 0 when truck travel time is greater than drone travel time
    mdl.addConstrs((e[i] == 1) >> (z[i] <= tR[i]) for i in R)
    mdl.addConstrs((e[i] == 0) >> (z[i] >= tR[i]) for i in R)
    
    #if a[j] < a[j-1] then g[j] = 0, otherwise g[j] = 1 for even j
    #g[j] = 0 means starting new route, otherwise completing route
    #mdl.addConstrs(a[j-1] - a[j] <= (M/vT)*g[j] for j in eV)
    mdl.addConstrs((g[j] == 0) >> (a[j] <= a[j-1]) for j in eV)
    mdl.addConstrs((g[j] == 1) >> (a[j] >= a[j-1]) for j in eV)
    
    #if a[j] > a[j-1] then g[j] = 0, otherwise g[j] = 1 for odd j
    #g[j] = 0 means starting new route, otherwise completing route
    #mdl.addConstrs(a[j+1] - a[j] <= (M/vT)*g[j] for j in oV)
    mdl.addConstrs((g[j] == 0) >> (a[j] <= a[j+1]) for j in oV)
    mdl.addConstrs((g[j] == 1) >> (a[j] >= a[j+1]) for j in oV)

    #if g[j] = 0 (ending pending route) and 
    #e[j/2] = 1 (even j), or e[(j+1)/2] = 1 (odd j),
    #which means (truck time < drone time),
    #then w[j] = tR[j/2] - z[j/2] for even j
    #then w[j] = tR[(j+1)/2] - z[(j+1)/2 + 1] for odd j
    #otherwise w[j] = 0
    
    #h[i] is 1 when both are true, false otherwise
    mdl.addConstrs(h[i] == and_(g[i], e[i/2]) for i in eV)
    mdl.addConstrs(h[i] == and_(g[i], e[(i+1)/2]) for i in oV)

    #if h[i] = 0, then w[i] = 0
    #else, when h[i] = 1, then w[i] = drone time - truck time
    mdl.addConstrs((h[i] == 0) >> (w[i] == 0) for i in eV)
    mdl.addConstrs((h[i] == 0) >> (w[i] == 0) for i in oV)
    mdl.addConstrs((h[i] == 1) >> (w[i] == tR[i/2] - z[i/2]) for i in eV)
    mdl.addConstrs((h[i] == 1) >> (w[i] == tR[(i+1)/2] - z[(i+1)/2]) for i in oV)
    
    #if g[j] = 0, then f[j] = -1 since we are starting a new route, so capacity decreases
    mdl.addConstrs((g[j] == 0) >> (f[j] == -1) for j in V)
    #if g[j] = 1, then f[j] = 1 since we are completing old route, so capacity increases
    mdl.addConstrs((g[j] == 1) >> (f[j] == 1) for j in V)
    
    #if edge (i,j) is in solution, then capacity y[j] = y[i] + f[j]
    mdl.addConstrs((x[i, j] == 1) >> (y[i]+f[j] == y[j]) for i, j in A if j != 0)
    
    #0 <= y[j] <= D (# of drones) for all j
    mdl.addConstrs(y[j] >= 0 for j in V)
    mdl.addConstrs(y[j] <= D for j in V)
    
    #set drone availability at node 0 (depot) to be D
    y[0].lb = D
    y[0].ub = D
    
    mdl.params.MIPGap = 0.05
    mdl.params.Method = 5
    mdl.params.TimeLimit = 60  # seconds
    mdl.params.ImproveStartNodes = 1 #nodes
    mdl.params.MIPFocus = 1
    #mdl.params.ImproveStartTime = 1 #seconds
    #mdl.params.ImproveStartNodes = 1 #nodes
    #mdl.params.MIPFocus = 1
    mdl.optimize()
    
    """ print("\nVARIABLE MEANINGS:")
    print("x[i,j] is presence of edge in solution")
    print("a[i] is DEPARTURE time at node i")
    print("w[i] is waiting time at node i")
    print("h[i] is 1 if node i is collection node and if truck time < drone time")
    print("b[i] is signed truck travel time, including waiting time after collection, between endpoints of route i")
    print("s[i] is absolute truck travel time, including waiting time after collection, between endpoints of route i")
    print("z[i] is absolute truck travel time, excluding waiting time after collection, between endpoints of route i")
    print("e[i] is 1 if truck travel time z[i] on route i is less than drone travel time on route i, 0 otherwise")
    print("d[i] equals z[i] if e[i] is 1, tR[i] otherwise")
    print("f[i] equals -1 if starting new route at node i, 1 if completing old route at node i")
    print("g[i] equals 0 if truck arrives at node i before other endpoint of route, 1 otherwise.")
    print("y[i] is the number of available vehicles after node i") """

    #print("--------\n")
    #print("coords:", *xc, "\n")
    #print("drone route times:", tR, "\n")
    #print(c)
    
    """ for v in mdl.getVars():
        if(not("x" in v.varName)):
            print('%s %g' % (v.varName, v.x)) """

    active_arcs = [a for a in A if x[a].x > 0.98]

    depart_times = { k : round(v.X) for k,v in a.items() }
    wait_times = { k : round(v.X) for k,v in w.items() }
    deployments = { k : round(v.X) for k,v in f.items() }
    rounded_route_times = {i: round(tR[i]) for i in R}
    objective = round(mdl.objVal)

    print("\nOBJECTIVE:", objective)

    active_arcs = sorted(active_arcs, key=lambda x: x[0])
    
    visit_order = [0]
    route = "0 -> "
    i = 0
    curr = 0
    while i < n:
        curr = active_arcs[curr][1]
        if(curr != 0):
            visit_order.append(curr)
        route = route + str(curr) + " -> "
        i += 1
    route = route + "0"
    
    #print(route)

    """
    for i in R:
        e1 = xc[(2*i)-1]
        e2 = xc[(2*i)]
        c1x = e1 + (30*i)
        c1y = e1 - (30*i)
        c2x = e2 + (30*i)
        c2y = e2 - (30*i)
        plt.plot([e1, c1x], [e1, c1y], c='b', zorder=0)
        plt.plot([c1x, c2x], [c1y, c2y], c='b', zorder=0)
        plt.plot([c2x, e2], [c2y, e2], c='b', zorder=0)
    
        
    for i, j in active_arcs:
        plt.plot([xc[i], xc[j]], [yc[i], yc[j]], c='y', zorder=0)
        plt.plot(xc[0], yc[0], c='r', marker='s')
        plt.scatter(xc[1:], yc[1:], c = 'w', linewidth = 7)
        
    for i in range(len(visit_order)):
        dex = visit_order[i]
        plt.text(xc[dex],yc[dex],i+1, ha="center", va="center")
    
    plt.show()
    """
    
    return depart_times, wait_times, deployments, rounded_route_times, objective, xc, yc, visit_order


def tsp_truck_ortools(routedistances, usednodecoords, numdrones, dronevel, truckvel):

    print("Finding optimal truck routing...")
        
    n = len(usednodecoords)
    D = int(numdrones)
    vT = float(truckvel)
    vD = float(dronevel)
    
    if(n % 2 != 0):
        print("Number of nodes must be even.")
        return
        
    bound_length = 1000
    
    dRs = np.asarray(routedistances)
    tRs = (dRs/vD).tolist()
    
    xc = [0] + usednodecoords
    #xc = np.linspace(0, np.sqrt(2)*bound_length, n+2, dtype = 'int32').tolist()
    #xc.pop(-1)
    yc = xc
    
    #plt.plot(xc[0], yc[0], c='r', marker='s')
    #plt.scatter(xc[1:], yc[1:], c='b')
    
    #sets needed
    V = [i for i in range(1, n+1)]
    eV = [i for i in V if i % 2 == 0]
    oV = [i for i in V if i % 2 == 1]
    N = [0] + V
    A = [(i, j) for i in N for j in N if i != j]
    R = [i for i in range(1,(n//2) + 1)]
    c = {(i, j): round(np.hypot(xc[i]-xc[j], yc[i]-yc[j])/vT) for i, j in A}
    #add a very small cost to every 0 cost edge so the solver never has identical departure times for two given nodes
    for i,j in A:
        if c[i,j] == 0:
            c[i,j] = 1

    tR = {i: round(tRs[i-1]) for i in R}
    
    #computed values for better variable bounds
    M = (np.sqrt(2)*bound_length*2)
    costs = list(c.values())
    ckmax = sum(nlargest(n+1, costs))
    cmax = max(costs)
    maxdiff = 0
    for i in N:
        for j in N:
            diff = costs[i] - costs[j]
            if diff > maxdiff:
                maxdiff = diff
    
    mdl = Model('Truck Problem')

    #variables needed, most are intermediate helpers lol
    x = mdl.addVars(A, vtype=GRB.BINARY, name = "x")
    a = mdl.addVars(N, ub = ckmax + sum(tRs), vtype=GRB.CONTINUOUS, name = "a")
    w = mdl.addVars(V, ub = max(tRs), vtype=GRB.CONTINUOUS, name = "w")
    h = mdl.addVars(V, vtype=GRB.BINARY, name = "h")
    b = mdl.addVars(R, lb = -1*(ckmax + sum(tRs)), ub = ckmax + sum(tRs), vtype=GRB.CONTINUOUS, name = "b")
    s = mdl.addVars(R, ub = ckmax + sum(tRs), vtype=GRB.CONTINUOUS, name = "s")
    z = mdl.addVars(R, ub = ckmax + sum(tRs), vtype=GRB.CONTINUOUS, name = "z")
    #d = mdl.addVars(R, ub = max(maxdiff,max(tRs)), vtype=GRB.CONTINUOUS, name = "d")
    e = mdl.addVars(R, vtype=GRB.BINARY, name = "e")
    f = mdl.addVars(V, lb = -1, ub = 1, vtype=GRB.INTEGER, name = "f")
    g = mdl.addVars(V, vtype=GRB.BINARY, name = "g")
    y = mdl.addVars(N, lb = 0, ub = 1*D, vtype=GRB.INTEGER, name = "y")
    #l = mdl.addVar(lb = 0, ub = ckmax + sum(tRs), vtype=GRB.CONTINUOUS, name = "l")    
    mdl.modelSense = GRB.MINIMIZE

    model = cp_model.CpModel()

    x = {}
    for (i,j) in A:
        x[(i,j)] = model.NewBoolVar('x[%i,%i]' % (i, j))
    
    a = {}
    for i in N:
        if i == 0:
            a[i] = model.NewIntVar(0, 0, 'a[%i]' % i)
        else:
            a[i] = model.NewIntVar(0, int(ckmax + sum(tRs)), 'a[%i]' % i)

    w = {}
    for i in V:
        w[i] = model.NewIntVar(0, int(max(maxdiff,max(tRs))), 'w[%i]' % i)
    
    h = {}
    for i in V:
        h[i] = model.NewBoolVar('h[%i]' % i)
    
    b = {}
    for i in R:
        b[i] = model.NewIntVar(int(-1*(ckmax + sum(tRs))), int(ckmax + sum(tRs)), 'b[%i]' % i)

    s = {}
    for i in R:
        s[i] = model.NewIntVar(0, int(ckmax + sum(tRs)), 's[%i]' % i)
    
    z = {}
    for i in R:
        z[i] = model.NewIntVar(0, int(ckmax + sum(tRs)), 'z[%i]' % i)
    
    e = {}
    for i in R:
        e[i] = model.NewBoolVar('e[%i]' % i)
    
    f = {}
    for i in V:
        f[i] = model.NewIntVar(-1, 1, 'f[%i]' % i)
    
    g = {}
    for i in V:
        g[i] = model.NewBoolVar('g[%i]' % i)
    
    y = {}
    for i in N:
        if i == 0:
            y[i] = model.NewIntVar(int(D), int(D), 'y[%i]' % i)
        else:
            y[i] = model.NewIntVar(0, int(D), 'y[%i]' % i)
        
    
    p = {}
    for i in eV:
        p[i] = model.NewIntVar(0, 2, 'p[%i]' % i)
    
    q = {}
    for i in oV:
        q[i] = model.NewIntVar(0, 2, 'p[%i]' % i)

    objective_terms = []

    for i,j in A:
        objective_terms.append(int(c[i,j]) * x[i,j])
    
    for i in V:
        objective_terms.append(w[i])

    model.Minimize(sum(objective_terms))

    for i in N:
        model.Add(sum(x[i, j] for j in N if j != i) == 1)

    for j in N:
        model.Add(sum(x[i, j] for i in N if i != j) == 1)
    
    for i,j in A:
        if(j != 0):
            model.Add(int(c[i,j]) + a[i] + w[j] == a[j]).OnlyEnforceIf(x[i,j])
            #model.Add(int(c[i,j]) + a[i] == a[j]).OnlyEnforceIf(x[i,j])
            model.Add(y[i]+f[j] == y[j]).OnlyEnforceIf(x[i,j])
    
    for i in R:
        model.Add(b[i] == a[2*i] - a[(2*i)-1])
        model.AddAbsEquality(s[i], b[i])
        model.Add(z[i] == s[i] - w[2*i] - w[(2*i)-1])
        model.Add(z[i] <= tR[i]).OnlyEnforceIf(e[i])
        model.Add(z[i] >= tR[i]).OnlyEnforceIf(e[i].Not())
    
    for j in eV:
        model.Add(a[j] <= a[j-1]).OnlyEnforceIf(g[j].Not())
        model.Add(a[j] >= a[j-1]).OnlyEnforceIf(g[j])
        model.Add(p[j] == g[j] + e[j/2])
        model.Add(p[j] == 2).OnlyEnforceIf(h[j])
        model.Add(p[j] <= 1).OnlyEnforceIf(h[j].Not())
        model.Add(w[j] == 0).OnlyEnforceIf(h[j].Not())
        model.Add(w[j] == tR[j/2] - z[j/2]).OnlyEnforceIf(h[j])
    
    for j in oV:
        model.Add(a[j] <= a[j+1]).OnlyEnforceIf(g[j].Not())
        model.Add(a[j] >= a[j+1]).OnlyEnforceIf(g[j])
        model.Add(q[j] == g[j] + e[(j+1)/2])
        model.Add(q[j] == 2).OnlyEnforceIf(h[j])
        model.Add(q[j] <= 1).OnlyEnforceIf(h[j].Not())
        model.Add(w[j] == 0).OnlyEnforceIf(h[j].Not())
        model.Add(w[j] == tR[(j+1)/2] - z[(j+1)/2]).OnlyEnforceIf(h[j])
    
    for j in V:
        model.Add(f[j] == -1).OnlyEnforceIf(g[j].Not())
        model.Add(f[j] == 1).OnlyEnforceIf(g[j])
        
    solver = cp_model.CpSolver()

    solver.parameters.max_time_in_seconds = 60

    status = solver.Solve(model)

    depart_times = {}
    wait_times = {}
    deployments = {}
    rounded_route_times = {}
    objective = -1
    visit_order = []

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        active_arcs = []
        for (i,j) in A:
            if (solver.BooleanValue(x[i,j]) > 0):
                active_arcs.append((i,j))
        
        depart_times = {k : round(solver.Value(v)) for k,v in a.items() }
        wait_times = { k : round(solver.Value(v)) for k,v in w.items() }
        deployments = { k : round(solver.Value(v)) for k,v in f.items() }
        rounded_route_times = {i: round(tR[i]) for i in R}
        objective = round(solver.ObjectiveValue())

        active_arcs = sorted(active_arcs, key=lambda x: x[0])
        
        visit_order = [0]
        route = "0 -> "
        i = 0
        curr = 0
        while i < n:
            curr = active_arcs[curr][1]
            if(curr != 0):
                visit_order.append(curr)
            route = route + str(curr) + " -> "
            i += 1
        route = route + "0" 
    else:
        print('No solution found.')
    
    return depart_times, wait_times, deployments, rounded_route_times, objective, xc, yc, visit_order

if __name__ == '__main__':
    main()