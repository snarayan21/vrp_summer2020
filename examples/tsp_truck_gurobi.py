import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
import math
import gurobipy
from heapq import nlargest
from gurobipy import Model, GRB, quicksum, abs_, and_

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
    
    #z[i] = |b[i]|, is another helper variable
    for i in R:
        mdl.addGenConstrAbs(z[i],b[i])
    
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
    
    #z[i] = |b[i]|, is another helper variable
    for i in R:
        mdl.addGenConstrAbs(z[i],b[i])
    
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
    
    return active_arcs, xc, yc, visit_order

if __name__ == '__main__':
    main()