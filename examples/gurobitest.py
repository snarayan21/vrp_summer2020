import numpy as np
import matplotlib.pyplot as plt
from gurobipy import Model, GRB, quicksum

rnd = np.random
rnd.seed(0)

n = 30  # numbre of clients
xc = rnd.rand(n+1)*200
yc = rnd.rand(n+1)*100

plt.plot(xc[0], yc[0], c='r', marker='s')
plt.scatter(xc[1:], yc[1:], c='b')

N = [i for i in range(1, n+1)]
V = [0] + N
A = [(i, j) for i in V for j in V if i != j]
c = {(i, j): np.hypot(xc[i]-xc[j], yc[i]-yc[j]) for i, j in A}
#Q = 20
#q = {i: rnd.randint(1, 10) for i in N}

mdl = Model('CVRP')

x = mdl.addVars(A, vtype=GRB.BINARY)
u = mdl.addVars(V, ub = n+1, vtype=GRB.CONTINUOUS)
mdl.modelSense = GRB.MINIMIZE
mdl.setObjective(quicksum(x[i, j]*c[i, j] for i, j in A))
mdl.addConstrs(quicksum(x[i, j] for j in V if j != i) == 1 for i in V)
mdl.addConstrs(quicksum(x[i, j] for i in V if i != j) == 1 for j in V)
mdl.addConstrs((x[i, j] == 1) >> (u[i]+1 == u[j]) for i, j in A if j != 0)
#mdl.addConstrs(u[i] >= q[i] for i in N)
#mdl.addConstrs(u[i] <= Q for i in N)
mdl.Params.MIPGap = 0.1
mdl.Params.TimeLimit = 60  # seconds
mdl.optimize()

active_arcs = [a for a in A if x[a].x > 0.99]

for i, j in active_arcs:
    plt.plot([xc[i], xc[j]], [yc[i], yc[j]], c='g', zorder=0)
    plt.plot(xc[0], yc[0], c='r', marker='s')
    plt.scatter(xc[1:], yc[1:], c='b')
    
plt.show()