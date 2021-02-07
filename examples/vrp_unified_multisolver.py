import vrp_multidepot_nodistlim as multidepot_nodistlim
import vrp_multidepot_tu as multidepot_tu
import numpy as np
import sys

#command line function call
def main():
    """Solve the CVRP problem."""
    if len(sys.argv) != 3:
        print('Should be called as follows: python vrp_multipledepots.py [density of nodes per unit^2] [number of depots to approximate diagonal road]')
        return
    
    print("Solving for optimal number of vehicles in 100x100 square area using max distance and total utility methods\n")
    
    density = float(sys.argv[1])
    bound_length = 100
    num_depots = int(sys.argv[2])
    num_nodes = int((bound_length*bound_length)*density) + num_depots
        
    #create depot coordinates along the line y=x in the square region
    depotcoords = np.linspace(0, bound_length, num_depots+2, dtype = 'int32').tolist()
    depotcoords.pop(0)
    depotcoords.pop(-1)
    
    #create data for problem
    xcoords = np.random.randint(low=0,high=bound_length,size=num_nodes).tolist()
    ycoords = np.random.randint(low=0,high=bound_length,size=num_nodes).tolist()
    
    print("\nMINIMIZE MAX DISTANCE METHOD:\n")
    optimal_dist = multidepot_nodistlim.multidepot_nodistlim(density, num_depots, xcoords, ycoords, depotcoords)
    print("------------------------")
    
    #optimal_dist may be None if initial solution was broken, or if last iteration was broken in which case it defaults to previous max_dist
    if(optimal_dist == None):
        print("Initial solution could not be found.")
        return None
        
    print("\nMAXIMIZE TOTAL NODES VISITED METHOD:\n")
    multidepot_tu.multidepot_tu(density, num_depots, optimal_dist, xcoords, ycoords, depotcoords)
    print("------------------------")
    

if __name__ == '__main__':
    main()