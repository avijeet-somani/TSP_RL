import sys
#sys.path.append('/Users/avijeet/Desktop/avijeet/work related/code/vs projects/hard problems/tsp-solver1/TSP-solver-using-reinforcement-learning/elkai')
import elkai
import numpy as np
import torch

#CONST = 100000.0
CONST = 1.0
def calc_dist(p, q):
    return np.sqrt(((p[1] - q[1])**2)+((p[0] - q[0]) **2)) * CONST

def get_ref_reward(pointset):
    
    if isinstance(pointset, torch.cuda.FloatTensor) or  isinstance(pointset, torch.FloatTensor):
        pointset = pointset.detach().numpy()

    
    num_points = len(pointset)
    ret_matrix = np.zeros((num_points, num_points))
    
    '''
    for i in range(num_points):
        for j in range(i+1, num_points):
            ret_matrix[i,j] = calc_dist(pointset[i], pointset[j])
    '''

    for i in range(num_points):
        for j in range(num_points):
            ret_matrix[i,j] = calc_dist(pointset[i], pointset[j])

    #print(ret_matrix)
    cities = elkai.DistanceMatrix(ret_matrix)  
          
    q = cities.solve_tsp()
    #print('get_ref_reward : ' , q)
    #q = elkai.solve_float_matrix(ret_matrix) # Output: [0, 2, 1]
    
    
    
    dist = 0
    for i in range(num_points):
        dist += ret_matrix[q[i], q[(i+1) % num_points]]
        #print(dist)
    return dist / CONST , q
