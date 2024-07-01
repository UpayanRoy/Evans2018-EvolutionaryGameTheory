import numpy as np
import matplotlib.pyplot as plt 

N = 30 # N**2: population size
T = 1000  # total time
lamda_birth = 0.1  # rate of death
lamda_coin = 0.001  # rate of coin offer
mut = 0.01  # strategy mutation rate

np.random.seed(1009)
strategy_mat = np.random.rand(N, N, 2)
health_mat = np.zeros(shape=(N, N))

initial_mean_strategy = [np.mean(strategy_mat[:, :, 0]), np.mean(strategy_mat[:, :, 1])]
print(initial_mean_strategy)

def list_neighbors(pos: list):
    
    """Returns list of positions of neighbors to input position individual"""

    neigh_list = [[pos[0]-1, pos[1]], [pos[0], pos[1]-1]]
    if pos[0] == N-1 and pos[1] == N-1:
        neigh_list.append([0, N-1])
        neigh_list.append([N-1, 0])
    elif pos[0] == N-1:
        neigh_list.append([N-1, pos[1]+1])
    elif pos[1] == N-1:
        neigh_list.append([pos[0]+1, N-1])
    else:
        neigh_list.append([pos[0]+1, pos[1]])
        neigh_list.append([pos[0], pos[1] + 1])

    return neigh_list

time_list = [0]
strategy_list = [initial_mean_strategy]

for t in range(1, T+1):
    
    
    reproduces = np.random.poisson(lam=lamda_birth, size=(N, N)) == 1
    gets_coin = np.random.poisson(lam=lamda_coin, size=(N, N)) == 1

    for i in range(N):
        for j in range(N):
            
            if gets_coin[i, j]:
                for neighbor in list_neighbors([i, j]):
                    if strategy_mat[i, j, 0] > strategy_mat[neighbor[0], neighbor[1], 1]:
                        # offer fraction > neighbour threshold fraction
                        
                        health_mat[i, j] += 1 - strategy_mat[i, j, 0]
                        health_mat[neighbor[0], neighbor[1]] += strategy_mat[i, j, 0]
                        break
            
            if reproduces[i, j]:
                neighbors = list_neighbors([i, j])
                weakest_neighbor = neighbors[0]
                for neighbor in neighbors:
                    if health_mat[neighbor[0], neighbor[1]] < health_mat[weakest_neighbor[0], weakest_neighbor[1]]:
                        weakest_neighbor = neighbor
                replace_pos = weakest_neighbor
                health_mat[replace_pos[0], replace_pos[1]] = 0
                parent = strategy_mat[i, j]
                strategy_mat[replace_pos[0], replace_pos[1]] = [np.random.normal(loc=parent[0], scale=mut), np.random.normal(loc=parent[1], scale=mut)]

    if t % 10 == 0:
        print(f"Step number: {t}")
        time_list.append(t)
        strategy_list.append([np.mean(strategy_mat[:, :, 0]), np.mean(strategy_mat[:, :, 1])])

plt.plot(time_list, strategy_list)
plt.show()







    

