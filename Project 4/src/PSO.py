'''gbest Particle Swarm Optimization
'''
import numpy as np
import random

class PSO():
    
    data = {}
    agents = {}
    fitness_scores = []
    lbests = [] #holds subarrays  of lbest fitness and centroids for each particle
    gbest = [] #fitness and centroids of gbest
    inertia = 0
    constriction = 0
    grid = []
    

        #initialize each particle to contain Nc random centroids
        #
        #unti max_iter
            #for each particle
                #for each data vector
                    #calculate euclidean distance to centroids
                    #assign to closest centroid
                    #calculate fitness
                #update gbest and lbest
                #update centroids

    #

    def __init__(self, data, num_agents, num_centroids, inertia, c1,c2,max_iter):
        self.agents = {}
        self.fitness_scores = [None]*num_agents
        self.lbests = [[9999999999,[]]]*num_agents #holds subarrays  of lbest fitness and centroids for each particle
        self.gbest = [999999999,[]]
        self.data = data
        self.max_iter = max_iter
        self.inertia = inertia
        self.c1 = c1
        self.c2 = c2
        num_dims = len(next(iter(data)))
        self.max_vals = next(iter(data))
        self.max_vals = list(self.max_vals)
        self.min_vals = next(iter(data))
        self.min_vals = list(self.min_vals)
        for data_key in data.keys():
            k = 0
            for feature in data_key:
                if feature > self.max_vals[k]:
                    self.max_vals[k] = feature
                elif feature < self.min_vals[k]:
                    self.min_vals[k] = feature
                k += 1
        for i in range(num_agents):
            new_agent_key = "agent" + str(i)
            centroids = [None]*num_centroids #needs to be a dictionary, key is coords, value is list of data vectors assigned
            velocities = [[None]*num_dims]*num_centroids
            for k in range(num_centroids):
                centroid = []
                for j in range(num_dims):
                    centroid.append(random.uniform(self.min_vals[j], self.max_vals[j]))
                    velocities[k][j] = (0.01*random.uniform(self.min_vals[j], self.max_vals[j]))
                    #print(k)
                    #print(velocities[k])
                centroids[k] = centroid
            self.agents[new_agent_key] = [centroids,[[]]*num_centroids,velocities]
    

    def velocity(self):
        pass
    
    #agents is dictionary of agents: key = agent_i, data = [[list of centroids],[lists of data vectors[]],[list of velocities]]

    def optimize(self):
        for iter in range(self.max_iter):
            i = -1
            for agent in self.agents:
                #assign data points to nearest centroid
                i+=1
                for x in range(len(self.agents[agent][1])):
                    self.agents[agent][1][x] = []
                for data_key in self.data.keys():
                    distances = []
                    for centroid in self.agents[agent][0]:
                        distances.append(np.linalg.norm(np.array(list(data_key)) - np.array(centroid)))
                    min_dist = min(distances)
                    self.agents[agent][1][distances.index(min_dist)].append([data_key,min_dist])
                #calculate fitness
                fitness = 0
                j = -1
                sum = 0
                for centroid in self.agents[agent][0]:
                    j+=1
                    size = len(self.agents[agent][1][j])
                    for data_vector in self.agents[agent][1][j]:
                        sum += data_vector[1]/size #((np.linalg.norm(np.array(list(data_vector)) - np.array(centroid)))/size)
                if sum == 0:
                    print(sum)
                fitness = sum / len(self.agents[agent][0])
                if fitness == 0:
                    print("0 fitness")
                    #print(self.agents[agent][1])
                self.fitness_scores[i] = fitness
                if fitness < self.lbests[i][0]:
                    print('new lbest')
                    self.lbests[i] = [fitness, self.agents[agent][0]]
                    if fitness < self.gbest[0]:
                        print('new gbest')
                        self.gbest = [fitness, self.agents[agent][0]]
                #update cluster velocity
                j = 0
                for velocity in self.agents[agent][2]:
                    for x in range(len(velocity)):
                       self.agents[agent][2][j][x] += self.inertia * velocity[x]
                       self.agents[agent][2][j][x] += self.c1*random.uniform(0,1)*(self.lbests[0][1][j][x] - self.agents[agent][0][j][x])
                       self.agents[agent][2][j][x] += self.c2*random.uniform(0,1)*(self.gbest[1][j][x] - self.agents[agent][0][j][x])
                    j += 1
                #update cluster position    
                j=0
                for centroid in self.agents[agent][0]:
                    for x in range(len(centroid)):
                        self.agents[agent][0][j][x] += self.agents[agent][2][j][x]
                    j+=1
            print(self.fitness_scores)
            print(self.gbest[0])
    

    def output(self):
        pass
    
    def update_position(self):
        pass