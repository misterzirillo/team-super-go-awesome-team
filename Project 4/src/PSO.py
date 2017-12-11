'''gbest Particle Swarm Optimization
'''
import numpy as np
import random

class PSO():

    def __init__(self, data, num_agents, num_centroids, inertia, c1,c2,max_iter):
        self.agents = {}                            #key = agent_i, data = [[list of centroids],[lists of data vectors[]],[list of velocities]]
        self.fitness_scores = [None]*num_agents
        self.lbests = [[9999999999,[]]]*num_agents  #holds subarrays  of lbest fitness and centroids for each particle
        self.gbest = [999999999,[]]
        self.data = data
        self.max_iter = max_iter
        self.inertia = inertia                      #inertia value for velocity update
        self.c1 = c1                                #weight for influence of local best position
        self.c2 = c2                                #weight for influence of global best positon
        num_dims = len(next(iter(data)))            #number of dimensions in dataset
        self.max_vals = next(iter(data))            #largest values in each dimension present in dataset
        self.max_vals = list(self.max_vals) 
        self.min_vals = next(iter(data))            #smallest values in each dimension present in dataset
        self.min_vals = list(self.min_vals)
        for data_key in data.keys():                #finding min and max vals
            k = 0
            for feature in data_key:
                if feature > self.max_vals[k]:
                    self.max_vals[k] = feature
                elif feature < self.min_vals[k]:
                    self.min_vals[k] = feature
                k += 1
        for i in range(num_agents):                 #initialize agents. centroid position and velocities random but bounded by min and max vals
            new_agent_key = "agent" + str(i)
            centroids = [None]*num_centroids
            velocities = [[None]*num_dims]*num_centroids
            for k in range(num_centroids):
                centroid = []
                for j in range(num_dims):
                    centroid.append(random.uniform(self.min_vals[j], self.max_vals[j]))
                    velocities[k][j] = (0.001*random.uniform(self.min_vals[j], self.max_vals[j]))
                centroids[k] = centroid
            self.agents[new_agent_key] = [centroids,[[]]*num_centroids,velocities]
     

    #optimization loop. runs for max_iter times
    def optimize(self):
        for iter in range(self.max_iter):
            i = -1
            for agent in self.agents: 
                i+=1
                for x in range(len(self.agents[agent][1])): #clear lists of assigned data points 
                    self.agents[agent][1][x] = []
                for data_key in self.data.keys(): #assign data points to nearest centroid
                    distances = []
                    for centroid in self.agents[agent][0]:
                        dist = np.linalg.norm(np.array(list(data_key)) - np.array(centroid))
                        distances.append(dist)
                    min_dist = min(distances)
                    self.agents[agent][1][distances.index(min_dist)].append([data_key,min_dist,self.data[data_key]]) 
                #calculate fitness for this agent
                fitness = 0
                j = -1
                sum = 0
                for centroid in self.agents[agent][0]:
                    j+=1
                    size = len(self.agents[agent][1][j])
                    for data_vector in self.agents[agent][1][j]:        #for each data point, add distance divided by centroid population count to sum
                        sum += data_vector[1]/size 
                fitness = sum / len(self.agents[agent][0])              #weight fitness by number of centroids
                self.fitness_scores[i] = fitness
                if fitness < self.lbests[i][0]:                         #update local best if previous exceeded, save centroid positions
                    self.lbests[i] = [fitness, self.agents[agent][0]]   
                    if fitness < self.gbest[0]:                         #update global best if previous exceeded, save centroid positions and data point centroid assigments
                        print('new gbest')
                        self.gbest = [fitness, self.agents[agent][0], self.agents[agent][1]]
                j = 0
                for velocity in self.agents[agent][2]:                  #update cluster velocity = inertia*velocity + c1*random(0,1)*(lbest-present) + c2*random(0,1)*(gbest-present)
                    for x in range(len(velocity)):
                       self.agents[agent][2][j][x] += self.inertia * velocity[x]
                       self.agents[agent][2][j][x] += self.c1*random.uniform(0,1)*(self.lbests[0][1][j][x] - self.agents[agent][0][j][x])
                       self.agents[agent][2][j][x] += self.c2*random.uniform(0,1)*(self.gbest[1][j][x] - self.agents[agent][0][j][x])
                    j += 1
                j=0
                for centroid in self.agents[agent][0]:                  #update cluster position = present + velocity
                    for x in range(len(centroid)):
                        self.agents[agent][0][j][x] += self.agents[agent][2][j][x]
                    j+=1
            print(self.gbest[0])
        gbest_clusters = {}                                             #create dictionary of final gbest clusters for data processing
        for i in range(len(self.gbest[2])):
            if len(self.gbest[2][i]) > 0:
                key = "cluster_"+str(i)
                print(self.gbest[2][i])
                points = []
                for x in self.gbest[2][i]:
                    points.append(np.array(list(x[0])))
                gbest_clusters[key] = points
            else:
                pass
        return gbest_clusters