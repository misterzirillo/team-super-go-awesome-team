'''gbest Particle Swarm Optimization
'''
import numpy as np
import random

class PSO():
    
    data = {}
    agents = {}
    gbest = []
    inertia = 0
    constriction = 0
    grid = []
    

        #initialize each particle to contain Nc random centroids
        #unti max_iter
            #for each particle
                #for each data vector
                    #calculate euclidean distance to centroids
                    #assign to closest centroid
                    #calculate fitness
                #update gbest and lbest
                #update centroids

    def __init__(self, data, num_agents, num_centroids):
        self.data = data
        num_dims = len(next(iter(data)))
        self.max_vals = next(iter(data))
        self.min_vals = next(iter(data))
        for data_key in data.iterkeys():
            k = 0
            for feature in data_key:
                if feature > self.max_vals[k]:
                    self.max_vals[k] = feature
                elif feature < self.min_vals[k]:
                    self.min_vals[k] = feature
                k += 1

        for i in range(num_agents):
            new_agent_key = "agent" + i
            centroids = []
            for k in range(num_centroids):
                centroid = []
                for j in range(num_dims):
                    centroid.append([random.randrange(self.min_vals[j], self.max_vals[j])])
                centroids.append(centroid)
            agents[new_agent_key] = centroids
    
    def velocity(self):
        pass
    
    def optimize(self):
        pass
    
    def output(self):
        pass
    
    def update_position(self):
        pass