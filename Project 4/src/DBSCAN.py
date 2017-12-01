'''DBSCAN clustering
'''
import numpy as np
import random

class DBSCAN():
    
    data = {}
    min_pts = 0
    radius = 0
    labels = {}
    eps = 0
    
    def __init__(self, data, radius, min_pts):
        self.data = data
        self.radius =radius
        self.min_pts = min_pts
        
    def optimize(self):
        C = 0
        #get initial core points
        #for every point in data
        for p in data:
            #if point has a label, skip
            if self.label.get(p) != None:
                continue
            
            #get its neighbors
            neighbors = self.range_query(p)
            #if it has neighbors less than min_pts
            if len(neighbors) < self.min_pts:
                #label as noise
                self.labels.update({p:'Noise'})
                continue
            
            C =+1
            #give p label
            self.labels.update({p:C}) #potential problem here with strings and ints in the dictionary
            seed = neighbors
            while(new_neighbors==False):
                for Q in seed:
                    if self.labels.get(Q) == 'Noise':
                        self.labels.update({Q:C})
                    elif self.labels.get(Q) == None:
                        continue
                    else:
                        self.labels.update({Q:C})
                        neighbors = self.range_query(Q)
                        if len(neighbors) >= self.min_pts:
                            seed.append(neighbors)
                            new_neighbors = True
       
    #scans the data and finds the neighbors of a point
    def range_query(self,p):
        neighbors = []
        for x in self.data:
            if x != p:
                if np.linalg.norm(p,x) <= self.eps:
                    neighbors.append(x)
        return neighbors
    
    
    def output(self):
        print("Wow look at all these clusters\n" + str(key in self.labels))
        pass
    
