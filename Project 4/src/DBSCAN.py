'''DBSCAN clustering
'''
import numpy as np
import helpers
import operator

class DBSCAN():
    
    data = []
    min_pts = 0
    radius = 0
    labels = {}
    eps = 0
    
    keys = {}
    def __init__(self, data, eps, min_pts):
        self.data = helpers.get_input_matrix_from_dict(data)
        self.eps = eps
        self.min_pts = min_pts
        
        #make an integer key to represent data point
        for i in range(len(self.data)):
            self.keys.update({i: self.data[i]})

        
    def optimize(self):
        C = 0
        #get initial core points
        #for every point in data
        for p in self.keys:
            # print("\n####################Point ",p," out of ", len(self.keys),"############################\n")
            # print(self.labels)
            #if point has a label, skip
            if self.labels.get(p) is not None:
                continue
            
            #get its neighbors
            neighbors = self.range_query(p)
            #if it has neighbors less than min_pts
            if len(neighbors) < self.min_pts:
                #label as noise
                self.labels.update({p: 'Noise'})
                continue
            
            C += 1
            #give p label
            # print("Point added to cluster ,",C)
            self.labels.update({p: C})
            seed = neighbors
            new_neighbors = True
            while new_neighbors:
                new_neighbors = False
                for Q in seed:
                    # print("Neighbor ", Q, " out of ", len(seed))
                    #if the point was previously noise, give it the label of the current point
                    if self.labels.get(Q) == 'Noise':
                        #print("Was noise, now member of cluster")
                        self.labels.update({Q: C})
                    #if the point was previously processed
                    elif self.labels.get(Q) is not None:
                        # print("Previously labeled ,", self.labels.get(Q))
                        continue
                    else:
                        # print("Is now member of cluster ", C)
                        self.labels.update({Q: C})
                        neighbors = self.range_query(Q)
                        #density check
                        if len(neighbors) >= self.min_pts:
                            seed = list(set(seed + neighbors))
                            new_neighbors = True
                            
        return self.output()
       
    # scans the data and finds the neighbors of a point
    def range_query(self, p):
        return list(
            map(
                operator.itemgetter(0),
                filter(lambda x: x[0] is not p and np.linalg.norm(x[1] - self.keys[p]) <= self.eps, self.keys.items())))
    
    def output(self):
        cluster_dict = {}
        for label in set(self.labels.values()):
            guys = []
            for k, v in self.labels.items():
                if v is label:
                    guys.append(self.keys[k])

            cluster_dict[label] = guys

        return cluster_dict
