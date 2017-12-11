'''K-means clustering

'''
import numpy as np
import random
import helpers
import re

class kmeans():
    
    #centroids
    K =[]
    #big D, holds x's
    data = []
    actuals=[]
    
    #dictionary the holds the actual data vectors of the centroids
    keys = {}
    
    #big C holds all c clusters, dictionary of lists. keys are the centroids, values are the points in the cluster
    C = {}
    
    def __init__(self, data, num_k, max_iter):
        self.data = helpers.get_input_matrix_from_dict(data)

        self.max_iter = max_iter
        
        # initialize the k centroids randomly
        K = random.sample(self.data.tolist(), num_k)
        
        # create integer keys to represent the centroids, {int :[data vector of centroid]}
        for i in range(len(K)):
            self.keys.update({i: K[i]})
            


    def optimize(self):
        iter = 0
        #until the centroids stop changing
        while(True):
            iter+=1

            for key in self.keys:
                self.C.update({key: []})

            #for all xi in the data
            for xi in self.data:
                #compute the closest centroid, int key representing cluster
                xi_c=self.find_centroid(xi)
                self.C[xi_c].append(xi)

            #recalculate means based on the new clusters
            update = self.recalc_centroids()
            
            if iter == self.max_iter:
                return self.C
#             print("Kmeans iteration %s." % iter)
#             #if no change to centroids
#             if(update==False):
#                 #return the centroids
#                 return self.C
    
    #find the closest centroid to the data point
    def find_centroid(self, x):
        return min(self.C.keys(), key=lambda c: np.linalg.norm(x - self.keys[c]))

    
    #determine new centroids for each cluster and check convergence
    def recalc_centroids(self):
        update = False
        new_C = {}
        #for each cluster
        for c in self.C:
            #get members of cluster
            clust_list = self.C.get(c)
            clust = np.array(clust_list)
            
            #find new center
            ci = np.mean(clust, axis=0)

            #check if change to centroid
            if not np.array_equal(self.keys.get(c), ci):
                update = True
                
            #add to new dictionary
            new_C.update({c: ci})

        #update cluster centroids
        self.keys = new_C
        return update

