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
    keys ={}
    
    #big C holds all c clusters, dictionary of lists. keys are the centroids, values are the points in the cluster
    C = {}
    
    def __init__(self, data, num_k, max_iter):
        self.data = data
        #self.actuals = actuals
        self.max_iter = max_iter
        
        #initialize the k centroids randomly
        self.K = random.sample(self.data.tolist(), num_k)
        
        #create integer keys to represent the centroids, {int :[data vector of centroid]}
        for i in range(len(self.K)):
            self.keys.update({i:self.K[i]})
            
        #update cluster dictionary{int : [data vectors of cluster]}
        for key in self.keys:
            self.C.update({key:[]})
            


    def optimize(self):
        iter = 0
        #until the centroids stop changing
        while(True):
            iter+=1
            #for all xi in the data
            for xi in self.data:
                #compute the closest centroid, int key representing cluster
                xi_c=self.find_centroid(xi)
                
                #assign xi to that closest centroid
                clust = self.C.get(xi_c)
                #print(xi_c)
                clust.append(xi)
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
        dst = {}
        #calculate the distance from the data point to each centroid
        for c in self.C:
            center = self.keys.get(c)
            c_dst = np.linalg.norm(x-center)
            dst.update({c:c_dst}) 
        
        #find min distance
        min_clust=min(dst, key=dst.get)
        return min_clust
    
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
            ci = np.mean(clust, axis = 0)

            #check if change to centroid
            if(np.array_equal(self.keys.get(c), ci)==False):
                update = True
                
            #add to new dictionary
            new_C.update({c:ci})

        #update cluster centroids
        self.keys = new_C
        return update
    
    def output(self):
        print("Oh hai team.  I found these centers:\n" + str(self.C))