'''K-means clustering

'''
import numpy as np
import random
class kmeans():
    
    #centroids
    K =[]
    #big D, holds x's
    data = {}
    
    #big C holds all c clusters, dictionary of lists. keys are the centroids, values are the points in the cluster
    C = {}
    
    def __init__(self, data, num_k):
        self.data=data
        #initialize the k centroids randomly
        self.K = random.sample(self.data, num_k)
        for k in self.K:
            self.C.update({k:[]})

    def optimize(self):
        #until the centroids stop changing
        while(True):
            #for all xi in the data
            for xi in data:
                #compute the closest centroid
                xi_c=self.find_centroid(xi)
                #assign xi to that closest centroid
                clust =self.C.get(xi_c)
                clust.append(xi)
            #recalculate means based on the new clusters
            stop = self.recalc_centroids()
        #if no change to centroids
        if(stop):
            #return the centroids
            break
    
    #find the closest centroid to the data point
    def find_centroid(self, x):
        dst = {}
        #calculate the distance from the data point to each centroid
        for c in self.C:
            c_dst = np.linalg.norm(x,c)
            dst.update({c:c_dst}) 
        
        #find min distance
        c=min(d, key=d.get)
        return c
    
    #determine new centroids for each cluster and check convergence
    def recalc_centroids(self):
        update = False
        new_C = {}
        #for each cluster
        for c in C:
            #get members of cluster
            clust_list = self.C.get(c)
            clust = np.array(clust_list)
            #find new center
            ci = np.mean(clust, axis = 0)
            #check if change to centroid
            if(c != ci):
                update = true
            #add to new dictionary
            new_C.update({ci:{clust_list}})
        #update cluster centroids
        self.C = new_C
        return update
    
    def output(self):
        print("Oh hai team.  I found these centers:\n" + str(self.C))