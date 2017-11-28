#Ant colony optimization
#currently displaying SACO behavior

import numpy as np
import random
from _overlapped import NULL

class Aco():
    #nodes
    u = []
    origin = None
    destination = None
    
    #edges
    v= []
    
    #all ants
    colony = []
    
    
    def __init__(self):
        #Build graph, set origin and destination
        
        #initialize pheromones to small random values on each path
        
        #place all ants on the origin node
        pass
        
    #process for deciding which path to take at each node    
    #currently shows SACO behavior
    def decision(self, node):
        
        #grab a random probability
        r = random.rand(0,1)
        for link in node:
            #calculate Pa using transProb()
            Pa=self.transProb()

            if(r <= Pa):
                return link #follow that path
        pass
    
    def transProb(self):
        pass
    
    #pheromone evaporation
    def evap(self):
        pass
    
    def updatePheromone(self):
        pass
    
    def findMin(self):
        pass
    
    #Send out swarm, optimize path, currently looking for minimum path
    def search(self):
        #set time marker to zero
        t = 0
        
        while(True):
            #holds a path for each ant
            x_kt = [[]]*len(self.colony)
            for ant in self.colony:
                #construct a path x_kt for the ant
                x_kt[ant]= []
                while(True):
                    #Select next node 
                    link = self.decision()
                    x_kt[ant].append(link)
                    if(self.destination in x_kt[ant]):
                        break
                #Remove all loops from path x_kt
                
                #calculate path length of x_kt
                
            for link in self.v:
                #pheromone evaporation
                self.evap()
            
            for ant in colony:
                for link in x_kt[ant]:
                    #update pheromone
                    self.updatePheromone()
            t =+ 1

            #check stopping condition
            #terminate when either maxIter is passed or 
            #acceptable solution has been found, findMine(x_kt) < epsilon
            if(stop):
                #find min_path
                min_path = self.findMin(x_kt)
                return(min_path)
            
            
            
                
                
        