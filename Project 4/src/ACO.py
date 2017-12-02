#Ant colony optimization
#currently displaying SACO behavior

import numpy as np
import random
from _overlapped import NULL

class Aco():
    #2D grid
    grid=[]
    
    #all ants, {ant:(x_pos,ypos,unladen, item)}
    colony = {}
    
    data = {}
    max_iter =0
    
    
    def __init__(self, data, num_ants, grid_dim, max_iter):
        self.data = data
        
        #Build grid, place each data vector randomly on grid
        self.build_grid(grid_dim)
        
        
        #place all ants randomly on grid
        for i in range(num_ants):
            while(True):
                j = random(0,grid_dim)
                k = random(0,grid_dim)
                if(self.grid[j][k] is None):
                    
                    colony.update({i:{(j,k,False, None)}})#{ant:(x_pos,ypos,unladen, item)}
                    break 
                
        #set initial values for gamma1, gamma2, gamma and
        self.max_iter = max_iter
        
         
    def optimize(self):
        #for each ant in the colony
        for ant in self.colony:
            #get ant, and position info
            curr_ant = self.colony.get(ant)
            ant_x = curr_ant[0]
            ant_y = curr_ant[1]
            curr_item = self.grid[ant_x][ant_y]
            #if the ant is unladen and site is occupied by item y_a
            if curr_ant[2] == False and curr_item == None:
                #compute local density
                loc_dens=self.local_density()
                #compute pick up rate
                pick=self.pick_up_rate(curr_item)
                r = random.uniform(0,1)
                #if the pick up rate is higher than some random number
                if r <= pick_up:
                    #pick dat shit up
                    self.colony.update({ant:ant_x, ant_y, True, curr_item})
                    self.grid[ant_x,ant_y] = None
            else:
                #if the ant carrying an item and the current site is empty
                if curr_ant[2] ==True and curr_item == None:
                    #compute local density
                    loc_dens=self.local_density()
                    #compute drop_rate
                    drop = self.drop_rate(curr_item)
                    r = random.uniform(0,1)
                    if r <= drop:
                        #drop dat shit
                        self.colony.update({ant:ant_x, ant_y, False,None})
                        self.grid[ant_x, ant_y] = curr_item
                                  
            #move ant to a randomly selected neighboring site not occupied by another ant
            neighborhood = []
            #populate neighborhood with empty adjacent spots
            x=[ant_x-1, ant_x, ant_x+1]
            y=[ant_y-1, ant_y, ant_y+1]
            moves=np.permutation(x,y)
            while(True)
            #randomly pick one and move ant to it
                move = random.sample(moves)
                spot=self.grid[move[0]][move[1]
                if spot[1] == None:
                    self.colony.update({ant:(move[0],move[1],curr_ant[2], curr_ant[3])})
                    break
            
                
                
    def local_density(self,item):
        dens = []
        for y_b in neighborhood:
            dens.append((1 - (np.linalg.norm(item, y_b)/gamma)))
        sim = 1 / n**2 * sum(dens)
        if sim > 0:
            return(sim)
        else:
            return(0)
        
        
    def pick_up_rate(self, item):
        pick_up = (gamma1 / (gamma1 + self.local_density(item)))**2
        return(pick_up) 
    
    
    def drop_rate(self, item):
        dens = self.local_density(item)
        if dens < gamma2
            drop = 2*dens
        else:
            drop = 1
        return drop
    
    def build_grid(self):
        pass
        
#     #process for deciding which path to take at each node    
#     #currently shows SACO behavior
#     def decision(self, node):
#         
#         #grab a random probability
#         r = random.rand(0,1)
#         for link in node:
#             #calculate Pa using transProb()
#             Pa=self.transProb()
# 
#             if(r <= Pa):
#                 return link #follow that path
#         pass
#     
#     def transProb(self):
#         pass
#     
#     #pheromone evaporation
#     def evap(self):
#         pass
#     
#     def updatePheromone(self):
#         pass
#     
#     def findMin(self):
#         pass
#     
#     #Send out swarm, optimize path, currently looking for minimum path
#     def search(self):
#         #set time marker to zero
#         t = 0
#         
#         while(True):
#             #holds a path for each ant
#             x_kt = [[]]*len(self.colony)
#             for ant in self.colony:
#                 #construct a path x_kt for the ant
#                 x_kt[ant]= []
#                 while(True):
#                     #Select next node 
#                     link = self.decision()
#                     x_kt[ant].append(link)
#                     if(self.destination in x_kt[ant]):
#                         break
#                 #Remove all loops from path x_kt
#                 
#                 #calculate path length of x_kt
#                 
#             for link in self.v:
#                 #pheromone evaporation
#                 self.evap()
#             
#             for ant in colony:
#                 for link in x_kt[ant]:
#                     #update pheromone
#                     self.updatePheromone()
#             t =+ 1
# 
#             #check stopping condition
#             #terminate when either maxIter is passed or 
#             #acceptable solution has been found, findMine(x_kt) < epsilon
#             if(stop):
#                 #find min_path
#                 min_path = self.findMin(x_kt)
#                 return(min_path)
#             
#             
#             
#                 
                
        