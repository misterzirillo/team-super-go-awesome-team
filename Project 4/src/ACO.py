#Ant colony optimization
#currently displaying SACO behavior

import numpy as np
import random
from _overlapped import NULL

class Aco():
    #2D grid, holds tuples (item at position, ant at position)
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
                    self.colony.update({ant:(ant_x, ant_y, True, curr_item)})
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
                        self.colony.update({ant:(ant_x, ant_y, False,None)})
                        self.grid[ant_x, ant_y] = curr_item
                                  
            #move ant to a randomly selected neighboring site not occupied by another ant
            neighborhood = []
            #populate neighborhood with empty adjacent spots
            x=[ant_x-1, ant_x, ant_x+1]
            y=[ant_y-1, ant_y, ant_y+1]
            moves=np.permutation(x,y)
            while True:
            #randomly pick one and move ant to it
                move = random.sample(moves)
                spot = self.grid[move[0]][move[1]]
                if spot[1] == None:
                    self.colony.update({ant:(move[0],move[1],curr_ant[2], curr_ant[3])})
                    break
            
                
    #the local density of a data vector within the ants neighborhood
    #equation 17.45 in engelbrecht            
    def local_density(self,item):
        dens = []
        for y_b in neighborhood:
            dens.append((1 - (np.linalg.norm(item, y_b)/gamma)))
        sim = 1 / n**2 * sum(dens)
        if sim > 0:
            return(sim)
        else:
            return(0)
        
    #computing the pick up probability 
    #equation 17.46 in engelbrecht 
    def pick_up_rate(self, item):
        pick_up = (gamma1 / (gamma1 + self.local_density(item)))**2
        return(pick_up) 
    
    #computing the drop probability
    #equation 17.47 in engelbrecht
    def drop_rate(self, item):
        dens = self.local_density(item)
        if dens < gamma2
            drop = 2*dens
        else:
            drop = 1
        return drop
    
    #
    def build_grid(self):
        pass
    