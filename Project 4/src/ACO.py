#Ant colony optimization
#currently displaying SACO behavior

import numpy as np
import random
from random import randint
from _overlapped import NULL
import itertools

class Aco():
    #2D grid, holds tuples (ant at position,item at position)
    grid=[]
    
    #all ants, {ant:(x_pos,ypos,holding, item)}
    colony = {}
    
    data = {}
    max_iter =0
    items = {}
    
    
    def __init__(self, data, num_ants, grid_dim, max_iter):
        self.data = data
        
        #assign integer key to data vector
        for i in range(len(self.data)):
            self.items.update({i:self.data[i]})
        
        #Build grid, place each data vector and ant randomly on grid
        self.grid = self.build_grid(grid_dim)
        self.grid_dim = grid_dim
        self.num_ants = num_ants
        self.place_items()
        self.place_ants()
           
        #set initial values for gamma1, gamma2, gamma
        self.max_iter = max_iter
        self.gamma =.5
        self.gamma1=.5
        self.gamma2=.5
    
    def place_items(self):
        #place all items randomly on grid
        for i in range(len(self.data)):
            while(True):
                j = randint(0,self.grid_dim-1)
                k = randint(0,self.grid_dim-1)
                if(self.is_empty(j, k)):
                    #print("Placing item ", i, "at position ",j,",",k)
                    self.grid[j][k][1]=i#{ant:(x_pos, y_pos, holding, item)}
                    break
                
    #place ants randomly on unoccupied spaces
    def place_ants(self):
        #place all ants randomly on grid
        for i in range(self.num_ants+1):
            while(True):
                j = randint(0,self.grid_dim-1)
                k = randint(0,self.grid_dim-1)
                if(self.is_empty(j, k)):
                    #print("Placing ant ", i, "at position ",j,",",k)
                    self.colony.update({i:[j,k,False, None]})#{ant:(x_pos, y_pos, holding, item)}
                    break
                
    #check if a location is unoccupied
    def is_empty(self, x, y):    
         if(self.grid[x][y][0]==None and self.grid[x][y][1] == None):
             return True
         else:
             return False
         
    def optimize(self):
        iter=0
        while(True):
            #for each ant in the colony
            for ant in self.colony:
                #get ant, and position info
                curr_ant = self.colony.get(ant)
                #print(curr_ant)
                ant_x = curr_ant[0]
                ant_y = curr_ant[1]
                curr_item = self.grid[ant_x][ant_y][1]
                #if the ant is unladen and site is occupied by item y_a
                if curr_ant[2] == False and curr_item != None:
                    #compute local density
                    loc_dens=self.local_density(curr_item,ant_x, ant_y)
                    #compute pick up rate
                    pick=self.pick_up_rate(curr_item, loc_dens)
                    r = random.uniform(0,1)
                    #if the pick up rate is higher than some random number
                    if r <= pick_up:
                        #pick dat shit up
                        print("Ant ", ant, "picked up item ", curr_item )
                        self.colony.update({ant:(ant_x, ant_y, True, curr_item)})
                        self.grid[ant_x,ant_y] = None
                else:
                    #if the ant carrying an item and the current site is empty
                    if curr_ant[2] ==True and curr_item == None:
                        #compute local density
                        loc_dens=self.local_density(curr_ant[3], ant_x, ant_y)
                        #compute drop_rate
                        drop = self.drop_rate(curr_item, loc_dens)
                        
                        r = random.uniform(0,1)
                        if r <= drop:
                            #drop dat shit
                            print("Ant ", ant, "dropped item ", curr_item )
                            self.colony.update({ant:(ant_x, ant_y, False,None)})
                            self.grid[ant_x][ant_y][1] = curr_item
                                      
                #move ant to a randomly selected neighboring site not occupied by another ant
                moves = self.get_neighborhood(ant_x, ant_y)
                while True:
                    moves_tried =0
                    #randomly pick one and move ant to it
                    move = random.choice(moves)
                    move_x = move[0]
                    move_y = move[1]
                    spot = self.grid[move_x][move_y]
                    if spot[0] == None and spot[1]==None:
                        self.colony.update({ant:(move[0],move[1],curr_ant[2], curr_ant[3])})
                        break
                    else:
                        moves_tried +=1
                        if moves_tried >8:
                            print("This ant is stuck.  Plz halp")
            iter +=1
            if iter == self.max_iter:
                print("Job's done")
                break
            
                
    #the local density of a data vector within the ants neighborhood
    #equation 17.45 in engelbrecht            
    def local_density(self, item, x, y):
        dens = []
        item = self.items[item]
        neighborhood= self.get_neighborhood(x, y)
        for y_b in neighborhood:
            item_b =self.grid[y_b[0]][y_b[1]][1]
            item_b = self.items[item_b]
            if item_b != None:
                dens.append((1 - (np.linalg.norm(item- item_b)/gamma)))
        n = len(neighborhood)
        sim = 1 / n**2 * sum(dens)
        if sim > 0:
            return(sim)
        else:
            return(0)
    
    #find all adjacent locations
    def get_neighborhood(self, x, y):
        #populate neighborhood with empty adjacent spots
        if x is 0:
            x=[x, x+1]
        elif x == (len(self.grid)-1):
            x = [x-1, x]
        else:
            x=[x-1,x,x+1]
        if y == 0:
            y=[y, y+1]
        elif y is (len(self.grid)-1):
            y = [y-1, y]
        else:
            y=[y-1,y,y+1]
        moves =[]
        #print(x,y)
        for hoz in x:
            for ver in y:
                moves.append([hoz,ver])
        return moves
        
    #computing the pick up probability 
    #equation 17.46 in engelbrecht 
    def pick_up_rate(self, item, dens):
        pick_up = (gamma1 / (gamma1 + dens))**2
        return(pick_up) 
    
    #computing the drop probability
    #equation 17.47 in engelbrecht
    def drop_rate(self, item, dens):
        item = self.items[item]
        if dens < gamma2:
            drop = 2*dens
        else:
            drop = 1
        return drop
    
    #build a 2D grid for the ants to traverse
    #access a position self.grid[x][y]
    #ant at self.grid[x][y][0], item at self.grid[x][y][1] 
    def build_grid(self,dim):
        grid=[[(None,None)]*dim]*dim
        grid = np.array(grid)
        return grid
    