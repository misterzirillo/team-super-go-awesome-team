#Ant colony optimization
#currently displaying SACO behavior

import numpy as np
import random
from random import randint
from _overlapped import NULL
import itertools
import matplotlib.pyplot as plt

class Aco():
    #2D grid, holds tuples (ant at position,item at position)
    grid=[]
    
    #all ants, {ant:(x_pos,ypos,holding, item)}
    colony = {}
    
    data = {}
    max_iter =0
    items = {}
    classes=[]
    
    
    def __init__(self, data, classes, num_ants, grid_dim, max_iter):
        self.data = data
        self.classes = classes
            
        #assign integer key to data vector
        for i in range(len(self.data)):
            self.items.update({i:self.data[i]})
        
        #Build grid, place each data vector and ant randomly on grid
        self.grid = self.build_grid(grid_dim)
        self.grid_dim = grid_dim
        self.num_ants = num_ants
        self.place_items()
#         for i in range(len(self.grid)):
#             for j in range(len(self.grid)):
#                 item=self.grid[i][j][1]
#                 print(item)
        self.place_ants()
#         for i in range(len(self.grid)):
#             for j in range(len(self.grid)):
#                 ant=self.grid[i][j][0]
#                 print(ant)
           
        #set initial values for gamma1, gamma2, gamma
        self.max_iter = max_iter
        #scale of dissimilarity, too large mis-clustering, too little small clusters are formed
        self.gamma =30
        #pick up constant
        self.gamma1=.35
        #drop constant
        self.gamma2=.7
    
    def place_items(self):
        #place all items randomly on grid
        for i in range(len(self.data)):
            while(True):
                j = randint(0,self.grid_dim-1)
                k = randint(0,self.grid_dim-1)
                if(self.is_empty(j, k, 'item')):
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
                if(self.is_empty(j, k, 'ant')):
                    #print("Placing ant ", i, "at position ",j,",",k)
                    self.grid[j][k][0] = i
                    self.colony.update({i:[j,k,False, None]})#{ant:(x_pos, y_pos, holding, item)}
                    break
                
    #check if a location is unoccupied
    def is_empty(self, x, y, type): 
        if type == 'ant':   
            if(self.grid[x][y][0]==None):
                return True
            else:
                return False
        else:
            if(self.grid[x][y][1]==None):
                return True
            else:
                return False
         
    def optimize(self):
        #plot initial item positions
        self.plot_items()
        iter=0
        while(True):
            #for each ant in the colony
            for ant in self.colony:
                #print(ant)
                #get ant, and position info
                curr_ant = self.colony.get(ant)
                #print(curr_ant)
                #print(curr_ant)
                ant_x = curr_ant[0]
                ant_y = curr_ant[1]
                curr_item = self.grid[ant_x][ant_y][1]
                
                #if the ant is unladen and site is occupied by item y_a
                if not curr_ant[2] and curr_item is not None:
                    #print(curr_ant[2])
                    #compute local density
                    loc_dens=self.local_density(curr_item,ant_x, ant_y)
                    #compute pick up rate
                    pick_up=self.pick_up_rate(curr_item, loc_dens)
                    r = random.uniform(0,1)
                    #print(pick_up," ", r)
                    #if the pick up rate is higher than some random number
                    if r <= pick_up:
                        #pick dat shit up
                        print("Ant ", ant, "picked up item ", curr_item )
                        #self.colony.update({ant:[ant_x, ant_y, True, curr_item]})
                        curr_ant[2] = True
                        curr_ant[3] = curr_item
                        #print(curr_ant)
                        self.grid[ant_x][ant_y][1] = None
                        
                elif curr_ant[2]:
                    #print("hey man, dropping this item")  
                    if curr_item is None:
                        #if the ant carrying an item and the current site is empty
                        #print("double check")
                        #compute local density
                        loc_dens=self.local_density(curr_ant[3], ant_x, ant_y)
                        #compute drop_rate
                        drop = self.drop_rate(loc_dens)
                        
                        r = random.uniform(0,1)
                        if r <= drop:
                            #drop dat shit
                            print("Ant ", ant, "dropped item ", curr_ant[3])
                            #self.colony.update({ant:[ant_x, ant_y, False,None]})
                            curr_ant[2] = False
                            curr_ant[3] = None
                            self.grid[ant_x][ant_y][1] = curr_item
                
                                      
                #move ant to a randomly selected neighboring site not occupied by another ant
                moves = self.get_neighborhood(ant_x, ant_y)
                if curr_ant[2] ==False:
                    while True:
                        moves_tried =0
                        #randomly pick one and move ant to it
                        move = random.choice(moves)
                        move_x = move[0]
                        move_y = move[1]
                        spot = self.grid[move_x][move_y]
                        if spot[0] == None:
                            #self.colony.update({ant:[move[0],move[1],curr_ant[2], curr_ant[3]]})
                            curr_ant[0]=move_x
                            curr_ant[1]=move_y
                            self.grid[ant_x][ant_y][0]= None
                            self.grid[move_x][move_y][0]=ant
                            break
                        else:
                            moves_tried +=1
                            if moves_tried >8:
                                print("This ant is stuck.  Plz halp")
#                 else:
#                     while True:
#                         moves_tried =0
#                         #randomly pick one and move ant to it
#                         move = random.choice(moves)
#                         move_x = move[0]
#                         move_y = move[1]
#                         spot = self.grid[move_x][move_y]
#                         if spot[0] == None and spot[1]==None:
#                             self.colony.update({ant:(move[0],move[1],curr_ant[2], curr_ant[3])})
#                             break
#                         else:
#                             moves_tried +=1
#                             if moves_tried >8:
#                                 print("This ant is stuck.  Plz halp")
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
        n=1
        for y_b in neighborhood:
            item_b =self.grid[y_b[0]][y_b[1]][1]
            if item_b != None:
                item_b = self.items[item_b]
                dens.append((1 - (np.linalg.norm(item- item_b)/self.gamma)))
                n +=1
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
        pick_up = (self.gamma1 / (self.gamma1 + dens))**2
        return(pick_up) 
    
    #computing the drop probability
    #equation 17.47 in engelbrecht
    def drop_rate(self, dens):
        #item = self.items[item]
        if dens < self.gamma2:
            drop = 2*dens
        else:
            drop = 1
        #print(drop)
        return(drop)
    
    def plot_items(self):
        x=[]
        y=[]
        z =[]
        nones=0
        count =0
        for i in range(len(self.grid)):
            for j in range(len(self.grid)):
                x.append(i)
                y.append(j)
                item=self.grid[i][j][1]
                #print(item)
                if item == None:
                    nones +=1
                    z.append(0)
                else:
                    count +=1
                    z.append(self.classes[item])
        print(nones, ",", count)
                   
        #build plot of 2D grid
        zi, yi, xi = np.histogram2d(y, x, bins=(len(self.grid),len(self.grid)), weights=z, normed=False)
        counts, _, _ = np.histogram2d(y, x, bins=(len(self.grid),len(self.grid)))
          
        zi = zi / counts
        zi = np.ma.masked_invalid(zi) 
        
        fig, ax = plt.subplots()
        ax.pcolormesh(xi, yi, zi, edgecolors='black')    
        scat = ax.scatter(x, y, c=z, s=20)
        fig.colorbar(scat)
        ax.margins(0.05)
        
        plt.show()
    
    #build a 2D grid for the ants to traverse
    #access a position self.grid[x][y]
    #ant at self.grid[x][y][0], item at self.grid[x][y][1] 
    def build_grid(self,dim):
        grid=[[(None,None)]*dim]*dim
        grid = np.array(grid)
        return grid
    