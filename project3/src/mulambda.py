from ea import EA
import random
import helpers

class MuLambda(EA):

    def __init__(self, shape, mu, lambdeh, alpha):
        super().__init__(shape, mu)
        self.lambdeh = lambdeh
        self.sigma = 10
        self.alpha = alpha

        if lambdeh % 2 != 0:
            raise "Lamda is not divis 2 fix it"
		
    def train(self, x, y, validationX, validationY, maxGen = 100000):
        super().train()
		#mark the time step, or generation
        t = 0
        
        self.fitness={}
        self.fitnext={}

        #evaluate the fitness of the population
        #need to establish train_set, validation_set#######
        for i in range(len(self.pop)):
            self.fitness.update({i : self.evaluateFitness(self.pop[i], x, y)})
        
        #sort by fitness
        self.sortFit = sorted(self.fitness.items(), key=lambda x:x[1])
        
        #store current best individual
        best = max(self.fitness, key=(lambda key: self.fitness[key]))

        converged = False
        while(t <= maxGen and not converged):
            
            t = t +1
            self.selectFrom()
            offSpring = self.crossOver()
            newPop = self.mutate(offSpring, self.parents, x, y)
            #print(offSpring)

            #add each child to population
            for i in range(len(newPop)):
                self.pop.append(newPop[i])
                        
            #new pop list
            self.sortPop = sorted(self.pop, key=lambda j:self.evaluateFitness(j, x, y))
            self.pop = self.sortPop[len(newPop):]
            
            #update
            for i in range(len(self.pop)):
                self.fitness.update({i : self.evaluateFitness(self.pop[i], x, y)})          
            
            #new sortFit list
            self.sortFit = sorted(self.fitness.items(), key=lambda x:x[1]) 
            
            #store current best individual
            best = self.pop[-1]
            
            #print(self.sortFit[-1][1])

            converged = self.postIterationProcess(validationX, validationY)


    def crossOver(self):
        offspring=[]
        x1 = self.pop[self.parents[0][0]]
        #for each other parent, pair it with x1
        for i in range(1, len(self.parents)):
            offspring.extend(self.uniCross(x1, self.pop[self.parents[i][0]]))
        return(offspring)
        
    #given two parents, create two offspring using a binary mask
    def uniCross(self, x1, x2):
        spawn = []
        goodTwin = []
        evilTwin = []
        mask =[]
       #chromLength = len(self.pop[self.parents[0][0]])
        chromLength = len(x1)
        #x2 = self.pop[self.parents[1][0]]
        #print(x1[0:2])
        #print(x2[0:2])
        
        #build binary mask
        for i in range(chromLength):
            mask.append(random.randint(0, 1))
        #print(mask[0:2])
    
        #create first offspring
        for i in range(chromLength):
            if mask[i] == 1:
                goodTwin.append(x1[i])
            else:
                goodTwin.append(x2[i])
        spawn.append(goodTwin)
        #create second offspring     
        for i in range(chromLength):
            if mask[i] == 1:
                evilTwin.append(x2[i])
            else:
                evilTwin.append(x1[i])
        spawn.append(evilTwin)
        
        #print(spawn[0][0:2])
        return spawn

    def selectFrom(self):
        self.parents = helpers.rankBasedSelection(self.pop, self.sortFit, int(self.lambdeh / 2) + 1)
       
    def mutate(self, childrens, parents, x, y):

        parents = list(map(lambda i: self.pop[i[0]], parents)) # get actual parent objs
        avgParentsFitness = sum(getAllFitness(self, parents, x, y)) / len(parents)
        childrensFitness = getAllFitness(self, childrens, x, y)

        oneFifthBetter = len(list(filter(lambda fitness: fitness >= avgParentsFitness, childrensFitness))) / len(childrens) > .2

        if oneFifthBetter:
            self.sigma *= self.alpha
        else:
            self.sigma /= self.alpha

        for i in range(len(childrens)):
            for j in range(len(childrens[i])):
                childrens[i][j] = random.normalvariate(0, self.sigma)
                
            #print(check)
        return(childrens)


def getAllFitness(network, guys, x, y):
    return list(map(lambda guy: network.evaluateFitness(guy, x, y), guys))