class GA(EA):
    
    def __init__(self, shape, maxGen):
        #super(shape, max)
        self.shape=shape
        self.maxGen=maxGen
        
      
    def train(self, train_set, validation_set):
        
        #initialize random population
        #10*number of dimensions
        self.pop = initializePop(self,sum(self.shape))
        
        #mark the time step, or generation
        t = 0
        
        #evaluate the fitness of the population
        fitness = self.evaluateFitness(pop)
        
        #store current best individual
        best = [max[fitness]]

        terminated = False
        while(terminated==False):
            #create test_fold
            
            t = t +1
            parents = self.selectFrom(pop)
            offSpring = self.crossOver(parents)
            newPop = self.mutate(offSpring)
            fitness = self.evaluateFitness(newPop)
            pop = self.replace(newPop, fitness)
            if(t== self.maxGen):
                return(best)