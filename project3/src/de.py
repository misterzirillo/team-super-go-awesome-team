from ea import EA
import random
import helpers

class DE(EA):
    
    def __init__(self, shape, mu, beta, proboRecombo):
        #super(shape, max)
        super().__init__(shape, mu)
        self.parents = []
        self.beta = beta
        self.pr = proboRecombo

    def check(self):
        print("Network shape: "+ str(self.shape) + "\n"+ "Individual Length: " +  str(len(self.pop[1])) + "\n"+ "First five weights: " + str(self.pop[1][0:5]) + "\n")

    def crossOver(self, parent, trialVec):
        #use the trial vectors to create offspring

        #uniform crossover between parent and trial vector

        #compare parent to child, winner survives
        pass

    def mutate(self, parent):
        #for each parent (xi)

        #select the target, the best individual (x1)

        #randomly select two different(x2,x3 !=x1, xi)

        #create trial vector, append to list of trial vectors
        pass

    def selectFrom(self):
        pass

    def train(self, x, y, valX, valY, maxGen):
        ###Initialize###
        super().train() # sets up some vars
        #mark the time step, or generation
        t = 0
        self.fitness={}
        self.fitnext={}

        #evaluate the fitness of the population
        for i in range(len(self.pop)):
            self.fitness.update({i : self.evaluateFitness(self.pop[i], x, y)})
        #sort by fitness
        self.sortFit = sorted(self.fitness.items(), key=lambda x:x[1])
        #store current best individual
        best = max(self.fitness, key=(lambda key: self.fitness[key]))

        # ###Loop###
        # while(t <= maxGen and not converged):
        #     #nextGen
        #     t = t +1
        #     newPop = []

        #     #for each individual
        #     for i in range(self.pop):
        #         guy = self.pop[i]
        #         #evaluate fitness
        #         fit = self.evaluateFitness(guy, x, y)
        #         #create trial vector, mutate
        #         tv = self.mutate(guy)
        #         #create the offspring, crossover
        #         offSpring = self.crossOver(guy, tv)
        #         newFit = self.evaluateFitness(offSpring, x, y)
        #         #if child better than parent
        #         if newFit > fit:
        #             #add child to nextPop
        #             newPop.append(offSpring)
        #         #else add parent to nextPop
        #         else:
        #             newPop.append(guy)

        #     #generational replacement
        #     self.pop = newPop
            
        #     #check convergence
        #     converged = self.postIterationProcess(validationX, validationY)

        #     #new pop list
        #     #self.sortPop = sorted(self.pop, key=lambda j:self.evaluateFitness(j, x, y))
        #     #self.pop = self.sortPop[len(newPop):]
            
        #     #update
        #     #for i in range(len(self.pop)):
        #         #self.fitness.update({i : self.evaluateFitness(self.pop[i], x, y)})          
            
        #     #new sortFit list
        #     #self.sortFit = sorted(self.fitness.items(), key=lambda x:x[1]) 
            
        #     #store current best individual
        #     #best = self.pop[-1]
            
        #     #print(self.sortFit[-1][1])

            