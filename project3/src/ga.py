from ea import EA
from mlpnetwork import MLPNetwork
import numpy as np
import random
import helpers

class GA(EA):
	
	def __init__(self, shape, mu):
		#super(shape, max)
		super().__init__(shape, mu)
		self.parents = []
		
	def check(self):
		print(self.shape, self.trueShape, len(self.pop[1]), self.pop[1][1:10])
		
	
	def train(self, x, y, valX, valY, maxGen):
		super().train() # sets up some vars
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
		self.bestEva = [self.pop[-1], self.evaluateFitness(self.pop[-1], x, y)]
		best = max(self.fitness, key=(lambda key: self.fitness[key]))
		print ("Starting Best at "+str(best) + " with fitness ", str(self.fitness[best]))

		converged = False
		while(t <= maxGen and not converged):
			#create test_fold
			
			t = t +1
			self.selectFrom()
			offSpring = self.crossOver()
			newPop = self.mutate(offSpring)
			#print(offSpring)

			#add each child to population
			for i in range(len(newPop)):
				self.pop.append(newPop[i])
						
			#new pop list
			self.sortPop = sorted(self.pop, key=lambda j:self.evaluateFitness(j, x, y))
			self.pop = self.sortPop[len(newPop):] # elitist replacement
			
			#update
			for i in range(len(self.pop)):
				self.fitness.update({i : self.evaluateFitness(self.pop[i], x, y)})          
			
			#new sortFit list
			self.sortFit = sorted(self.fitness.items(), key=lambda x:x[1])
 
			
			#store current best individual
			best = [self.pop[-1], self.evaluateFitness(self.pop[-1], x, y)]

			if best[1] > self.bestEva[1] and t > 2:
				self.bestEva = best
				print ("Current Best at fitness " + str(best[1]) + " after " + str(t) +" generations")
				print("Training Error: " + str(self.trainingErrors[-1]) + "\tValidation Error: " + str(self.validationErrors[-1]))
			
			#print(self.sortFit[-1][1])
			converged = self.postIterationProcess(valX, valY, best[1], t)
			if t == maxGen:
				print("Max generations reached.")
			
			
	
	#select the parents from the population
	#rank based
	#bugs, never picking the worst or best individual
	def selectFrom(self):
		self.parents = helpers.rankBasedSelection(self.pop, self.sortFit, 2)
		
	#generate offspring according to the crossover rate
	#global, uniform
	# should return a whole new copy of the population
	# example: self.pop = self.crossOver()
	def crossOver(self):
		super().crossOver()
		
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
		

	#mutate the offspring according to the mutation rate
	# should return a whole new copy of the population
	# example: self.pop = self.mutate()

	def mutate(self, specimens):
		super().mutate()
		#probability of mutation on a feature
		Pm =.1
		check = []
		for i in range(len(specimens)):
			for j in range(len(specimens[i])):
				num = random.random()
				if(num < Pm):
					specimens[i][j] = random.normalvariate(specimens[i][j], 1)
					check.append(("yes", (i, j)))
				else:
					check.append(("no", (i, j)))
			#print(check)
		return(specimens)
					
	#find the slackers and replace them with new dudes
	def replace(self, newPop, fitNext):
		for i in range(len(newPop)-1):
			pass

		
		
		
		
		
		
		
		
		
				
			
	