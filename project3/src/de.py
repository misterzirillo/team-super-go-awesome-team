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


	def crossOver(self, parent, trialVec):
		#use the trial vectors to create offspring
		mask = []
		spawn = []
		#uniform crossover between parent and trial vector
		#build mask
		for i in range(len(parent)):
			mask.append(random.randint(0, 1))

		#construct child
		for i in range(len(parent)):
			if mask[i] == 1:
				spawn.append(parent[i])
			else:
				spawn.append(trialVec[i])
		#print(len(spawn), spawn[0:2])
		return(spawn)

	def mutate(self, parent, best):
		dudes =[]

		#parent (xi)
		xi = parent
		dudes.append(xi)
		
		#select the target, the best individual (x1)
		if(xi == best):
			x1 = self.pop[-2]
			dudes.append(x1)
		else:
			x1 = best
			dudes.append(x1)
		
		#randomly select two different(x2,x3 !=x1, xi)
		x2 = self.grabDude(dudes)
		dudes.append(x2)
		x3 = self.grabDude(dudes)
		dudes.append(x3)
		# print("xi " + str(xi[0:2]))
		# print("x1 " + str(x1[0:2]))
		# print("x2 " + str(x2[0:2]))
		# print("x3 " + str(x3[0:2]))
		#create trial vector, append to list of trial vectors
		ut=[]
		for i in range(len(xi)):
			ut.append(x1[i] + self.beta*(x2[i]-x3[i]))
		#print(len(ut), ut[0:3])
		return(ut)

	#get a guy that ain't in dudes
	def grabDude(self, dudes):
			while(True):
				#random position in population
				x = self.pop[random.randint(0,len(self.pop)-1)]
				if x in dudes:
					continue
				else:
					return(x)

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
		
		best = max(self.fitness, key=(lambda key: self.fitness[key]))
		print ("Starting Best at "+str(best) + " with fitness ", str(self.fitness[best]))

		#sort population by fitness
		self.sortPop = sorted(self.pop, key=lambda j:self.evaluateFitness(j, x, y))
		self.pop = self.sortPop

		#print(self.evaluateFitness(self.sortPop[-1],x,y))
		#print(self.evaluateFitness(self.pop[-1],x,y))
		#print(len(self.sortPop))
		self.bestEva = [self.pop[-1], self.evaluateFitness(self.pop[-1], x, y)]
		
		##Loop###
		converged = False
		while(t <= maxGen and not converged):
			#nextGen
			t = t +1
			newPop = []

			#store current best individual
			best = self.pop[-1]

			#for each individual
			for i in range(len(self.pop)):
				guy = self.pop[i]
				#evaluate fitness
				fit = self.evaluateFitness(guy, x, y)
				# print(fit)
				#create trial vector, mutate
				tv = self.mutate(guy, best)
				#create the offspring, crossover
				offSpring = self.crossOver(guy, tv)
				newFit = self.evaluateFitness(offSpring, x, y)
				#if child better than parent
				if newFit > fit:
					#add child to nextPop
					newPop.append(offSpring)
				#else add parent to nextPop
				else:
					newPop.append(guy)

			#generational replacement
			self.pop = newPop
			
			#sort population by fitness
			self.sortPop = sorted(self.pop, key=lambda j:self.evaluateFitness(j, x, y))
			self.pop = self.sortPop

			best = [self.pop[-1], self.evaluateFitness(self.pop[-1], x, y)]

			if best[1] > self.bestEva[1] and t > 2:
				self.bestEva = best
				print ("Current Best at fitness " + str(best[1]) + " after " + str(t) +" generations with beta at " + str(self.beta))
				print("Training Error: " + str(self.trainingErrors[-1]) + "\tValidation Error: " + str(self.validationErrors[-1]))
			if t % 30 ==0:
				self.beta = self.beta +.1
			#check convergence
			converged = self.postIterationProcess(valX, valY, best[1], t)

			