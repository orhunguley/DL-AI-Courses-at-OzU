# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 17:44:58 2018

@author: og4428
"""

import numpy as np
from matplotlib import pyplot as plt 



def CalculateFitness(realY,predictedY):
    
    errorMatrix = realY - predictedY
    squaredError = errorMatrix * errorMatrix
    squareRootError = np.sqrt(squaredError)
    squareRootError = np.sum(squareRootError,axis=0)
#    squareRootError = squareRootError[np.newaxis,:]
    return squareRootError
    

def CalculateProbabilities(weightErrors):
    
    densities = 1 / weightErrors
    sumOfDensities = sum(densities)
    CDF = densities / sumOfDensities
    CDF = np.cumsum(CDF)
    
    return CDF

def FindIndexOfParent(cdf , prob):
    
    index = 0
    for i in range(len(cdf)):
        
        if prob <= cdf[i]:
            index = i
            return index
    
    return index

def CrossOverOperation( pr1, pr2):
    
    #Uniform crossover is used, two children will be generated
    child1 = []
    child2 = []
    
    for i in range(12):
        n1 = np.random.uniform()
        n2 = np.random.uniform()
        
        if n1 <= 0.5:
            child1.append(pr1[i])
        else:
            child1.append(pr2[i])
            
        if n2 <= 0.5:
            child2.append(pr1[i])
        else:
            child2.append(pr2[i])
        
    mutationProb = np.random.uniform()
    if mutationProb<0.8:
        child2 = MutationOperation(child2)
    
    
    return child1,child2


def MutationOperation(mutatedGene):
    
    
    mutatedChild = mutatedGene.copy()
    rand1 = np.random.randint(0 ,len(mutatedGene))
    rand2 = np.random.randint(0 ,len(mutatedGene))
    rand3 = np.random.randint(0 ,len(mutatedGene))
    
    mutatedChild[rand1] = mutatedChild[rand1] * 1.1
    
    # Control the weights to be in the range [-5,5]
    if mutatedChild[rand1] > 5:
        mutatedChild[rand1] = 5
    
    if mutatedChild[rand1] < -5:
        mutatedChild[rand1] = -5
    
    
    mutatedChild[rand2] = mutatedChild[rand2] * 0.9
    
    mutatedChild[rand3] = np.random.uniform(-5,5)
    
    return mutatedChild



X = np.random.uniform(-5,5,(100,3))
W = np.random.uniform(-1,1,(200,12))
    
Y = np.mean(X , axis=1)
Y = Y[: , np.newaxis]

#W[:,0:3] takes the first 3 column of weights

H1 = np.matmul(X,W[:,0:3].transpose())
H2 = np.matmul(X,W[:,3:6].transpose())

#F1 = np.matmul(H1,W[:,6:8])
#F2 = np.matmul(H1,W[:,8:10])
#F3 = np.matmul(H1,W[:,10:12])

F1 = W[:,6] * H1 + W[:,7] * H2
F2 = W[:,8] * H1 + W[:,9] * H2
F3 = W[:,10] * H1 + W[:,11] * H2

Y_Pred = F1 + F2 + F3
#Y_Pred = np.concatenate((F1,F2,F3),axis=1)
#Y_Pred = np.sum(Y_Pred,axis=1)
errorArray = []
iterationArray = []

for i in range(100001):
    
    # Calculatin Fitness and the and cumulative distributions to be a parent
    squareRootError = CalculateFitness(Y,Y_Pred)
    cdf = CalculateProbabilities(squareRootError)
    
    #creating random numbers to select parents and the gene which is gonna be mutated
    randNumbers = np.random.uniform(0,1, (3))
    
    
    parent1_index = FindIndexOfParent(cdf , randNumbers[0])
    parent2_index = FindIndexOfParent(cdf , randNumbers[1])
    
    mutatedGene_index = FindIndexOfParent(cdf , randNumbers[2])
    
    
    parent1 = W[ parent1_index , : ]
    parent2 = W[ parent2_index , : ]
    mutatedGene = W[mutatedGene_index , : ]
    
    #Crossover and mutation are made below
    if squareRootError[parent1_index] != squareRootError[parent2_index]:
        child1,child2 = CrossOverOperation(parent1 , parent2)
    else:
        child1 = MutationOperation(parent1)
        child2 = MutationOperation(parent2)
        
    
    mutatedChild = MutationOperation(mutatedGene) 
    
    # Put parents and children into an array to sort them according to their fitness
    A = np.array([parent1,parent2,mutatedGene,child1,child2,mutatedChild])
    
    H11 = np.matmul(X,A[:,0:3].transpose())
    H22 = np.matmul(X,A[:,3:6].transpose())
    
    F11 = A[:,6] * H11 + A[:,7] * H22
    F22 = A[:,8] * H11 + A[:,9] * H22
    F33 = A[:,10] * H11 + A[:,11] * H22
    
    Y_Pred1 = F11 + F22 + F33
    
    fitness = CalculateFitness(Y,Y_Pred1)
    indices=np.argsort(fitness)
    
    W[parent1_index,:] = A[indices[0],:].copy()
    W[parent2_index,:] = A[indices[1],:].copy()
    W[mutatedGene_index,:] = A[indices[2],:].copy()
    
    Y_Pred[:,parent1_index] = Y_Pred1[:,indices[0]].copy()
    Y_Pred[:,parent2_index] = Y_Pred1[:,indices[1]].copy()
    Y_Pred[:,mutatedGene_index] = Y_Pred1[:,indices[2]].copy()

    if i %1000 == 0:
        print("Error of the fittest in iteration ",i," is :" ,squareRootError.min() ,"\n" )
        print("Weights of the fittest in iteration ",i," is :" , "\n", W[np.argmin(squareRootError),:] ,"\n","\n" )
        print("Predictions of chromosomes ",i," is :" , "\n", Y_Pred ,"\n","\n" )
        iterationArray.append(i)
        errorArray.append(squareRootError.reshape(200,1).transpose())
    
    if squareRootError.min() <= 0.05:
        print("Error of the fittest in iteration ",i," is :" ,squareRootError.min() ,"\n" )
        print("Weights of the fittest in iteration ",i," is :" , "\n", W[np.argmin(squareRootError),:] ,"\n","\n" )
        print("Predictions of chromosomes ",i," is :" , "\n", Y_Pred ,"\n","\n" )
        
        errorArray.append(squareRootError.reshape(1,200))
        iterationArray.append(i)
        break

plt.plot(np.array(errorArray).reshape(-1,200),'-.')
plt.xlabel("Iteration")
plt.ylabel("Error of the Fittest")


