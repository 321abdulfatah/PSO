# This file contains functions to implement the FastICA algorithm.

import numpy as np
from numpy.matlib import rand, eye
import math

# Function description: centers data.
# Inputs:
#   X = (un-centered) data
#       size: (num_sig, num_samples)
# Outputs:
#   X = centered data
#       size: (num_sig, num_samples)
def center(X):
    # center data:
    X = X - np.mean(X, axis=1, keepdims=True)

    return X

def whitening(X):
    cov = np.cov(X)
    #cov=np.round(cov,4)
    d, E = np.linalg.eigh(cov)
    D = np.diag(d)
    D_inv = np.sqrt(np.linalg.inv(D))
    Q=np.dot(D_inv,E)
    X_whiten = np.dot(Q,X)
    return X_whiten

def pso(X1,le):
    
    np.random.seed(1)
    # Center signals
 
    X1 = center(X1)

    # Whiten mixed signals
    X1 = whitening(X1) 
    
    alpha=0.75
    
    k=2
    
    x2=eye(k)
    
    population=10
    
    W = np.zeros((population, k, k))
    
    for i in range (0,population):
        W[i]=x2  
    
    SS=np.zeros((population,k,le))
 
   
    negentropy=[]
    fitness=[]
    fitnessnew=[]
    fitnessmax=[]
    
    for i in range(0,population):
        Y=W[i,:,:].dot(X1)
    
        # Center signals
        Y = center(Y)  
        # Whiten mixed signals
        Y = whitening(Y) 
    
        for K in range(0,2):
            kurtosis = np.mean(Y[K, :] ** 4) - 3 * (np.mean(Y[K, :] ** 2)) ** 2                      # equation ok
            negentropy.append(1 / 12 * ((np.mean(Y[K, :]) ** 3) ** 2) + 1 / 48 * ((kurtosis) ** 2))  # equation ok
         
        fitness.append(np.sum(np.abs(negentropy)))
        negentropy=[]
    c,b=max(fitness),np.argmax(fitness)
    
    pgmax=W[b]
    pimax=W
    fitnessmax=fitness
    fgmax=max(fitness)
    
    maxiter=20
    
    for _ in range(maxiter):
        
        mbest=np.sum(fitness)/population
        fitnessnew=[]
        
        for n in range (0,population):
            for i in range(0,k):
                for j in range(0,k):
                    fi=0.2937
                    p=fi*pimax[n,i, j]+(1-fi)*pgmax[i, j]
                    u=0.4306
                    W[n,i,j]=p+((-1)**math.ceil(0.5+rand(1)))*(alpha*np.abs(mbest-W[n,i,j])*math.log(1/u))

        for m in range(0,population):
            SS[m,:,:]=W[m,:,:].dot(X1)
        
            # Center signals
            SS[m,:,:] = center(SS[m,:,:])   
        
            # Whiten mixed signals
            SS[m,:,:]= whitening(SS[m,:,:]) 
        
            for k in range(0, 2):
                kurtosis = np.mean(SS[m,k, :] ** 4) - 3 * (np.mean(SS[m,k, :] ** 2)) ** 2                      # equation ok
                negentropy.append(1 / 12 * ((np.mean(SS[m,k, :]) ** 3) ** 2) + 1 / 48 * ((kurtosis) ** 2))   # equation ok
        
            fitnessnew.append(np.sum(np.abs(negentropy)))
            negentropy=[]
        
        for i in range (0,population):

            if fitnessnew[i] > fitnessmax[i]:
                fitnessmax[i]=fitnessnew[i]
                pimax[i,:,:]=W[i,:,:]

            if fitnessnew[i] > fgmax:
                pgmax = W[i,:,:]
                fgmax = fitnessnew[i]

        g, v2 = max(fitnessnew),np.argmax(fitnessnew)
    
        if g >= c:
            c = g
            v3=v2
            S_best = SS[v3,:,:]
    
    return S_best                 