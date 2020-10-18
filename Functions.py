#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 21:42:15 2020

@author: philipwinchester
"""
import numpy as np
import pandas as pd

def NMod(Vector,n=1):
    # Takes vector and returns n*mod      
    return n*np.sqrt(np.inner(Vector, Vector))
    
def tau(x,y,lamb,mu,rho):
    # Defining tau function
    if x == 0 and y == 0:
        return 1 - (lamb*mu*rho)
    elif x == 0 and y == 1:
        return 1 + (lamb*rho)
    elif x == 1 and y == 0:
        return 1 + (mu*rho)
    elif x == 1 and y == 1:
        return 1 - rho
    else:
        return 1 
    
def phi(t,eps = 0.0025):
    # Define the weight function
    return np.exp(-eps*t)


def MatchLL(x,y,ai, aj, bi, bj, gamma, rho, t):
    # A function which calculates the log likelihood of some game
    lamb = ai*bj*gamma
    mu = aj*bi
    return phi(t)*(np.log(tau(x, y, lamb, mu, rho)) - lamb + x*np.log(lamb) - mu + y*np.log(mu))

def LL(Match_Data, Parameters, Teams):
  # Function which calculates the LL for all the games
  # This can also be made quicker if we avoid the for loop
  LL = 0 
  
  # Fixing gamma and rho, as these are constant for all games
  gamma = Parameters[2*len(Teams)]
  rho = Parameters[2*len(Teams)+1]
  
  for k in range(0,len(Match_Data.index)):
    # Finding index for the home and away team
    IndexHome = Teams.index(Match_Data['HomeTeam'][k])
    IndexAway = Teams.index(Match_Data['AwayTeam'][k])
    
    # Finding relevant Parameters and other variables
    ai = Parameters[IndexHome]
    aj = Parameters[IndexAway] 
    bi = Parameters[IndexHome + len(Teams)]
    bj = Parameters[IndexAway + len(Teams)] 
    t = Match_Data['t'][k]
    x =  Match_Data['FTHG'][k]
    y =  Match_Data['FTAG'][k]
    
    #Adding the LL from game k to the total
    LL = LL + MatchLL(x,y,ai, aj, bi, bj, gamma, rho, t)
  
  return LL
    
# Functions for alpha derivative are below

def GradAlphaHomeZeroZero(ai, aj, bi, bj, gamma, rho,t):
  lamb = ai*bj*gamma
  mu = aj*bi
  return phi(t)*bj*(-gamma-mu*gamma*rho/(1-lamb*mu*rho)) 

def GradAlphaHomeZeroOne(ai, bj, gamma, rho,t):
  lamb = ai*bj*gamma
  return phi(t)*bj*(-gamma+gamma*rho/(1+lamb*rho)) 

def GradAlphaHomeNotZero(ai, bj, gamma, x,t):
  return phi(t)*(x/ai-bj*gamma) 

def GradAlphaHome(ai, aj, bi, bj, gamma, rho,t,x,y):
  # Funtion which determines the addition to the gradient of the home attacking strenth from some game
  if x == 0 and y == 0:
    return GradAlphaHomeZeroZero(ai, aj, bi, bj, gamma, rho,t)
  elif x == 0 and y == 1:
    return GradAlphaHomeZeroOne(ai, bj, gamma, rho,t) 
  else:
    return GradAlphaHomeNotZero(ai, bj, gamma, x,t)

def GradAlphaAwayZeroZero(ai, aj, bi, bj, gamma, rho,t):
  lamb = ai*bj*gamma
  mu = aj*bi
  return phi(t)*bi*(-1-lamb*rho/(1-lamb*mu*rho))

def GradAlphaAwayOneZero(aj, bi, rho,t):
  mu = aj*bi
  return phi(t)*bi*(-1+rho/(1+mu*rho)) 


def GradAlphaAwayNotZero(aj, bi, y,t):
  return phi(t)*(y/aj-bi)

def GradAlphaAway(ai, aj, bi, bj, gamma, rho,t,x,y):
  # Funtion which determines the addition to the gradient of the away attacking strenth from some game
  if x == 0 and y == 0:
    return GradAlphaAwayZeroZero(ai, aj, bi, bj, gamma, rho,t)
  elif x == 1 and y == 0:
    return GradAlphaAwayOneZero(aj, bi, rho,t)
  else:
    return GradAlphaAwayNotZero(aj, bi, y,t) 

# Functions for beta derivative are below

def GradBetaHomeZeroZero(ai, aj, bi, bj, gamma, rho,t):
  lamb = ai*bj*gamma
  mu = aj*bi
  return phi(t)*aj*(-1-lamb*rho/(1-lamb*mu*rho)) 

def GradBetaHomeOneZero(aj, bi, rho,t):
  mu = aj*bi
  return phi(t)*aj*(-1+rho/(1+mu*rho))

def GradBetaHomeNotZero(aj, bi, y,t):
  return phi(t)*(y/bi-aj)

def GradBetaHome(ai, aj, bi, bj, gamma, rho,t,x,y):
  # Funtion which determines the addition to the gradient of the home defense strenth from some game
  if x == 0 and y == 0:
    return GradBetaHomeZeroZero(ai, aj, bi, bj, gamma, rho,t) 
  elif x == 1 and y == 0:
    return GradBetaHomeOneZero(aj, bi, rho,t)
  else:
    return GradBetaHomeNotZero(aj, bi, y,t)

def GradBetaAwayZeroZero(ai, aj, bi, bj, gamma, rho,t):
  lamb = ai*bj*gamma
  mu = aj*bi
  return phi(t)*ai*(-gamma-mu*gamma*rho/(1-lamb*mu*rho)) 


def GradBetaAwayZeroOne(ai, bj, gamma, rho,t):
  lamb = ai*bj*gamma
  return phi(t)*ai*(-gamma+rho*gamma/(1+lamb*rho))

def GradBetaAwayNotZero(ai, bj, gamma,x,t):
  return phi(t)*(x/bj-ai*gamma)

def GradBetaAway(ai, aj, bi, bj, gamma, rho,t,x,y):
  # Funtion which determines the addition to the gradient of the away defense strenth from some game
  if x == 0 and y == 0:
    return GradBetaAwayZeroZero(ai, aj, bi, bj, gamma, rho,t)
  elif x == 0 and y == 1:
    return GradBetaAwayZeroOne(ai, bj,gamma, rho,t)
  else:
    return GradBetaAwayNotZero(ai, bj, gamma, x,t)

# Functions for gamma derivative are below

def GradGammaZeroZero(ai, aj, bi, bj, gamma, rho,t):
  lamd = ai*bj*gamma
  mu = aj*bi
  return phi(t)*ai*bj*(-1-mu*rho/(1-lamd*mu*rho))

def GradGammaZeroOne(ai, bj, gamma, rho,t):
  lamd = ai*bj*gamma
  return phi(t)*ai*bj*(-1+rho/(1+lamd*rho))

def GradGammaNotZero(ai, bj, gamma, x,t):
  return phi(t)*(-ai*bj+x/gamma)

def GradGamma(ai, aj, bi, bj, gamma, rho,t,x,y):
  # Funtion which determines the addition to the gradient of the gamma param from some game
  if x == 0 and y == 0:
    return GradGammaZeroZero(ai, aj, bi, bj, gamma, rho,t)
  elif x == 0 and y == 1:
    return GradGammaZeroOne(ai, bj, gamma, rho,t)
  else:
    return GradGammaNotZero(ai, bj, gamma, x,t)
  
# Functions for rho derivative are below

def GradRhoZeroZero(ai, aj, bi, bj, gamma, rho,t):
  lamd = ai*bj*gamma
  mu = aj*bi
  return -phi(t)*lamd*mu/(1-lamd*mu*rho)

def GradRhoZeroOne(ai,bj, gamma, rho,t):
  lamd = ai*bj*gamma
  return phi(t)*lamd/(1+lamd*rho)

def GradRhoOneZero(aj,bi, rho,t):
  mu = aj*bi
  return phi(t)*mu/(1+mu*rho)

def GradRhoOneOne (rho,t):
  return -phi(t)/(1-rho)

def GradRho(ai, aj, bi, bj, gamma, rho,t,x,y):
  # Funtion which determines the addition to the gradient of the gamma param from some game
  if x == 0 and y == 0:
    return GradRhoZeroZero(ai, aj, bi, bj, gamma, rho,t)
  elif x == 0 and y == 1:
    return GradRhoZeroOne(ai,bj, gamma, rho,t)
  elif x == 1 and y == 0:
    return GradRhoOneZero(aj,bi, rho,t)
  elif x == 1 and y == 1:
    return GradRhoOneOne(rho,t) 
  else:
    return 0 

def GradAdder(Match_Data, Parameters, GradientVector,i, gamma, rho, Teams):
  # Function which takes the df of mathches, the current Parameters and calcualtes the addition to gradient vector for the i'th match
  # Returns the resulting gradient vector
  
  # Finding index for the home and away team
  IndexHome = Teams.index(Match_Data['HomeTeam'][i])
  IndexAway = Teams.index(Match_Data['AwayTeam'][i])
  
  # Finding relevant Parameters and other variables
  ai = Parameters[IndexHome]
  aj = Parameters[IndexAway] 
  bi = Parameters[IndexHome + len(Teams)]
  bj = Parameters[IndexAway + len(Teams)] 
  t = Match_Data['t'][i]
  x =  Match_Data['FTHG'][i]
  y =  Match_Data['FTAG'][i]
  
  # Adding onto the Gradient vector
  GradientVector[IndexHome] = GradientVector[IndexHome] + GradAlphaHome(ai, aj, bi, bj, gamma, rho,t,x,y)
  GradientVector[IndexAway] = GradientVector[IndexAway] + GradAlphaAway(ai, aj, bi, bj, gamma, rho,t,x,y)
  GradientVector[IndexHome + len(Teams)] = GradientVector[IndexHome + len(Teams)] + GradBetaHome(ai, aj, bi, bj, gamma, rho,t,x,y)
  GradientVector[IndexAway + len(Teams)] = GradientVector[IndexAway + len(Teams)] + GradBetaAway(ai, aj, bi, bj, gamma, rho,t,x,y)
  GradientVector[2*len(Teams)] = GradientVector[2*len(Teams)] + GradGamma(ai, aj, bi, bj, gamma, rho,t,x,y)
  GradientVector[2*len(Teams) + 1] = GradientVector[2*len(Teams) + 1] + GradRho(ai, aj, bi, bj, gamma, rho,t,x,y)
  
  return GradientVector

def GradientVectorFinder(Match_Data, Parameters, Teams):
  # Function whcih takes the match data, current Parameters and returns the Gradient Vector
  
  # Building the gradient vector 
  GradientVector = np.zeros(len(Teams)*2+2)
  
  # Setting gamma and rho
  gamma = Parameters[2*len(Teams)]
  rho = Parameters[2*len(Teams)+1]
  
  # Running through all the matches, every i makes an addition to the gradient vector
  for i in range(0,len(Match_Data.index)):
    GradientVector = GradAdder(Match_Data, Parameters, GradientVector,i, gamma, rho, Teams)
  
  return GradientVector

def NormalisingTheGradientVector(GradientVector,n, Teams):
  # Function which takes the GradientVector and normalises it such that the average of the alpha gradients is 0.

  AlphaGradValues = GradientVector[0:len(Teams)]
  AverageAlphaGradValues = np.mean(AlphaGradValues) # This is the average of paramaters in notes. But in our corrections, we want to add the gradint. Hence, there should be a net 0 efferct on the everage of the alphas from the gradint, as they already add up to one.
  Normaliser = np.concatenate((AverageAlphaGradValues*np.ones(len(Teams)), np.zeros(len(Teams)+2)))
  
  return (GradientVector - Normaliser)/NMod(GradientVector - Normaliser,n)

def Optimise(Match_Data, Teams,Max = 200, m = 10):
  # Takes some match data and returns returns the parameters which maximise the log liklihood function.
  # This is done with a gradient ascent alogorithm 
  # The default maximum step size is is 1/200, can be changed in the Max variable
  # The default is that we start with a step size of 1/10, which then goes to 1/20 etc... this can be changed in m
  
  # Setting all Parameters equal to 1 at first
  Parameters = np.ones(2*len(Teams)+2)
  
  # Setting gamma equal to 1.3 and rho equal to -0.05
  Parameters[2*len(Teams)] = 1.3
  Parameters[2*len(Teams)+1] = -0.05

  Mult = 1
  Step = m
  
  count = 0
  # Doing itertaitons until we have added just one of the smallets gradient vecor we want to add
  while Step <= Max:
    
    count = count + 1
    print("count is " + str(count))
    
    # Finding gradient 
    GradientVector = GradientVectorFinder(Match_Data, Parameters, Teams)
    
    # Normalising (Avergage of alhpas is 1), and adjusting the length
    GradientVectorNormalised = NormalisingTheGradientVector(GradientVector,Step, Teams)
    print("step is " + str(Step))
    
    PresentPoint = Parameters
    StepToPoint = Parameters + GradientVectorNormalised
    LLLoop = 0
    
    # Adding GradientVectorNormalised until we have maxemised the LL
    while LL(Match_Data, StepToPoint, Teams) > LL(Match_Data, PresentPoint, Teams):
      PresentPoint = StepToPoint
      StepToPoint = PresentPoint + GradientVectorNormalised
      LLLoop = LLLoop + 1 
    
    print("LLLoop is " + str(LLLoop))
    
    # If there has only been one itteration (or zero), we increase the step size
    if LLLoop < 2:
      Mult = Mult + 1
      Step = Mult*m
    
    Parameters = PresentPoint
  
  Alpha = Parameters[0:len(Teams)]
  Beta = Parameters[len(Teams):(len(Teams)*2)]
  Gamma = Parameters[len(Teams)*2]
  Rho = Parameters[len(Teams)*2+1]
  d = {'Team': Teams, 'Alpha': Alpha, 'Beta': Beta, 'Gamma': Gamma*np.ones(len(Teams)), 'Rho': Rho*np.ones(len(Teams))} 
  Results = pd.DataFrame(data=d)  
  
  return Results


