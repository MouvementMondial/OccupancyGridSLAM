# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 11:57:46 2018

@author: Thorsten
"""

import numpy as np
from numba import jit

class Particle(object):
    def __init__(self, x, y, yaw, weight):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.weight = weight

@jit(nopython=True)
def scan2mapDistance(grid,pcl,offset,resolution):
    distance = 0;
    for i in range(pcl.shape[0]):
        # round points to cells
        xi = int ( (pcl[i,0]-offset[0]) / resolution )
        yi = int ( (pcl[i,1]-offset[1]) / resolution ) 
        distance += grid[xi,yi]
    return distance

def weightAndSortParticles(particles,grid,pcl,startPose,offset,resolution):
    for p in particles:
        # transform pointcloud to estimated pose
        # rotation
        R = np.matrix([[np.cos(p.yaw),-np.sin(p.yaw)],
                       [np.sin(p.yaw),np.cos(p.yaw)]])
        pclTransformed = pcl * np.transpose(R)
        # translation            
        pclTransformed = pclTransformed + np.matrix([p.x,p.y])
        # weight this solution
        p.weight = scan2mapDistance(grid,pclTransformed,
                                    startPose-offset,resolution)
    # sort particles by weight (best first)
    particles.sort(key = lambda Particle: Particle.weight,reverse=True)
    return particles

def fitScan2Map(grid,pcl,nrParticle1,nrParticleResample,nrParticle2,
                firstEstimate,stddPos,stddYaw,
                startPose,offset,resolution):
    
    # save the particles of the particle filter here
    particles = []
    
    """
    Particle filter with nrParticle1 Particles for first estimate
    """                
    # Create particles
    for _ in range(0,nrParticle1):
        particles.append(Particle(np.random.normal(firstEstimate[0,0],stddPos),
                                  np.random.normal(firstEstimate[0,1],stddPos),
                                  np.random.normal(firstEstimate[0,2],stddYaw),
                                  0.0))
    
    # Weight and sort particles (best first)                       
    particles = weightAndSortParticles(particles,grid,pcl,startPose,offset,resolution)    
    
    """
    Return best particle if there is no resampling
    """
    if nrParticleResample == 0:
        bestEstimateParticle = particles[0]
        return np.matrix([bestEstimateParticle.x,bestEstimateParticle.y,bestEstimateParticle.yaw])
   
    """
    Look in the neighborhood of the best particles for better solutions
    """
    # get only the best particles
    bestParticles = particles[:nrParticleResample]
    # delete the old particles
    particles.clear()
    
    # for each of the good particles create new particles
    for p in bestParticles:
        for _ in range(0,nrParticle2):
                particles.append(Particle(np.random.normal(p.x,stddPos/6.0),
                                          np.random.normal(p.y,stddPos/6.0),
                                          np.random.normal(p.yaw,stddYaw/3.0),
                                          0.0))
    
    # Weight and sort particles (best first)                           
    particles = weightAndSortParticles(particles,grid,pcl,startPose,offset,resolution) 
    
    """
    Return the best estimate
    """
    bestEstimateParticle = particles[0]
    return np.matrix([bestEstimateParticle.x,bestEstimateParticle.y,bestEstimateParticle.yaw])

def fitScan2Map2(grid,pcl,nrParticle1,nrParticleResample,
                nrParticle2,nrParticle3,
                firstEstimate,stddPos,stddYaw,
                startPose,offset,resolution):
    
    # save the particles of the particle filter here
    particles = []
    
    """
    Particle filter with nrParticle1 Particles for first estimate
    """                
    # Create particles
    for _ in range(0,nrParticle1):
        particles.append(Particle(np.random.normal(firstEstimate[0,0],stddPos),
                                  np.random.normal(firstEstimate[0,1],stddPos),
                                  np.random.normal(firstEstimate[0,2],stddYaw),
                                  0.0))
    
    # Weight and sort particles (best first)                       
    particles = weightAndSortParticles(particles,grid,pcl,startPose,offset,resolution)    
    
    """
    Return best particle if there is no resampling
    """
    if nrParticleResample == 0:
        bestEstimateParticle = particles[0]
        return np.matrix([bestEstimateParticle.x,bestEstimateParticle.y,bestEstimateParticle.yaw])
   
    """
    Look in the neighborhood of the best particles for better solutions
    """
    # get only the best particles
    bestParticles = particles[:nrParticleResample]
    # delete the old particles
    particles.clear()
    
    # for each of the good particles create new particles
    for p in bestParticles:
        for _ in range(0,nrParticle2):
                particles.append(Particle(np.random.normal(p.x,stddPos/6.0),
                                          np.random.normal(p.y,stddPos/6.0),
                                          np.random.normal(p.yaw,stddYaw/3.0),
                                          0.0))
    # Weight and sort particles (best first)                       
    particles = weightAndSortParticles(particles,grid,pcl,startPose,offset,resolution)                                          
                                          
    """
    Look in the neighborhood of the best particles for better solutions
    """
    # get only the best particles
    bestParticles = particles[:nrParticleResample]
    # delete the old particles
    particles.clear()
    
    # for each of the good particles create new particles
    for p in bestParticles:
        for _ in range(0,nrParticle3):
                particles.append(Particle(np.random.normal(p.x,stddPos/12.0),
                                          np.random.normal(p.y,stddPos/12.0),
                                          np.random.normal(p.yaw,stddYaw/6.0),
                                          0.0))
    
    # Weight and sort particles (best first)                           
    particles = weightAndSortParticles(particles,grid,pcl,startPose,offset,resolution) 
    
    """
    Return the best estimate
    """
    bestEstimateParticle = particles[0]
    return np.matrix([bestEstimateParticle.x,bestEstimateParticle.y,bestEstimateParticle.yaw])    