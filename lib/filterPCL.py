# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 13:07:28 2017

@author: Thorsten
"""

import numpy as np
from numba import jit

@jit
def filterZ(pcl,zMin,zMax):
    binary = np.logical_and( pcl[:,2]>zMin, pcl[:,2]<zMax)  
    binary = np.column_stack((binary,binary,binary)) 
    pclFiltered = pcl[binary]
    return np.reshape(pclFiltered,(-1,3))