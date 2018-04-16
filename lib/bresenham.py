# -*- coding: utf-8 -*-
"""
Information about Bresenham algorithm:
https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm

@author: Thorsten
"""

from numba import jit
import numpy as np

@jit
def bresenham2D(startPoint, endPoint):
    """
    Returns array with cells (points) which are affected by a ray between the
    sensor and the measurement.    
    
    Thanks to Anton Fletcher:
    https://gist.github.com/salmonmoose/2760072

    Edited by balzer82
    https://github.com/balzer82/3D-OccupancyGrid-Python
    
    Adapted by Thorsten
    """        
    
    # Preallocate array for all points
    delta_x = abs(endPoint[0,0] - startPoint[0,0])
    delta_y = abs(endPoint[0,1] - startPoint[0,1])
    
    if delta_x > delta_y:        
        path = np.zeros((delta_x,2))
    else:
        path = np.zeros((delta_y,2))
      
    steepXY = (abs(endPoint[0,1] - startPoint[0,1]) > abs(endPoint[0,0] - startPoint[0,0]))
    if(steepXY):   
        startPoint[0,0], startPoint[0,1] = startPoint[0,1], startPoint[0,0]
        endPoint[0,0], endPoint[0,1] = endPoint[0,1], endPoint[0,0]
        
    delta = [abs(endPoint[0,0] - startPoint[0,0]), abs(endPoint[0,1] - startPoint[0,1])]

    errorXY = delta[0] / 2
    
    step = [
      -1 if startPoint[0,0] > endPoint[0,0] else 1,
      -1 if startPoint[0,1] > endPoint[0,1] else 1,
    ]
    
    y = startPoint[0,1]
    
    for x in range(startPoint[0,0], endPoint[0,0], step[0]):
        point = [x, y]

        if(steepXY):
            point[0], point[1] = point[1], point[0]

        errorXY -= delta[1]
        
        if(errorXY < 0):
            y += step[1]             
            errorXY += delta[0]
            
        path[x-startPoint[0,0],0] = point[0]
        path[x-startPoint[0,0],1] = point[1]
    
    return path