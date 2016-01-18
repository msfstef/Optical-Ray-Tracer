"""
A module containing generators for polar coordinate pairs.
"""

import numpy as np


def rtpairs(R,N):
    """
    R - list of radii
    N - list of points per radius
    
    Takes two list arguments containing the desired radii
    and the number of equally spread points per radii respectively.
    The generator, when iterated, will return radius-angle polar
    coordinate pairs, in metres and radians, which can be used 
    to plot shapes, e.g. a disc in the x-y plane. 
    """
    for i in range(len(R)):
        theta = 0.
        dTheta = 2*np.pi/N[i]
        for j in range(N[i]):
            theta = j*dTheta   
            yield R[i], theta
            

def rtuniform(n,rmax,m):
    """
    n - number of radii
    rmax - maximum radius
    m - scaling of points with radius
    
    This generator will return a disc of radius rmax, 
    with equally spread out points within it. The number 
    of points within the disc depends on the n and m parameters.
    """
    R = [0.]
    N = [1]
    rmax_f = float(rmax)    
    for i in range(int(n)):
        ri = (i+1)*(rmax_f/int(n))
        ni = int(m)*(i+1)
        R.append(ri)
        N.append(ni)
    return rtpairs(R,N)