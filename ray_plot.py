"""
A module containing utility functions for plotting results yielded by the
related ray tracing modules.
"""
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


def plotRayTrace_ray(ray,colour):
    """
    colour - string argument, determines colour of ray in plot.
    
    Plots the trajectory of a ray in a 2D x-z plot.
    """
    plt.figure("Ray 2D")
    x = []
    z = []
    for point in ray.vertices():
        x.append(point[0])
        z.append(point[2])
    plt.xlabel("Z axis / $mm$")
    plt.ylabel("X axis / $mm$")
    plt.plot(z, x, c=colour)
    plt.axes().set_aspect('equal', 'datalim')
    plt.show()
    
def plotRayTrace_beam(beam):
    """
    Plots the trajectory of a beam in a 2D x-z plot.
    """
    plt.figure("Beam 2D")
    for ray in beam._rays:
        x = []
        z = []
        for point in ray.vertices():
            x.append(point[0])
            z.append(point[2])
        plt.plot(z, x, 'r-')
    plt.xlabel("Z axis / $mm$")
    plt.ylabel("X axis / $mm$")
    plt.axes().set_aspect('equal', 'datalim')
    plt.show()
    
def plotRayTrace_beam_3D(beam, colour = 'r'):
    """
    Plots the trajectory of a ray in a 3D plot.
    """
    fig3D = plt.figure("Beam 3D")
    ax = fig3D.add_subplot(111, projection='3d')
    ax.set_xlabel('X axis / $mm$')
    ax.set_ylabel('Y axis / $mm$')
    ax.set_zlabel('Z axis / $mm$')
    for ray in beam._rays:
        vert = ray.vertices()
        for i in range(len(vert)-1):
            ax.plot([vert[i][0],vert[i+1][0]],[vert[i][1],vert[i+1][1]],zs=[vert[i][2],vert[i+1][2]], c=colour)
            
    max_range = 200
    Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten()
    Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten()
    Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten()
    for xb, yb, zb in zip(Xb, Yb, Zb):
        plt.plot([xb], [yb], [zb], 'w')
    plt.show()
    
def plotOutputPlane_beam(beam):
    """
    Plots a beam on the output plane as a 2D x-y plot.
    """
    plt.figure("Output Plane")
    x = []
    y = []
    for ray in beam._rays:
        if ray._reachedOutputPlane is True:
            point = ray.p()
            x.append(point[0])
            y.append(point[1])
    plt.scatter(x, y, c = "r")
    plt.xlabel("X axis / $mm$")
    plt.ylabel("Y axis / $mm$")
    plt.title("Focus Point")
    plt.axes().set_aspect('equal', 'datalim')
    plt.show()
    
    
def plotDroplet2D(z_c, x_c, R):
    """
    Plots a cyan circle to represent a water droplet.
    """
    droplet = plt.Circle((float(z_c), float(x_c)), float(R), 
                            color = '#87CEFA', fill=True)
    plt.gca().add_artist(droplet)
    
def plotDroplet3D(z_c, R):
    """
    Returns coordinates of 3D sphere with its centre at (x_c, y_c, z_c),
    with a radius R. It represents a 3D water droplet.
    """
    x_c, y_c = 0, 0
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x=np.cos(u)*np.sin(v)
    y=np.sin(u)*np.sin(v)
    z=np.cos(v)
    x = R*x + x_c
    y = R*y + y_c
    z = R*z + z_c
    plt.figure('Beam 3D')
    plt.gca(projection='3d').plot_wireframe(x, y, z, color="#87CEFA")
    plt.show()