"""
A module that allows the user to create optical rays using the Ray class.
It is meant to be used in conjunction with the optical.py module that
contains all the optical instruments the a ray can be propagated through.
"""

import numpy as np
import genpolar as gen
import ray_plot as rtplot


class Ray:
    """
    The Ray class models an optical ray that can be propagated
    through various optical elements and thus be used to optimise
    optical setups and model physical situations related to optics.
    
    It can be initiated with two arguments, the initial position and
    direction vector of the ray, given as python lists (e.g. [2,1,4]).
    
    If testing dispersive optical elements, an additional argument can
    be given in the initiation to determine the ray's wavelength. The
    wavelength should be given in units of nanometers. It is set by
    default to None to avoid redundant refractive index calculations.
    """
    def __repr__(self):
        return ("%s(position=[%g, %g, %g], direction=[%g, %g, %g])" % 
                ("Ray", self.p()[0], self.p()[1], self.p()[2], 
                        self.k()[0], self.k()[1], self.k()[2]))
    
    def __str__(self):
        return ("R(p=[%g, %g, %g], k=[%g, %g, %g], lambda=%g)" % 
                (self.p()[0], self.p()[1], self.p()[2],
                 self.k()[0], self.k()[1], self.k()[2], self.wavelength))  

    def __init__(self, p_init = [0,0,0], k_init = [0,0,1], wavelength = None):
        if np.array_equal(np.array(k_init), np.array([0,0,0])) is True:
            raise Exception("Ray must be initialised with some direction")
        if len(p_init) != 3 or len(k_init) != 3:
            raise Exception("Ray must be initialised in 3 dimensions")
        self._step_num = 0
        self._p = []
        self._k = []
        self.wavelength = wavelength
        self._reachedOutputPlane = False
        
        self._p.append(np.array(p_init))
        k_init_norm = np.array(k_init)/np.linalg.norm(np.array(k_init))
        self._k.append(k_init_norm)
        
    def p(self):
        """
        Returns current position of the ray object as a numpy array.
        """
        return self._p[self._step_num]
        
    def k(self):
        """
        Returns current normalised direction vector of 
        the ray object as a numpy array.
        """
        return self._k[self._step_num]
        
    def append(self, p, k):
        """
        Takes a point p and a direction k as arguments,
        where the point and directions should be given in a python 
        list or numpy array format, and appends them to the end of
        the position and direction lists of the ray object.
        The append method is equivalent to the ray moving from
        one point to another.
        """
        self._p.append(np.array(p))
        self._k.append(np.array(k))
        self._step_num += 1
    
    def vertices(self):
        """
        Returns the list of points, or positions, the
        ray object has gone through.
        """
        return self._p
        
    def propagate(self, optical_elements = [], plot = False, colour = 'r'):
        """
        Propagates the ray through an array of optical elements, given
        as an argument in list format.
        A plot argument, set to false by default, can be given to
        show a 2D plot of the ray trajectory.
        """
        for opt_elem in optical_elements:
            opt_elem.propagate_ray(self)
        if plot is True:
            rtplot.plotRayTrace_ray(self, colour)
            
class CollimatedBeam(Ray):
    """
    p_init - list argument, initial position.
    k_init - list argument, initial direction.
    r_spacing - integer argument, determines tha radial density of rays.
    r_max - float argument, determines maximum radius of beam.
    phi_density - integer argument, determines angular density of rays.
    wavelength - float argument, determines wavelength of component rays,
    must be given in units of nanometers.
    
    The CollimatedBeam class models a beam composed of uniformly spread
    parallel rays, described by Ray objects, that can be propagated
    through various optical elements and thus be used to optimise
    optical setups and model physical situations related to optics.
    
    It can be initiated with 6 arguments, the initial position and
    direction vector of the ray, given as python lists (e.g. [2,1,4]),
    and some parameters that define the diameter and density of rays
    of the collimated beam.
    
    If testing dispersive optical elements, an additional argument can
    be given in the initiation to determine the ray's wavelength. The
    wavelength should be given in units of nanometers. It is set by
    default to None to avoid redundant refractive index calculations.
    """
    def __repr__(self):
        return ("%s(central position=[%g, %g, %g], central direction=[%g, %g, %g])" % 
                ("CollimatedBeam", self.p()[0], self.p()[1], self.p()[2], 
                                   self.k()[0], self.k()[1], self.k()[2]))
    
    def __str__(self):
        return ("B(p_c=[%g, %g, %g], k_c=[%g, %g, %g]), lambda = %g" % 
                (self.p()[0], self.p()[1], self.p()[2], 
                 self.k()[0], self.k()[1], self.k()[2], self._rays[0].wavelength))
  
    def __init__(self, p_init = [0,0,0], k_init = [0,0,1], r_spacing = 5, 
                    r_max = 5., phi_density = 2, wavelength = None):
        if np.array_equal(np.array(k_init), np.array([0,0,0])) is True:
            raise Exception("Beam must be initialised with some direction")
        if len(p_init) != 3 or len(k_init) != 3:
            raise Exception("Beam must be initialised in 3 dimensions")
        self._step_num = 0
        self._rays = []
        self._p = []
        self._k = []
        k_init_norm = np.array(k_init)/np.linalg.norm(np.array(k_init))

        for r, phi in gen.rtuniform(r_spacing, r_max, phi_density):
            x = r*np.cos(phi) + p_init[0]
            y = r*np.sin(phi) + p_init[1]
            self._rays.append(Ray([x,y,p_init[2]],k_init_norm, wavelength))
            
        self._p.append(np.array(p_init))        
        self._k.append(k_init_norm)
    
    def p_rays(self):
        """
        Returns current position of the component rays as a numpy array.
        """
        p_group = []
        for ray in self._rays:
            p_group.append(ray.p())
        return p_group
        
    def k_rays(self):
        """
        Returns current normalised direction vector of 
        the component rays as a numpy array.
        """
        k_group = []
        for ray in self._rays:
            k_group.append(ray.k())
        return k_group
        
        
    def propagate(self, optical_elements = [], plot2D = False,
                    plot3D = False, plotOutputPlane = False, colour = 'r'):
        """
        Propagates the beam through an array of optical elements, given
        as an argument in list format.
        Three plot arguments, set to false by default, can be set to
        True in order to plot the desired figure:        
        plot2D - plots a 2D trajectory of the beam.
        plot3D - plots a 3D trajectory of the beam.
        plotOutputPlane - Plots the projection of the beam on the output
        plane as a 2D plot of the x-y plane at the z position of the
        output plane.
        """
        for opt_elem in optical_elements:
            for ray in self._rays:
                opt_elem.propagate_ray(ray)
            self.append(self._rays[0].p(), self._rays[0].k())
        if plot2D is True:
            rtplot.plotRayTrace_beam(self)
        if plot3D is True:
            rtplot.plotRayTrace_beam_3D(self, colour)
        if plotOutputPlane is True:
            rtplot.plotOutputPlane_beam(self)
        