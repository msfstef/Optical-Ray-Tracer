"""
A module containing all optical instruments that a ray can be propagated through.
"""

import numpy as np


class OpticalElement:
    """
    This class is the parent of all optical instruments, but is not useful
    in itself.
    """
    def propagate_ray(self, ray):
        "Propagate a ray through the optical element"
        raise NotImplementedError()
    
    def intercept(self, ray):
        """
        Takes a Ray object as an argument and returns the point
        of intersection of the ray with the optical element
        as a numpy array, if it exists. If the ray does not 
        intersect the optical element it returns None.
        """
        r = ray.p() - self._O
        rk = np.inner(r,ray.k())
        r_magsquared = np.inner(r,r)
        
        #Checks all possible cases to get correct intercept.
        #See lab book for details on the approach used.
        if self._C != 0:
            #Checks for existence of intercepts.
            if rk*rk >= r_magsquared - self._Rc*self._Rc:
                determinant_root = np.sqrt(rk*rk - (r_magsquared - self._Rc*self._Rc))         
                length_1 = -rk - determinant_root
                length_2 = -rk + determinant_root
                #Checks if ray initial point is inside or outside of lens 'sphere'.
                if r_magsquared >= (1. + 1e-5)*self._Rc*self._Rc:
                    #Checks if direction allows intercept.
                    if length_1 < 0 and length_2 < 0:
                        return None
                        
                    if self._C == 'sphere':
                        length = min(length_1, length_2)
                    else:
                        if self._z_0 > ray.p()[2]:
                            if self._C > 0:
                                length = min(length_1, length_2)
                            else:
                                length = max(length_1, length_2)
                        else:
                            if self._C > 0:
                                length = max(length_1, length_2)
                            else:
                                length = min(length_1, length_2)
                else:
                    length = max(length_1, length_2)
            else:
                return None
        else:
            k_z = ray.k()[2]
            delta_z = ray.p()[2] - self._z_0
            if (delta_z < 0 and k_z > 0) or (delta_z > 0 and k_z < 0):
                length = np.absolute(r[2]/k_z)
            else:
                return None
        #Checks if intercept is actually ray's initial point.        
        if -1e-6 < length < 1e-6:
            return None
    
        intercept = ray.p() + length*ray.k()
        intercept_dist_from_opt_axis = np.sqrt(intercept[0]*intercept[0] + 
                                                intercept[1]*intercept[1])
        #Checks if intercept lies within the aperture of the lens.
        if intercept_dist_from_opt_axis > self._Ra:
            return None
        if self._C != 0 and self._C != 'sphere':
            #Checks if intercept is on correct side of lens 'sphere'.
            if ((intercept[2] > self._z_0 + self._Rc)  or 
                (intercept[2] < self._z_0 - self._Rc)):
                return None
        return intercept


class SphericalRefraction(OpticalElement):
    """
    A SphericalRefraction object can be initiated with an optical axis
    intercept z0, a curvature C, defined to be 1/radius of curvature, 
    an aperture radius Ra, and the refractive indexes on either 
    side of the optical element n1 and n2.
    """
    def __repr__(self):
        return (("%s(z-axis intercept=%g, curvature=%g, aperture radius=%g, "
                "refractive indices n1,n2=%g,%g)") %
                ("SphericalRefraction", self._z_0, self._C, 
                self._Ra, self._n_1, self._n_2))
    
    def __str__(self):
        return ("%s(z0=%g, C=%g, Ra=%g, n1=%g, n2=%g)" % 
                ("SphericalRefraction", self._z_0, self._C, 
                self._Ra, self._n_1, self._n_2))
 
    
    def __init__(self, z_0=0, C=0, Ra=1, n_1=1, n_2=1):
        self._z_0 = float(z_0)
        self._C = float(C)
        if C!=0:
            self._Rc = np.absolute(1/C)
            if Ra >= np.absolute(self._Rc):
                raise Exception("Aperture must be smaller than radius of curvature")
                
            if C > 0:
                self._O = np.array([0.,0.,z_0+self._Rc])
            else:
                self._O = np.array([0.,0.,z_0-self._Rc])
        else:
            self._O = np.array([0.,0.,z_0])
        self._n_1 = float(n_1)
        self._n_2 = float(n_2)
        self._Ra = float(Ra)
    
    def refractive_index(self, ray):
        """
        Takes a Ray object as an argument and determines the refractive index
        of the optical element using the ray's wavelength. This is caused
        by dispersion, and its scale is different for every dispersive
        material. In this case only water is considered, to be able to
        model rainbow formation due to rain droplet refraction.
        
        Temperature is assumed to be 20 degrees Celsius and the density
        of the water is assumed to be 1 kg/m^3.
        
        Source: http://www.iapws.org/relguide/rindex.pdf 
        """  
        lambda_bar = ray.wavelength/589.
        lambda_bar_ir_squared = 29.51680445
        lambda_bar_uv_squared = 0.0525335568
        expression = (0.2333358238 + 0.0002883510674*lambda_bar*lambda_bar +
                    (0.0015892057/lambda_bar) +
                    (0.00245934259/(lambda_bar*lambda_bar - lambda_bar_uv_squared)) +
                    (0.90070492/(lambda_bar*lambda_bar - lambda_bar_ir_squared)))
        refractive_index = np.sqrt((2*expression +1)/(1-expression))
        return refractive_index
        
    def refract(self, ray, intercept):
        """
        Takes a Ray object and its intercept with the optical element
        as arguments and returns the new normalised direction vector of 
        the refracted ray. 
        
        If the ray is totally internally reflected, it returns None.
        
        If the ray does not intersect the optical element, it returns
        the original direction of the vector.
        """
        if intercept is None:
            k_refracted = ray.k()
            return k_refracted
        if ray.wavelength is not None:
            if self._n_2 != 1.:
                self._n_2 = self.refractive_index(ray)
            else:
                self._n_1 = self.refractive_index(ray)
        
        r = self._n_1/self._n_2
        if self._C != 0:
            n_unnorm = intercept - self._O
            n = n_unnorm/(np.linalg.norm(n_unnorm))
            cos_dot_product = - np.inner(ray.k(),n)
            if cos_dot_product < 0:
                n = -n
        else:
            n = np.array([0,0,1.])
            cos_dot_product = - np.inner(ray.k(),n)
            if cos_dot_product < 0:
                n = -n 
        c = - np.inner(n, ray.k())
        if r*r*(1 - c*c) < 1.:
            k_refracted_unnorm = r*ray.k() + (r*c - np.sqrt(1 - r*r*(1 - c*c)))*n
            k_refracted = k_refracted_unnorm/(np.linalg.norm(k_refracted_unnorm))     
        else:
            #Total internal reflection.
            k_refracted = None
        return k_refracted
        
        
    def propagate_ray(self, ray):
        """
        Takes a Ray object as an argument and appends
        to it a new position and direction depending
        on how the ray was refracted.
        
        If the ray does not intersect with the optical element, 
        it is left unchanged and a message is returned.
        
        If the ray does intersect, but is totally internally
        reflected, it is terminated (e.g. direction vector
        is set to a zero vector (numpy array with 0 components).
        """
        if np.array_equal(ray.k(),np.array([0,0,0])):
            return "Ray is terminated."
            
        intcpt = self.intercept(ray)
        k_ref = self.refract(ray, intcpt)
        if (intcpt is not None) and (k_ref is not None):
            ray.append(intcpt,k_ref)
        elif (intcpt is not None) and (k_ref is None):
            ray.append(intcpt,np.array([0,0,0]))
            return "Ray undergoes total internal reflection."
        else:
            return "Ray does not intersect optical element."


class SphericalReflection(OpticalElement):
    """
    A SphericalReflection object can be initiated with an optical axis
    intercept z0, a curvature C, defined to be 1/radius of curvature, 
    and an aperture radius Ra. It is a mirror element that reflects
    rays.
    """
    def __repr__(self):
        return ("%s(z-axis intercept=%g, curvature=%g, aperture radius=%g)" %
                ("SphericalReflection", self._z_0, self._C, self._Ra))
    
    def __str__(self):
        return ("%s(z0=%g, C=%g, Ra=%g)" % 
                ("SphericalReflection", self._z_0, self._C, self._Ra,))
 
    def __init__(self, z_0=0, C=0, Ra=1):
        self._z_0 = float(z_0)
        self._C = float(C)
        if C!=0:
            self._Rc = np.absolute(1/C)
            if Ra >= np.absolute(self._Rc):
                raise Exception("Aperture must be smaller than radius of curvature")
                
            if C > 0:
                self._O = np.array([0.,0.,z_0+self._Rc])
            else:
                self._O = np.array([0.,0.,z_0-self._Rc])
        else:
            self._O = np.array([0.,0.,z_0])
        self._Ra = float(Ra)
        
    def reflect(self, ray, intercept):
        """
        Takes a Ray object and its intercept with the optical element
        as arguments and returns the new normalised direction vector of 
        the reflected ray.
        
        If the ray does not intersect the optical element, it returns
        the original direction of the vector.
        """
        if intercept is None:
            k_refracted = ray.k()
            return k_refracted
        
        if self._C != 0:
            n_unnorm = intercept - self._O
            n = n_unnorm/(np.linalg.norm(n_unnorm))
            cos_dot_product = - np.inner(n, ray.k())
            if cos_dot_product < 0:
                n = -n
        else:
            n = np.array([0,0,1.])
            cos_dot_product = - np.inner(n, ray.k())
            if cos_dot_product < 0:
                n = -n
            
        #Vector form of reflection.
        n_dot_k = np.inner(n, ray.k())
        k_reflected_unnorm = ray.k() - 2*n_dot_k*n
        k_reflected = k_reflected_unnorm/(np.linalg.norm(k_reflected_unnorm))
        return k_reflected
        
    def propagate_ray(self, ray):
        """
        Takes a Ray object as an argument and appends
        to it a new position and direction depending
        on how the ray was reflected.
        
        If the ray does not intersect with the optical element, 
        it is left unchanged and a message is returned.
        """
        if np.array_equal(ray.k(),np.array([0,0,0])):
            return "Ray is terminated."
            
        intcpt = self.intercept(ray)
        k_ref = self.reflect(ray, intcpt)
        if (intcpt is not None) and (k_ref is not None):
            ray.append(intcpt,k_ref)
        else:
            return "Ray does not intersect optical element."


class OutputPlane(OpticalElement):
    """
    An OutputPlane object can be initiated with an optical axis
    intercept z0, and is assumed to cover the whole x-y plane.
    It is used as the end point of rays and beams for purposes
    of analysis and visualisation.
    """
    
    def __repr__(self):
        return "%s(z-axis intercept=%g)" % ("OutputPlane", self._z_0)
    
    def __str__(self):
        return "%s(z0=%g)" % ("OutputPlane", self._z_0)
                
    def __init__(self, z_0):
        self._z_0 = float(z_0)
        self._O = np.array([0,0,self._z_0])
        self._Ra = float('inf')
        self._C = 0
        
    def propagate_ray(self, ray):
        """
        Takes a Ray object as an argument and appends
        to it a new position where it intersects with
        the output plane, and terminates it by setting
        its direction vector to a zero vector.
        
        If the ray does not intersect with the output plane, 
        it is left unchanged and a message is returned.
        """
        intcpt = self.intercept(ray)
        if intcpt is not None:
            ray.append(intcpt,np.array([0,0,0]))
            ray._reachedOutputPlane = True
        else:
            return "Ray does not intersect output plane."

class Droplet(SphericalRefraction, SphericalReflection):
    def __init__(self, centre = [0, 0, 100], radius = 50.):
        self._O = np.array([float(point) for point in centre])
        self._Ra = float('inf')
        self._Rc = float(radius)
        self._C = 'sphere'
        self._n_1 = 1.
        self._n_2 = 1.
    
    def reflectance(self, ray, normal_to_surface, n_list):
        """
        Takes a Ray, the normal vector to the surface at the intercept, and
        a list with the two refractive indices as arguments, and calculates
        the fraction of incident light intensity that gets reflected at the
        surface boundary. This fraction, called reflectance, is calculated 
        using Fresnel's equations.
        """
        n_1 = n_list[0]
        n_2 = n_list[1]
        r = n_1/n_2
        dot_product = np.inner(ray.k(),normal_to_surface)
        cos_theta_1 = - dot_product
        cos_theta_2 = np.sqrt(1 - (r*np.sqrt(1 - dot_product*dot_product))*
                                    (r*np.sqrt(1 - dot_product*dot_product)))
        reflectance_s = np.absolute((n_1*cos_theta_1 - n_2*cos_theta_2)/
                                    (n_1*cos_theta_1 + n_2*cos_theta_2))
        reflectance_p = np.absolute((n_1*cos_theta_2 - n_2*cos_theta_1)/
                                    (n_1*cos_theta_2 + n_2*cos_theta_1))
        reflectance = (reflectance_s*reflectance_s + 
                        reflectance_p*reflectance_p)/2.
        return reflectance        
        
    def refract(self, ray, intercept):
        """
        Takes a Ray object and its intercept with the optical element
        as arguments and returns the new normalised direction vector of 
        the refracted ray. 
        
        If the ray is totally internally reflected, it returns None.
        
        If the ray does not intersect the optical element, it returns
        the original direction of the vector.
        """
        if intercept is None:
            k_refracted = ray.k()
            return k_refracted
        if ray.wavelength is not None:
            self._n_2 = self.refractive_index(ray)
            
        n_unnorm = intercept - self._O
        n = n_unnorm/(np.linalg.norm(n_unnorm))
        cos_dot_product = - np.inner(ray.k(),n) 
        r = self._n_1/self._n_2
        n_list = [self._n_1, self._n_2]               
        if cos_dot_product < 0:
            n = -n
            r = self._n_2/self._n_1
            n_list = [self._n_2, self._n_1]

        reflectance = self.reflectance(ray, n, n_list)
        random_counter = np.random.uniform(0, 1)
        
        if reflectance > random_counter:
            #Gets reflected, Fresnel equations.
            k_refracted = self.reflect(ray, intercept)
            return k_refracted
            
        c = - np.inner(n, ray.k())
        if r*r*(1 - c*c) < 1. + 1e-8:
            k_refracted_unnorm = r*ray.k() + (r*c - np.sqrt(1 - r*r*(1 - c*c)))*n
            k_refracted = k_refracted_unnorm/(np.linalg.norm(k_refracted_unnorm))                   
        else:
            #Total internal reflection.
            k_refracted = self.reflect(ray, intercept)
        return k_refracted
        
    def propagate_ray(self, ray):
        """
        Takes a Ray object as an argument and appends
        to it a new position and direction depending
        on how the ray was reflected.
        
        If the ray does not intersect with the optical element, 
        it is left unchanged and a message is returned.
        """
        if np.array_equal(ray.k(),np.array([0,0,0])):
            return "Ray is terminated."
            
        intcpt = self.intercept(ray)
        k_ref = self.refract(ray, intcpt)
        if (intcpt is not None) and (k_ref is not None):
            ray.append(intcpt,k_ref)
        else:
            observer_front = OutputPlane(0.)
            observer_back = OutputPlane(2*self._O[2])
            observer_front.propagate_ray(ray)
            observer_back.propagate_ray(ray)
            return "Ray does not intersect optical element."