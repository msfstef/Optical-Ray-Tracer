"""
A module containing utility functions for the analysis of results generated 
using the related ray tracing modules such as raytracer.py and optical.py.
"""

from __future__ import print_function
import numpy as np
import optical as opt
import raytracer as rt
import scipy.optimize as scopt

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt


def find_paraxial_focal_point(lens = [], optical_axis_deviation = 0.01):
    """
    lens - list argument, contains optical elements that make up the lens
    in the order in which they are placed on the optical axis.
    
    optical_axis_deviation - float argument, determines the deviation of the
    ray used to find the ideal focal point from the optical axis. Essentially
    determines the accuracy desired for the position of the focal point. The
    smaller the deviation, the more accurate the position.
    
    Tests the given lens setup with a ray close to the optical axis to estimate
    the position of the focal point, which is then returned as a distance on the
    optical axis.
    """
    ray_near_optical_axis = rt.Ray([optical_axis_deviation, 0., 0.], [0., 0., 1.])
    ray_near_optical_axis.propagate(lens)
    k_refracted = ray_near_optical_axis.k()
    k_dot_x_axis = np.inner(k_refracted,np.array([-1.,0.,0.]))
    focal_length = ray_near_optical_axis._p[-1][0]*(
                    np.sqrt(1 - k_dot_x_axis*k_dot_x_axis)/k_dot_x_axis)
    focus_point_z = lens[-1]._z_0 + focal_length
    return focus_point_z
    
def rms_xy_spread(lens = [], focal_length = 100., beam = None, plot = False):
    """
    lens - list argument, contains optical elements that make up the lens
    in the order in which they are placed on the optical axis.
    
    beam - CollimatedBeam argument, has to be a beam object created using
    the raytracer module.
    
    plot - Boolean argument, determines whether a 2D graph tracing the beam's
    trajectory will be plotted. Set to False by default.
    
    Calculates the RMS spread of the output in the x-y plane at the focal point 
    of the lens for a given beam. It is a measure of how well the lens focuses
    the beam onto a point, i.e. how well it minimises spherical abberations.
    """
    middle_of_lens = (lens[0]._z_0 + lens[1]._z_0)/2.
    output_plane = opt.OutputPlane(middle_of_lens + focal_length)
    optical_setup = lens + [output_plane]
    beam.propagate(optical_setup, plot)
    
    central_deviations = []
    for point in beam.p_rays():
        deviation_squared = point[0]*point[0] + point[1]*point[1]
        central_deviations.append(deviation_squared)
    RMS_spread = np.sqrt(np.mean(central_deviations))
    return RMS_spread
    
def curvature_optimisation_func(curvatures = [0.01, -0.002], focal_length = 100.,
                                diameter = 5., thickness = 5., ref_index = 1.5):
    """
    This function is used to minimise the RMS spread of a beam of given
    diameter, propagated through a lens of given thickness, by changing
    the curvatures of the two sides of the lens. It is meant to be used
    in conjunction with an optimisation function.
    """
    
    aperture_radius_1 = np.absolute(1/float(curvatures[0]))
    aperture_radius_1 -= 0.001*aperture_radius_1
    aperture_radius_2 = np.absolute(1/float(curvatures[1]))
    aperture_radius_2 -= 0.001*aperture_radius_2
    lens_front = opt.SphericalRefraction(focal_length, curvatures[0], 
                                        aperture_radius_1, 1., ref_index)
    lens_back = opt.SphericalRefraction(focal_length + thickness, 
                                        curvatures[1], aperture_radius_2, 
                                        ref_index, 1.)                                    
    max_radius = diameter/2.
    test_beam = rt.CollimatedBeam([0,0,0], [0,0,1], 6, max_radius, 2)
    rms_spread = rms_xy_spread([lens_front, lens_back], focal_length, test_beam)
    return np.log10(rms_spread)

def optimise_curvature(focal_length = 100., beam_diameter = 5., 
                        thickness = 5., ref_index =1.5, initial_guess = []):
    """
    focal_length - float argument, desired focal length of lens.
    
    beam_diameter - float argument, diameter of collimated beam.
    
    thickness - float argument, thickness of lens. Measured as the distance
    between the two optical axis intercepts of the two sides of the lens.
    
    ref_index - float argument, refractive index of lens. Refractive index of
    outside medium is assumed to be 1 (air).
    
    Returns the optimal curvatures of the two sides of the lens for a given
    focal length, lens thickness, beam diameter, and refractive index by 
    minimising the RMS spread of the output on the x-y plane at the focal point 
    (i.e. minimising abberations).
    """
    max_curv = 1./(beam_diameter/2.)
    if initial_guess == []:
        initial_guess = [max_curv/20., -max_curv/200.]
    #ALTERNATIVE OPTIMISATION METHOD
    #optimize_result_q = scopt.minimize(curvature_optimisation_func, 
    #                                x0=initial_guess, 
    #                                args = (focal_length, beam_diameter, 
    #                                        thickness, ref_index), 
    #                                method = 'Powell')
    #if optimize_result_q.success is True:
    #    results_1 = optimize_result_q.x
    #    return results_1
    #else:
    #    return optimize_result_q.message
    optimize_result = scopt.fmin_tnc(curvature_optimisation_func,
                                    x0 = initial_guess,
                                    args = (focal_length, beam_diameter, 
                                            thickness, ref_index),
                                    approx_grad = True,
                                    messages = False,
                                    maxfun = 500,
                                    bounds = [(1e-13, max_curv),(-max_curv-0.002, -1e-13)])

    results_2 = optimize_result[0]
    return results_2.tolist()

def plot_RMS_curvature(curv_init = 0, curv_final = 0.1, fragmentation = 10):
    """
    curv_init, curv_final - float arguments, initial and final curvatures for
    which the RMS spread at the focal point is evaluated. They must be positive.
    
    fragmentation - integer argument, number of steps between the initial and
    final curvatures for which the RMS evaluation is done. Determines detail
    of resulting plot.
    
    Plots the logarithm (base 10) of the RMS spread at the focal point as a
    surface plot with respect to combinations of the two curvatures of the
    lens. Also plots the same data as a contour plot.
    """
    if curv_init == 0:
        curv_init += 1e-13
    curv_1 = np.linspace(curv_init, curv_final, fragmentation)
    curv_2 = np.linspace(-curv_final, -curv_init, fragmentation)
    curv1, curv2 = np.meshgrid(curv_1, curv_2)
    rms = np.array([curvature_optimisation_func([x, y])
                    for x,y in zip(np.ravel(curv1),np.ravel(curv2))])
    rms = rms.reshape(curv1.shape)
    
    fig = plt.figure("Surface Plot")
    ax = fig.gca(projection='3d')
    X, Y, Z = curv1, curv2, rms
    ax.plot_surface(X, Y, Z, alpha=0.8, cmap=plt.cm.coolwarm, linewidth=0)
    ax.set_xlabel('Curvature Front / $mm^{-1}$')
    ax.set_xlim(curv_init, curv_final)
    ax.set_ylabel('Curvature Back / $mm^{-1}$')
    ax.set_ylim(-curv_final, -curv_init)
    ax.set_zlabel('log$_{10}$ RMS Spread')
    ax.set_zlim(np.amin(rms), np.amax(rms))
    
    plt.show()
    
    fig = plt.figure("Contour Plot")
    ax = fig.gca()
    plt.contourf(X, Y, Z, 1000)
    plt.xlabel('Curvature Front / $mm^{-1}$')
    plt.ylabel('Curvature Back / $mm^{-1}$')
    plt.colorbar()
    
    plt.show()
    
def lensmaker_equation(curv_1, focal_length, thickness, ref_index):
    """
    The lensmaker equation relates the two curvatures of a lens, its thickness,
    its focal length, and its refractive index. Given the curvature of the front
    part of the lens and the other parameters of the lens, the curvature of the
    back of the lens can be calculated.
    
    The function returns the curvature of the back of the lens.
    """
    if type(curv_1) is np.ndarray:
        curv_1 = curv_1[0]
    constant_1 = 1./(focal_length*(ref_index - 1.))
    constant_2 = ((ref_index - 1.)*thickness)/ref_index
    curv_2 = (constant_1 - curv_1)/(constant_2*curv_1 - 1.)
    return curv_2
    
def curvature_optimisation_func_lensmaker(curv_1 = 0.002, focal_length = 100.,
                                diameter = 5., thickness = 5., ref_index = 1.5):
    """
    This function is used to minimise the RMS spread of a beam of given
    diameter, propagated through a lens of given thickness, by changing
    the curvatures of the two sides of the lens. It is meant to be used
    in conjunction with an optimisation function.
    
    This function is bounded by the lensmaker equation, so it only requires
    one curvature as an argument and the other is calculated using the
    given focal length, thickness, and refractive index.
    """
    curv_2 = lensmaker_equation(curv_1, focal_length, thickness, ref_index)
    if curv_2 == 0:
        curv_2 -= 1e-13
    if curv_1 == 0:
        curv_1 += 1e-13
    aperture_radius_1 = np.absolute(1/float(curv_1))
    aperture_radius_1 -= 1e-6*aperture_radius_1
    aperture_radius_2 = np.absolute(1/float(curv_2))
    aperture_radius_2 -= 1e-6*aperture_radius_2
    lens_front = opt.SphericalRefraction(focal_length, curv_1, 
                                        aperture_radius_1, 1., ref_index)
    lens_back = opt.SphericalRefraction(focal_length + thickness, 
                                        curv_2, aperture_radius_2, 
                                        ref_index, 1.)
    max_radius = diameter/2.
    test_beam = rt.CollimatedBeam([0,0,0], [0,0,1], 6, max_radius, 2)
    rms_spread = rms_xy_spread([lens_front, lens_back], focal_length, test_beam)
    return np.log10(rms_spread)

def optimise_curvature_lensmaker(focal_length = 100., beam_diameter = 5., 
                                thickness = 5., ref_index =1.5):
    """
    focal_length - float argument, desired focal length of lens.
    
    beam_diameter - float argument, diameter of collimated beam.
    
    thickness - float argument, thickness of lens. Measured as the distance
    between the two optical axis intercepts of the two sides of the lens.
    
    ref_index - float argument, refractive index of lens. Refractive index of
    outside medium is assumed to be 1 (air).
    
    Returns the optimal curvatures of the two sides of the lens for a given
    focal length, lens thickness, beam diameter, and refractive index by 
    minimising the RMS spread of the output on the x-y plane at the focal point 
    (i.e. minimising abberations).
    
    This function is bounded by the lensmaker equation.
    """
    curv_1_upper_bound = 1./(focal_length*(ref_index - 1.))
    max_curv_beam = 1./(beam_diameter/2.)
    max_curv = min(curv_1_upper_bound, max_curv_beam)
    initial_guess = max_curv/2.
    #ALTERNATIVE OPTIMISATION METHOD
    #optimize_result_q = scopt.minimize(curvature_optimisation_func_lensmaker, 
    #                                x0 = initial_guess,
    #                                args = (focal_length, beam_diameter, 
    #                                            thickness, ref_index),
    #                                method = 'Powell')
    #if optimize_result_q.success is True:
    #    optimal_curv_1 = np.ndarray.item(optimize_result_q.x)
    #    optimal_curv_2 = lensmaker_equation(optimal_curv_1, focal_length, 
    #                                        thickness, ref_index)
    #    result_1 = [optimal_curv_1, optimal_curv_2]
    #else:
    #    return optimize_result_q.message
    optimize_result = scopt.fmin_tnc(curvature_optimisation_func_lensmaker,
                                    x0 = initial_guess, messages = False,
                                    args = (focal_length, beam_diameter, 
                                            thickness, ref_index),
                                    approx_grad = True,
                                    bounds = [(1e-13, max_curv)])
    
    optimal_curv_1 = optimize_result[0][0]
    optimal_curv_2 = lensmaker_equation(optimal_curv_1, focal_length, 
                                        thickness, ref_index)
    result_2 = [optimal_curv_1, optimal_curv_2]
    return result_2
    
def plot_lensmaker(curv_init = 0, focal_length = 100, 
                    thickness = 5., fragmentation = 1000):
    """
    Takes the initial curvature and the fragmentation level (the number of steps
    between the initial and final curvature) to plot the two curvatures of the
    lens and the RMS spread at the given focal point against the first curvature.
    The additional argument thickness determines the separation of the two sides
    of the lens.
    """
    if curv_init == 0:
        curv_init += 1E-13
    curv_final = 1./(focal_length*(1.5 - 1.))
    curv_1 = np.linspace(curv_init, curv_final, fragmentation)
    curv_2 = [lensmaker_equation(x, focal_length, thickness, 1.5) 
                                                for x in curv_1]
    rms = [curvature_optimisation_func_lensmaker(x, focal_length, thickness) 
                                                            for x in curv_1]
    
    plt.figure('Curvatures')
    plt.plot(curv_1, curv_2, 'r-')
    plt.xlabel('Curvature Front / $mm^{-1}$')
    plt.ylabel('Curvature Back / $mm^{-1}$')
    plt.show()
    
    plt.figure('RMS vs Curvature')
    plt.plot(curv_1, rms, 'b-')
    plt.xlabel('Curvature Front / $mm^{-1}$')
    plt.ylabel('log$_{10}$ RMS Spread')
    plt.show()
    
def curvature_ratio(curv_1_list, curv_2_list):
        ratio_list = []
        for i in range(len(curv_1_list)):
            ratio = (curv_1_list[i]/(-curv_2_list[i]))
            ratio_list.append(ratio)
        ratio_mean = np.mean(ratio_list)
        ratio_error = np.std(ratio_list)/len(ratio_list)
        return ratio_mean, ratio_error
        
def plot_curvatures_vs_focal_length(f_len_init = 5., f_len_final = 500., 
                                    fragmentation = 10, beam_diameter = 10., 
                                    thickness = 5., ref_index = 1.5168):
    """
    Takes an initial and final focal length along with a granularity argument,
    fragmentation, and plots the curvatures of both sides of an optimal lens for
    each focal length, along with the logarithm of the RMS spread at the focal
    point and the diffraction limit for the given wavelength (set to 588nm).
    
    Other arguments that can be given are all the properties of the desired
    lens, such as thickness and refractive index, and the diameter of the
    beam sent into the lens.
    """
    f_len_list = np.linspace(f_len_init, f_len_final, fragmentation)
    curv_1_list, curv_2_list, rms_list, diffraction_limit_list = [], [], [], []
    for f_len in f_len_list:
        #Progress Bar
        print(str(round(((f_len-f_len_init)/(f_len_final-f_len_init))*100,1))+"%", end='\r')
        initial_guess = optimise_curvature_lensmaker(f_len, beam_diameter,
                                                    thickness, ref_index)
        if initial_guess[1] == 0:
            initial_guess = [curv_1_list[-1], curv_2_list[-1]]
        curvatures = optimise_curvature(f_len, beam_diameter, thickness,
                                            ref_index, initial_guess)
        rms = curvature_optimisation_func(curvatures, f_len, beam_diameter, 
                                            thickness, ref_index)
        diffraction_limit = np.log10((588e-6)*f_len/(beam_diameter))
        
        curv_1_list.append(curvatures[0])
        curv_2_list.append(curvatures[1])
        rms_list.append(rms)
        diffraction_limit_list.append(diffraction_limit)
 
    fig, ax1= plt.subplots()
    curv1 = ax1.plot(f_len_list, curv_1_list, 'r-', label = 'Front Curvature')
    curv2 = ax1.plot(f_len_list, curv_2_list, 'b-', label = 'Back Curvature')
    ax1.set_xlabel('Focal Length / $mm$')
    ax1.set_ylabel('Curvatures / $mm^{-1}$')

    ax2 = ax1.twinx()
    rms = ax2.plot(f_len_list, rms_list, 'g-', label = 'RMS Spread')
    diff = ax2.plot(f_len_list, diffraction_limit_list, 'm--', label = 'Diffraction Limit')
    ax2.set_ylabel('log$_{10}$ of RMS Spread and Diff. Limit')
    #Fix scaling for aesthetics.
    #y_diff = (rms_list[-1] + diffraction_limit_list[-1])/2. - diffraction_limit_list[0]
    #ax2.set_ylim(diffraction_limit_list[0], diffraction_limit_list[0] + 2*y_diff)
    
    lines = curv1 + curv2 + rms + diff
    labels = [line.get_label() for line in lines]
    plt.legend(lines, labels, loc=1)
    
    print(curvature_ratio(curv_1_list, curv_2_list))
    
    plt.show()
    
    