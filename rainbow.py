"""
A module to model the physics of rainbow formation.
"""
from __future__ import print_function
import optical as opt
import raytracer as rt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import ray_plot as rtplot


def white_light(px_init = 40., k_init = [0,0,1], lambda_no = 10, 
                lambda_init = 390., lambda_final = 700., analysis = False,
                analysis_arg = 0):
    """
    px_init - float argument, determines starting x position of white light ray.
    k_init - list argument, determines direction of white light ray.
    lambda_no - integer argument, determines number of component wavelengths.
    lambda_init, lambda_final - float arguments, determine wavelength range.
    analysis - boolean argument, modifier for convenient use in analysis.
    analysis_arg - integer argument, used for convenience in analysis.
    
    The function returns a list of Ray objects with an array of different
    wavelengths, determined by the given arguments. The list can be used
    as a single white light ray that can be propagated through dispersive
    materials and split into component rays depending on their wavelength.
    """
    ray_list = []
    lambda_list = np.linspace(lambda_init, lambda_final, lambda_no)
    if analysis == True:
        lambda_list = [lambda_list[analysis_arg]]
    for wavelength in lambda_list:
        ray = rt.Ray([px_init, 0, 0], k_init, wavelength)
        ray_list.append(ray)
    return ray_list
    
def create_rainbow_sun_angle(droplet_centre_z = 100., droplet_radius = 40., 
                                sun_angle = 45., lambda_no = 10):
    """
    droplet_centre_z - float argument, determines droplet centre.
    droplet_radius - float argument, determines droplet radius.
    sun_angle - float argument, given in degrees, determines the angle the sun 
    makes with a point a radius of the droplet behind of the droplet.
    lambda_no - integer argument, determines the number of component wavelengths
    in the white light sent into the droplet.
    
    Plots a 2D x-z plane of the white light refracting and reflecting through
    the water droplet to form a rainbow.
    """
    if sun_angle >= 30. or sun_angle <= -30.:
        raise Exception("Rays must be sent into the droplet to form rainbow.")
    px_init = np.absolute((droplet_centre_z + 2*droplet_radius) * 
                                np.tan(np.radians(sun_angle)))
    k_init = [-px_init, 0, (droplet_centre_z + 2*droplet_radius)]
    
    droplet = opt.Droplet([0,0,droplet_centre_z], droplet_radius)

    
    colour_generator=iter(plt.cm.rainbow(np.linspace(0,1,lambda_no)))
    for ray in white_light(px_init, k_init, lambda_no):
        colour=next(colour_generator)
        while ray._reachedOutputPlane is False:
            ray.propagate([droplet], True, colour)
    rtplot.plotDroplet2D(droplet_centre_z, 0., droplet_radius)
    
def create_rainbow(droplet_centre_z = 100., droplet_radius = 40.,
                                                    px=35, lambda_no = 10):
    """
    droplet_centre_z - float argument, determines droplet centre.
    droplet_radius - float argument, determines droplet radius.
    px - float argument, determines initial x position of white light ray.
    lambda_no - integer argument, determines the number of component wavelengths
    in the white light sent into the droplet.
    
    Plots a 2D x-z plane of the white light refracting and reflecting through
    the water droplet to form a rainbow. The ray sent into the droplet is always
    parallel to the z axis.
    """
    if px >= droplet_radius or px <= -droplet_radius:
        raise Exception("Rays must be sent into the droplet to form rainbow.")
    droplet = opt.Droplet([0,0,droplet_centre_z], droplet_radius)

    colour_generator=iter(plt.cm.rainbow(np.linspace(1,0,lambda_no)))
    for ray in white_light(px, [0,0,1], lambda_no):
        colour=next(colour_generator)
        while ray._reachedOutputPlane is False:
            ray.propagate([droplet], True, colour)
    rtplot.plotDroplet2D(droplet_centre_z, 0., droplet_radius)
    
def white_light_3D(px_init = 40., k_init = [0,0,1], lambda_no = 10, 
                    lambda_init = 390., lambda_final = 700., rmax = 5.):
    """
    px_init - float argument, determines starting x position of white light beam.
    k_init - list argument, determines direction of white light beam.
    lambda_no - integer argument, determines number of component wavelengths.
    lambda_init, lambda_final - float arguments, determine wavelength range.
    rmax - the radius of the white light beam.
    
    The function returns a list of CollimatedBeam objects with an array of 
    different wavelengths, determined by the given arguments. The list can 
    be used as a single white light ray that can be propagated through 
    dispersive materials and split into component beams depending on their
    wavelength.
    """
    wavelength = lambda_init
    lambda_step = (lambda_final - lambda_init)/float(lambda_no)
    lambda_list = []
    while wavelength < lambda_final:
        beam = rt.CollimatedBeam([px_init, 0, 0], k_init, 
                                wavelength=wavelength, r_max=rmax, r_spacing=0)
        lambda_list.append(beam)
        wavelength += lambda_step
    return lambda_list

def create_rainbow_3D(droplet_centre_z = 100., droplet_radius = 40.,
                                                    px=30, lambda_no = 10):
    """
    DOES NOT PROPERLY PLOT RAYS IN 3D, ISSUE WITH MATPLOTLIB.
    
    droplet_centre_z - float argument, determines droplet centre.
    droplet_radius - float argument, determines droplet radius.
    px - float argument, determines initial x position of white light ray.
    lambda_no - integer argument, determines the number of component wavelengths
    in the white light sent into the droplet.
    
    Plots a 2D x-z plane of the white light refracting and reflecting through
    the water droplet to form a rainbow. The ray sent into the droplet is always
    parallel to the z axis.
    """
    droplet = opt.Droplet([0,0,droplet_centre_z], droplet_radius)
    
    
    colour_generator=iter(plt.cm.rainbow(np.linspace(0,1,lambda_no)))
    for beam in white_light_3D(px, [0,0,1], lambda_no):
        colour=next(colour_generator)
        #while beam._reachedOutputPlane is False:
        for i in range(10):
            beam.propagate([droplet], plot3D = True, plot2D=True, colour = colour)
    rtplot.plotDroplet3D(droplet_centre_z, droplet_radius)
    
def create_rainbow_analysis(droplet_centre_z = 100., droplet_radius = 50.,
                        px=35, lambda_no = 10, analysis = False, analysis_arg =0,
                        double_rainbow = False):
    """
    droplet_centre_z - float argument, determines droplet centre.
    droplet_radius - float argument, determines droplet radius.
    px - float argument, determines initial x position of white light ray.
    lambda_no - integer argument, determines the number of component wavelengths
    in the white light sent into the droplet.
    analysis - boolean argument, modifier for convenient use in analysis.
    analysis_arg - integer argument, used for convenience in analysis.
    double_rainbow - boolean argument, enables double rainbow analysis mode.
    
    Refracts and reflects white light through a water droplet to form a rainbow,
    returning a list of only the rays that make up the rainbow that the observer
    sees. This function is the same as create_rainbow(), but does not plot and
    is used for the analysis of the rainbow formation.
    """
    if px >= droplet_radius or px <= -droplet_radius:
        raise Exception("Rays must be sent into the droplet to form rainbow.")
    droplet = opt.Droplet([0,0,droplet_centre_z], droplet_radius)
    white_light_rays = white_light(px, [0,0,1], lambda_no, analysis = analysis, 
                                            analysis_arg = analysis_arg)
    rainbow_rays = []
    for ray in white_light_rays:
        while ray._reachedOutputPlane is False:
            ray.propagate([droplet])
        if len(ray.vertices()) == 5 and double_rainbow is False:
            rainbow_rays.append(ray)
        elif len(ray.vertices()) == 6 and double_rainbow is True:
            rainbow_rays.append(ray)
    return rainbow_rays
    
def find_rainbow_angle(rainbow_rays, double_rainbow):
    """
    rainbow_rays - list argument, consists of rays that make up the rainbow.
    double_rainbow - boolean argument, determines the treatment of the data.
    If it is set to True, it returns values from the analysis of the secondary
    rainbow. If set to False, returns values for primary rainbow.
    
    Uses the direction vectors of the rays that make up the rainbow to calculate
    the angle they make with the 'ground', or the observer, and returns that
    angle. If there are no rays in the list, it returns NaN.
    """
    angles = []
    for ray in rainbow_rays:
        if double_rainbow is False:
            angle = np.arccos(np.inner(ray._k[3], np.array([0,0,-1.])))
        else:
            angle = np.arccos(np.inner(ray._k[4], np.array([0,0,-1.])))
        angles.append(np.degrees(angle))
    if angles != []:
        angle = angles[0]
    else:
        angle = np.nan
    return angle

def plot_rainbow_angles(intensity = 100, lambda_no = 20, px_no = 50):
    """
    intensity - integer argument, determines amount of rays sent into the
    droplet for analysis. The higher, the more accurate the data.
    lambda_no - integer argument, determines number of wavelengths evaluated
    for the rainbow spectrum. Making this higher than 30-50 has minimum effect
    on the quality of the results and slows the function down. Keep low.
    px_no - integer argument, determines granularity of the impact parameter,
    or the initial x position of the ray. The higher, the smoother the results.
    
    Plots the angles at which the observer sees the rainbow for each wavelength
    against the impact parameter, or initial x position, of the rays falling in
    the water droplet. Both the primary and secondary rainbows are plotted, and
    the intensity of light is also plotted on a second axis. Alexander's dark
    band is added automatically to the graph.
    """
    droplet_radius = 1.
    px_range = np.linspace(0, droplet_radius-1e-2, px_no)
    angles = [[] for x in range(lambda_no)]
    angles_dr = [[] for x in range(lambda_no)]
    intensities = []
    for px in px_range:
        #Progress Counter
        print(str(round((px/droplet_radius)*100,1))+'%', end='\r')
        for lam_no in range(lambda_no):
            rainbow_rays, double_rainbow_rays = [], []
            for i in range(intensity):
                rainbow_iter = create_rainbow_analysis(lambda_no=lambda_no, 
                                px=px, analysis = True, analysis_arg = lam_no,
                                droplet_radius = droplet_radius)
                rainbow_rays += rainbow_iter
                
                rainbow_iter = create_rainbow_analysis(lambda_no=lambda_no, 
                                px=-px, analysis = True, analysis_arg = lam_no,
                                double_rainbow = True, 
                                droplet_radius = droplet_radius)
                double_rainbow_rays += rainbow_iter
                
            angles[lam_no].append(find_rainbow_angle(rainbow_rays, False))
            angles_dr[lam_no].append(find_rainbow_angle(double_rainbow_rays, True))
            if lam_no == 0:
                intensities.append(len(rainbow_rays)+len(double_rainbow_rays))
        
    fig, ax1= plt.subplots()
    colour_generator=iter(plt.cm.rainbow(np.linspace(0,1,lambda_no)))
    for lam_no in range(lambda_no):
        colour=next(colour_generator)
        angles[lam_no] = np.array(angles[lam_no])
        angle_mask = np.isfinite(angles[lam_no])
        angle = ax1.plot(px_range[angle_mask], angles[lam_no][angle_mask], 
                            c=colour, label = 'Rainbow Angles')
        
        angles_dr[lam_no] = np.array(angles_dr[lam_no])
        angle_mask_dr = np.isfinite(angles_dr[lam_no])
        angle_dr = ax1.plot(px_range[angle_mask_dr], 
                            angles_dr[lam_no][angle_mask_dr], 
                            c=colour, label = 'Rainbow Angles')
    ax1.set_xlabel('Impact Parameter / $mm$')
    ax1.set_ylabel('Angle / $\deg^{\circ}$')
    
    ax2 = ax1.twinx()
    
    #Normalisation
    intensities = [float(i)/sum(intensities) for i in intensities]
    intensities = [float(i)/max(intensities) for i in intensities]
    
    #POLYNOMIAL FIT
    #coeff = np.polyfit(px_range, intensities, 10)
    #def intensity_func(px): 
    #    return np.polyval(coeff, px)
    #intensity_line = ax2.plot(px_range, intensity_func(px_range), 'k-', 
    #                           alpha = 0.8, label = 'Intensity Fit')
    
    intensity = ax2.plot(px_range, intensities, 'k.', alpha = 0.3,
                        label = 'Intensity Data')
    
    ax2.set_ylabel('Intensity')
    
    #Alexander's Dark Band
    min_px = px_range[0]
    max_px = px_range[-1] - min_px + 1e-2
    max_rainbow = np.nanmax(angles[-1])
    min_drainbow = np.nanmin(angles_dr[-1]) - max_rainbow
    
    ax1.add_patch(
        patches.Rectangle(
            (min_px, max_rainbow),
            max_px,
            min_drainbow,
            #hatch='\\',
            facecolor='grey',
            fill=True,
            alpha = 0.4,
            linewidth = 0.5,
            linestyle = 'dotted',
            label = 'Alexander\'s Band'
        )
    )
    
    ax1.text(2*1e-2, max_rainbow+1.1*min_drainbow,
            'Alexander\'s Dark Band', fontsize = 14)
    
    lines = angle + intensity #+ intensity_line
    labels = [line.get_label() for line in lines]
    plt.legend(lines, labels, loc=1)
    
    plt.show()