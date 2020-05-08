#### Simulation II by Joshua Yao-Yu Lin, Warren Morningstar, Hang Yu

from __future__ import division
import sys
sys.path.insert(0, '../EvilLens/')
import numpy as np
import evillens as evil
import matplotlib.pyplot as plt
from astropy import units,constants
from astropy.cosmology import Planck15 as cosmo
from scipy.interpolate import interp1d
from scipy.spatial import distance_matrix
from scipy.ndimage import gaussian_filter
import scipy.misc
from PIL import Image
import copy

import argparse
# Define argument
parser = argparse.ArgumentParser()
parser.add_argument('--train', dest='IsTrain', action='store_true')
parser.add_argument('--test', dest='IsTrain', action='store_false')
args = parser.parse_args()


def draw_macro_lens(npix,pix_res,parameter_rms=[0.1,5*10**11,0.4,2*np.pi,0.04,0.04,0.,0.,0.,0.,0.,0.],redshift_range=[[0.1,1.2],[1.4,3.0]]):
    '''
    Create strong lensing simulation for the marco model (smooth model, without any dark matter substructures)
    '''
    zl = 0.5 #np.random.random()*(redshift_range[0][1]-redshift_range[0][0])+redshift_range[0][0]
    zs = 1.0 #np.random.random()*(redshift_range[1][1]-redshift_range[1][0])+redshift_range[1][0]

    ####
    lens = evil.PowerKappa(zl,zs)
    lens.setup_grid(NX=npix,NY=npix,pixscale=pix_res,offset=0.0)

    Gamma = np.random.normal(1.0,parameter_rms[0])
    logM = 1.17 + np.random.random() * 0.01 #np.log10(np.random.random()*(parameter_rms[1]-10**11)+10**11)/10.
    q = 1-np.random.random()*parameter_rms[2]
    angle = np.random.random()*parameter_rms[3]
    centroid = [np.random.random()*parameter_rms[4],np.random.random()*parameter_rms[5]]

    g1 = np.random.random()*parameter_rms[6]-parameter_rms[6]/2.
    g2 = np.random.random()*parameter_rms[7]-parameter_rms[7]/2.
    A3 = np.random.random()*parameter_rms[8]-parameter_rms[8]/2.
    B3 = np.random.random()*parameter_rms[9]-parameter_rms[9]/2.
    A4 = np.random.random()*parameter_rms[10]-parameter_rms[10]/2.
    B4 = np.random.random()*parameter_rms[11]-parameter_rms[11]/2.

    lens.build_kappa_map(logM=logM,q=q,angle=angle,centroid=centroid,Gamma=Gamma)
    lens.deflect()
    lens.add_multipoles([[g1,g2],[A3,B3],[A4,B4]])
    return lens


def draw_subhalo_population(lens,halo_mass=10**12.3,mass_cutoff=10**8.8):
    '''
    Given a lens, a halo mass for that lens, and a mass cutoff, draw a subhalo
    population from the aquarius simulation.
    '''

    # first, compute the CDF of the subhalos (include larger range than allowed subhalo masses)
    Msubs = 10**np.linspace(np.log10(mass_cutoff)-3.,np.log10(halo_mass),4000)
    CDF = evil.Subhalo_cumulative_mass_function(Msubs,halo_mass)

    # build interpolation kernel
    finterp = interp1d(CDF/np.max(CDF),Msubs,'linear')

    # now draw a number of subhalos equivalent to the max of the CDF from the CDF
    Subhalo_masses = finterp(np.random.random(np.max(CDF).astype('int')))

    # Cut all subhalos below the mass cutoff
    Subhalo_masses = Subhalo_masses[(Subhalo_masses>mass_cutoff)]

    # How many subhalos are left?
    Nsubs = len(Subhalo_masses)
    #print("#Subhalo : ", Nsubs)

    # now get radial distributions...
    # pdf of n(r) is an Einasto profile.  Get CDF
    r_kpc = np.linspace(0,10000.,10**6)
    pdf = evil.Einasto(r_kpc,0.678,199.)
    cdf = np.flipud(np.cumsum(np.flipud(pdf)))

    # create interpolation kernel
    radial_interp = interp1d(cdf/np.max(cdf),r_kpc,'linear')

    # draw subhalo radii
    r = radial_interp(np.random.random(Nsubs))

    # Have radius, now want position angle
    # random in 3D
    phi = np.random.random(Nsubs)*2*np.pi
    theta = np.random.random(Nsubs)*np.pi

    # radii and angles to xyz (also add ellipticity)
    xp = r*np.cos(phi)*np.sin(theta)/lens.q
    yp = r*np.sin(phi)*np.sin(theta)*lens.q

    # rotate to align with halo major axis and
    # produce ellipticity
    angle = lens.angle + np.pi/2.
    x =  np.cos(angle) * xp - np.sin(angle) * yp
    y =  np.sin(angle) * xp + np.cos(angle) * yp

    # Have masses and position, in kpc, now get them in
    # arcseconds
    x = (x*units.kpc/lens.Dd).decompose().value*3600.*180./np.pi
    y = (y*units.kpc/lens.Dd).decompose().value*3600.*180./np.pi

    # center them on the lens and stack them in a
    # single array
    x += lens.centroid[0]
    y += lens.centroid[1]
    sub_centroids = np.dstack([x,y])[0,:,:]

    return Subhalo_masses,sub_centroids

def Compute_subhalo_lens(lens,Subhalo_masses,sub_centroids):
    if lens.kappa is None:
        raise Exception('Need main lens kappa map \n')
    if lens.alpha_x is None:
        raise Exception('Need main lens deflection angles \n')
    if len(Subhalo_masses) == 0:
        return None
    assert lens.alpha_x.shape == lens.image_x.shape
    assert lens.alpha_y.shape == lens.image_y.shape
    assert lens.alpha_x.shape == lens.alpha_y.shape
    assert lens.x.shape == lens.kappa.shape
    assert lens.kappa.shape == lens.y.shape


    Nsubhalos = len(Subhalo_masses)
    Msub = Subhalo_masses*units.solMass
    # calculate tidal radius of subhalo using parameters of main halo
    EinRad_M = 4.0*np.pi*(lens.sigma/constants.c)**2* lens.Dds/lens.Ds


    Sigma_sub = (np.sqrt(4.0/np.pi) * constants.G * Msub *lens.sigma \
                    /(np.pi * EinRad_M * lens.Dd))**(1.0/3.0)
    Rtidal = (Sigma_sub / lens.sigma / np.sqrt(4.0/np.pi)) * EinRad_M
    Rcore = Rtidal.decompose().value * 3600.0*180.0/np.pi

    #create subhalo object.  Have coordinates be those of main halo.
    all_subhalos = evil.AnalyticPseudoJaffeLens(lens.Zd,lens.Zs)
    all_subhalos.setup_grid(NX=lens.NX,NY=lens.NY,pixscale=lens.pixscale,offset=lens.offset)
    all_subhalos.build_kappa_map(0.0,Rcore[0],[np.random.random(),np.random.random()],n=4,GAMMA=2)
    all_subhalos.deflect()

    no_subhalos = evil.AnalyticPseudoJaffeLens(lens.Zd,lens.Zs)
    no_subhalos.setup_grid(NX=lens.NX,NY=lens.NY,pixscale=lens.pixscale,offset=lens.offset)
    no_subhalos.build_kappa_map(0.00001,Rcore[0],[np.random.random(),np.random.random()],n=4,GAMMA=2)
    no_subhalos.deflect()

    this_subhalo = evil.AnalyticPseudoJaffeLens(lens.Zd,lens.Zs)
    this_subhalo.setup_grid(NX=lens.NX,NY=lens.NY,pixscale=lens.pixscale,offset=lens.offset)

    for i in range(Nsubhalos):
        this_subhalo.build_kappa_map(Msub[i].value, a = Rcore[i] , \
                    centroid = sub_centroids[i], n = 4, GAMMA = 2)
        this_subhalo.deflect()

        all_subhalos = all_subhalos + this_subhalo



    return all_subhalos


def Compute_detectable_subhalo_lens(lens,Subhalo_masses,sub_centroids):
    if lens.kappa is None:
        print("1")
        raise Exception('Need main lens kappa map \n')
    if lens.alpha_x is None:
        print("2")
        raise Exception('Need main lens deflection angles \n')
    if len(Subhalo_masses) == 0:
        return None
    assert lens.alpha_x.shape == lens.image_x.shape
    assert lens.alpha_y.shape == lens.image_y.shape
    assert lens.alpha_x.shape == lens.alpha_y.shape
    assert lens.x.shape == lens.kappa.shape
    assert lens.kappa.shape == lens.y.shape


    Nsubhalos = len(Subhalo_masses)
    Msub = Subhalo_masses*units.solMass
    # calculate tidal radius of subhalo using parameters of main halo
    EinRad_M = 4.0*np.pi*(lens.sigma/constants.c)**2* lens.Dds/lens.Ds


    Sigma_sub = (np.sqrt(4.0/np.pi) * constants.G * Msub *lens.sigma \
                    /(np.pi * EinRad_M * lens.Dd))**(1.0/3.0)
    Rtidal = (Sigma_sub / lens.sigma / np.sqrt(4.0/np.pi)) * EinRad_M
    Rcore = Rtidal.decompose().value * 3600.0*180.0/np.pi

    #create subhalo object.  Have coordinates be those of main halo.
    detectable_subhalos = evil.AnalyticPseudoJaffeLens(lens.Zd,lens.Zs)
    detectable_subhalos.setup_grid(NX=lens.NX,NY=lens.NY,pixscale=lens.pixscale,offset=lens.offset)
    detectable_subhalos.build_kappa_map(0.0,Rcore[0],[np.random.random(),np.random.random()],n=4,GAMMA=2)
    detectable_subhalos.deflect()

    no_subhalos = evil.AnalyticPseudoJaffeLens(lens.Zd,lens.Zs)
    no_subhalos.setup_grid(NX=lens.NX,NY=lens.NY,pixscale=lens.pixscale,offset=lens.offset)
    no_subhalos.build_kappa_map(0.00001,Rcore[0],[np.random.random(),np.random.random()],n=4,GAMMA=2)
    no_subhalos.deflect()

    this_subhalo = evil.AnalyticPseudoJaffeLens(lens.Zd,lens.Zs)
    this_subhalo.setup_grid(NX=lens.NX,NY=lens.NY,pixscale=lens.pixscale,offset=lens.offset)

    for i in range(Nsubhalos):
        this_subhalo.build_kappa_map(Msub[i].value, a = Rcore[i] , \
                    centroid = sub_centroids[i], n = 4, GAMMA = 2)
        this_subhalo.deflect()

        detectable_subhalos = detectable_subhalos + this_subhalo



    return detectable_subhalos


def Subhalo_einstein_radius(M_sub, lens):
    '''
    treat subhalo as point mass (instead of pseudo-jaffe profile) to save computational time
    '''
    theta_E = M_sub**0.5 * ((4 * constants.G * constants.M_sun.value /(constants.c.value** 2)) * lens.Dds.value/(lens.Dd.value*lens.Ds.value) * (10**6 * constants.pc.value)**(-1) )**0.5 * 180/np.pi * 3600
    return theta_E.value



def Generate_single_training_example(numpix=192,pix_res=0.04):

    subhalo_kappa,final_image, lens_image, lens_source = None, None, None, None

    lens = draw_macro_lens(numpix,pix_res)
    build_params = [1.0,0.5,np.random.random()*0.75+0.25,np.random.random()*2*np.pi,np.random.normal(0.,0.1,2),np.random.randint(1,10),0.25,10**-8,False,np.random.randint(0,4294967295,3),1.0]
    lens.source.setup_grid(NX=numpix,NY=numpix,pixscale=lens.pixscale/4.)
    lens.source.build_from_clumps(*build_params)
    lens.raytrace()
    lens_image = lens.image
    lens_source = lens.source.intensity
    lens_argmax = np.unravel_index(np.argmax(lens_image, axis=None), lens_image.shape) #np.argmax(lens_image)


    msub,pos_sub = draw_subhalo_population(lens,10**(lens.logM*10+0.5))
    Nsubhalos = len(pos_sub)

    subhalo_lens = Compute_subhalo_lens(lens,msub,pos_sub)
    if subhalo_lens is not None:
        output_lens = lens + subhalo_lens
    else:
        output_lens = lens
    ### testing zone
    lens_Zs = lens.Zs
    lens_Zl = lens.Zd

    adjusted_pos_sub = pos_sub/pix_res
    reduced_pos_sub_normalized = np.array([])
    reduced_msub = np.array([])

    #The origin point of the image is (numpix//2, numpix//2), positive-x is to the right, positive-y in to the buttom
    for i in range(len(adjusted_pos_sub)):
        pos = adjusted_pos_sub[i]
        # Selection criteria: within the range of the image
        if -numpix//2 < pos[0] < numpix//2 and -numpix//2 < pos[1] < numpix//2:
            reduced_msub = np.append(reduced_msub, msub[i])
            if len(reduced_pos_sub_normalized) == 0:
                reduced_pos_sub_normalized = np.array([pos[0]/(numpix//2), pos[1]/(numpix//2)])
            else:
                reduced_pos_sub_normalized = np.vstack((reduced_pos_sub_normalized,np.array([pos[0]/(numpix//2), pos[1]/(numpix//2)])))

    ###
    output_lens.source = evil.Source(lens.Zs)
    output_lens.source.setup_grid(NX=numpix,NY=numpix,pixscale=lens.pixscale/4.)
                #  size=2.0,clump_size = 0.1,axis_ratio=1.0, orientation=0.0,center=[0,0], Nclumps=50, n = 1 , error =10**-8,singlesource=False,seeds=[1,2,3],Flux=1.0
    #build_params = [1.0,0.5,np.random.random()*0.75+0.25,np.random.random()*2*np.pi,np.random.normal(0.,0.1,2),np.random.randint(1,10),0.25,10**-8,False,np.random.randint(0,4294967295,3),1.0]
    output_lens.source.build_from_clumps(*build_params)
    output_lens.raytrace()

    if subhalo_lens is not None:
        subhalo_kappa = subhalo_lens.kappa
        subhalo_kappa_log10 = np.log10(subhalo_kappa)
    else:
        subhalo_kappa = np.zeros((numpix, numpix))#subhalo_lens.kappa
        subhalo_kappa_log10 = np.zeros((numpix, numpix)) #np.log10(subhalo_kappa)
    final_image = output_lens.image


    PSF_sigma = 5
    blurred_output_lens = gaussian_filter(final_image, PSF_sigma)


    output_lens_qual_pos = np.argwhere(blurred_output_lens > (np.amax(blurred_output_lens)/10) )
    output_lens_non_pos = np.argwhere(blurred_output_lens < (np.amax(blurred_output_lens)/15) )


    Msub_solarmass = msub*units.solMass
    EinRad_M = 4.0*np.pi*(lens.sigma/constants.c)**2* lens.Dds/lens.Ds


    Sigma_sub = (np.sqrt(4.0/np.pi) * constants.G * Msub_solarmass *lens.sigma \
                    /(np.pi * EinRad_M * lens.Dd))**(1.0/3.0)
    Rtidal = (Sigma_sub / lens.sigma / np.sqrt(4.0/np.pi)) * EinRad_M
    Rcore = Rtidal.decompose().value * 3600.0*180.0/np.pi

    detectable_num_subhalo = 0
    detectable_pos_sub_list = []
    detectable_msub_list = []
    try:
        effective_radius = Subhalo_einstein_radius(msub,lens)
        for i, pos in enumerate(pos_sub):
            grid_pos = pos/pix_res
            grid_pos[0] += numpix//2
            grid_pos[1] += numpix//2
            min_distance = distance_matrix([grid_pos], output_lens_qual_pos).min()
            if min_distance < round(effective_radius[i]/pix_res):
                detectable_num_subhalo += 1
                detectable_pos_sub_list.append(pos_sub[i])
                detectable_msub_list.append(msub[i])
            detectable_pos_sub = np.array(detectable_pos_sub_list)
            detectable_msub = np.array(detectable_msub_list)


        detectable_subhalo_lens = Compute_detectable_subhalo_lens(lens,detectable_msub_list,detectable_pos_sub_list)

        if detectable_subhalo_lens is not None:
            detectable_subhalo_kappa = detectable_subhalo_lens.kappa
            detectable_subhalo_kappa_log10 = np.log10(detectable_subhalo_kappa)
        else:
            detectable_subhalo_kappa = np.zeros((numpix, numpix))#subhalo_lens.kappa
            detectable_subhalo_kappa_log10 = np.zeros((numpix, numpix)) #np.log10(subhalo_kappa)
    except:
        detectable_subhalo_kappa = np.zeros((numpix, numpix))
        detectable_subhalo_kappa += 0.00001
        detectable_subhalo_kappa_log10 = np.log10(detectable_subhalo_kappa)
        detectable_pos_sub = np.array(detectable_pos_sub_list)
        detectable_msub = np.array(detectable_msub_list)
        pass

    detectable_adjusted_pos_sub = detectable_pos_sub/pix_res
    detectable_reduced_pos_sub_normalized = np.array([])
    detectable_reduced_msub = np.array([])

        #The origin point of the image is (numpix//2, numpix//2), positive-x is to the right, positive-y in to the buttom
    for i in range(len(detectable_adjusted_pos_sub)):
        pos = detectable_adjusted_pos_sub[i]
        # Selection criteria: within the range of the image
        if -numpix//2 < pos[0] < numpix//2 and -numpix//2 < pos[1] < numpix//2:
            detectable_reduced_msub = np.append(detectable_reduced_msub, detectable_msub[i])
            if len(detectable_reduced_pos_sub_normalized) == 0:
                detectable_reduced_pos_sub_normalized = np.array([pos[0]/(numpix//2), pos[1]/(numpix//2)])
            else:
                detectable_reduced_pos_sub_normalized = np.vstack((detectable_reduced_pos_sub_normalized,np.array([pos[0]/(numpix//2), pos[1]/(numpix//2)])))

    return subhalo_kappa, subhalo_kappa_log10, detectable_subhalo_kappa, detectable_subhalo_kappa_log10, final_image, lens_image, lens_source, reduced_msub, reduced_pos_sub_normalized, lens_Zl, lens_Zs, lens_argmax, output_lens_qual_pos, detectable_num_subhalo, detectable_reduced_pos_sub_normalized, detectable_reduced_msub


if __name__=="__main__":

    import os
    import cv2
    from tqdm import tqdm
    import gc

    show_image = False
    IsTrain = args.IsTrain
    num_folder = 10
    if IsTrain:
        num_samples = 50000
    else:
        num_samples = 10000


    root_folder = "/media/joshua/Milano/NSF_Simulation_II/"
    if not os.path.exists(root_folder):
        os.mkdir(root_folder)

    for iter in range(0, num_folder):
        print("Generating data... Epoch ", iter+1)

        if IsTrain:
            np.random.seed(22345*(iter+1))
            file_path = root_folder+"train_" + str(iter) + "/"
        else:
            np.random.seed(54321*(iter+1))
            file_path = root_folder+"test_" + str(iter) + "/"

        if not os.path.exists(file_path):
            os.mkdir(file_path)

        #valid_sim = []
        for i in tqdm(range(0, num_samples)):
            subhalo_kappa, subhalo_kappa_log10, detectable_subhalo_kappa, detectable_subhalo_kappa_log10, final_image, lens_image, lens_source, msub, pos_sub,  lens_Zl, lens_Zs, lens_argmax, lens_qual_pos, detectable_num_subhalo,  detectable_pos_sub, detectable_msub = Generate_single_training_example(numpix=224,
                                                                                                                                                          pix_res=0.02)
            difference_image = final_image - lens_image
            difference = sum(sum(final_image - lens_image))
            failed_data = []
            try:
                #np.save(file_path + "lens_argmax" + "_" + "%07d" % (i+1) + ".npy", lens_argmax)
                np.save(file_path + "msub" + "_" + "%07d" % (i+1) + ".npy" , np.log10(msub))
                np.save(file_path +"pos_sub" + "_" + "%07d"  % (i+1) + ".npy" , pos_sub)
                np.save(file_path + "lens_source" + "_" + "%07d" % (i+1) + ".npy", lens_source)
                np.save(file_path + "output_lens" + "_" + "%07d" % (i+1) + ".npy", final_image)
                np.save(file_path + "smooth_lens" + "_" + "%07d" % (i+1) + ".npy", lens_image)
                np.save(file_path + "residual" + "_" + "%07d" % (i+1) + ".npy", difference_image)
                np.save(file_path +"subhalo_kappa_log10" + "_" + "%07d"  % (i+1) + ".npy" , subhalo_kappa_log10)
                np.save(file_path +"detectable_subhalo_kappa_log10" + "_" + "%07d"  % (i+1) + ".npy" , detectable_subhalo_kappa_log10)
                np.save(file_path + "detectable_num_subhalo" + "_" + "%07d"  % (i+1) + ".npy", detectable_num_subhalo)
                np.save(file_path + "detectable_pos_sub" + "_" + "%07d"  % (i+1) + ".npy", detectable_pos_sub)
                np.save(file_path + "detectable_msub" + "_" + "%07d"  % (i+1) + ".npy", detectable_msub)


            except:
                failed_data.append(i)



            if show_image:

                print(lens_Zl, lens_Zs)
                plt.subplot(1,5,1)
                plt.imshow(lens_image)
                #plt.title("Lensing Image (No Subhalo)", fontsize= 10)
                plt.title("(No Subhalo)", fontsize= 10)

                plt.subplot(1,5,2)
                plt.imshow(final_image)
                #plt.title("Lensing Image (Has Subhalo)", fontsize= 10)
                plt.title("(Has Subhalo)", fontsize= 10)

                plt.subplot(1,5,3)
                plt.imshow(difference_image)
                #plt.colorbar()
                #plt.title("Lensing Image difference(Has/No Subhalo)", fontsize= 10)
                plt.title("difference", fontsize= 10)

                plt.subplot(1, 5, 4)
                plt.imshow(subhalo_kappa_log10)
                plt.title("Subhalo_kappa_log10", fontsize= 10)

                plt.subplot(1, 5, 5)
                plt.imshow(detectable_subhalo_kappa_log10)
                plt.title("Detectable subhalo_kappa_log10", fontsize= 10)

                #plt.savefig(file_path + "subhalo" "_" + "%07d" % (i+1) + ".png")
                plt.show()

            gc.collect()

        np.save(file_path + 'failed_data.npy', np.array(failed_data))
