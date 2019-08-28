#### Simulation I by Joshua Yao-Yu Lin


from __future__ import division
import numpy as np
import numpy.random as npr
import struct
import itertools
import matplotlib as mpl
import matplotlib.pyplot as plt
from numpy import fft
from numpy.fft import fftn
import time
import scipy as sp
from scipy import constants
from scipy.integrate import quad
from scipy import interpolate
# using Planck15 H_0
from astropy.cosmology import Planck15 as cosmo
import astropy.units as u
from sympy.solvers import solve
from sympy import Symbol
from astropy.io import fits
from astropy import units, constants
import cv2
import os, sys
#import lenstronomy.Util.image_util as image_util
from tqdm import tqdm


num_sample = 5
train = True
folder = "./data/"
imsize = 64

if not os.path.exists(folder):
    os.mkdir(folder)

if train:
    folder = folder + 'data_train/'
else:
    folder = folder + 'data_test/'

if not os.path.exists(folder):
    os.mkdir(folder)

if train:
    npr.seed(12345)
else:
    npr.seed(54321)




class SIE_N_sub(object):
    def __init__(self, imsize):
        self.imsize = imsize
        self.image_x = np.linspace(-1,1, imsize)
        self.image_y = np.linspace(-1,1, imsize)
        self.imxgrid, self.imygrid = np.meshgrid(self.image_x, self.image_y)
        self.x, self.y = np.meshgrid(self.image_x,self.image_y)

    def ray_trace(self,lens_scale, x_lens, y_lens, elp, elp_angle, ex_shear_x, ex_shear_y, alpha_x_real, alpha_y_real):
        self.lens_scale = lens_scale
        self.x_lens = x_lens
        self.y_lens = y_lens
        self.elp = elp
        self.alpha_x_real = alpha_x_real
        self.alpha_y_real = alpha_y_real
        e = elp
        r = np.sqrt ((self.imxgrid-self.x_lens)**2 + (self.imygrid-y_lens)**2)
        r_sub = np.sqrt((self.imxgrid-x_lens)**2 + (self.imygrid-y_lens)**2)
        X = np.sqrt( self.imxgrid**2/(1- e) + self.imygrid**2 * (1 - e))

        alpha_x = lens_scale * (self.imxgrid-x_lens) / (X * (1-e)) + ex_shear_x
        alpha_y = lens_scale * (self.imygrid-y_lens) * (1 - e)/ X  + ex_shear_y

        srxgrid = self.imxgrid - alpha_x - alpha_x_real
        srygrid = self.imygrid - alpha_y - alpha_y_real
        srxgrid_no_sub = self.imxgrid - alpha_x
        srygrid_no_sub = self.imygrid - alpha_y
        return srxgrid,srygrid, srxgrid_no_sub, srygrid_no_sub



class source(object):
    def __init__(self, source_x_size, source_y_size, src_imsize):
        self.source_x_size = source_x_size
        self.source_y_size = source_y_size
        self.src_imsize = src_imsize
        self.x_truesource = np.linspace(-self.source_x_size, self.source_x_size, self.src_imsize)
        self.y_truesource = np.linspace(-self.source_y_size, self.source_y_size, self.src_imsize)

    def source_map(self, N_c, x_src, y_src):
        self.N_c = N_c
        xgrid,ygrid = np.meshgrid(self.x_truesource, self.y_truesource)
        true_source_temp = np.zeros((self.src_imsize, self.src_imsize))

        #print(x_c, y_c)
        for i in range(N_c):
            r_clump = abs(npr.normal(scale=1.5)) * 0.2
            phi_clump = 2 * np.pi * npr.random_sample()
            x_clump = r_clump * np.cos(phi_clump)
            y_clump = r_clump * np.sin(phi_clump)
            true_source_temp += 1.0 * np.exp(-0.5 * ((x_clump-xgrid-x_src)**2 + (y_clump - ygrid - y_src)**2)/ 0.1**2 )
        return true_source_temp



index = 0
for i in tqdm(range(num_sample)):


    src = source(1.0, 1.0, imsize)
    true_source = src.source_map(5, 0.0, 0.0)


    # lensing_image_parameters
    lens_scale= npr.uniform(0.2, 0.3)
    x_lens = npr.uniform(-0.3, 0.3)
    y_lens = npr.uniform(-0.3, 0.3)
    elp = npr.uniform(0.0, 0.3)
    elp_angle = np.pi/4
    ex_shear_x, ex_shear_y = 0.0, 0.0

    ####N_sub_workzone


    N = npr.randint(1, 4)
    x_sub_list = [0]*N
    y_sub_list = [0]*N
    alpha_0_sub_list = [0]*N
    r_t_sub_list = [0]*N
    real_x_sub_list = [0]*N
    real_y_sub_list = [0]*N
    has_subhalo_list = [0]*N

    alpha_sub_x_list = [0]*N
    alpha_sub_y_list = [0]*N

    imsize = imsize
    image_x = np.linspace(-1,1, imsize)
    image_y = np.linspace(-1,1, imsize)
    imxgrid, imygrid = np.meshgrid(image_x, image_y)
    for j in range(N): #npr.choice(N,(npr.randint(5)+1)):
        alpha_0_sub = 0.15#npr.uniform(0.1, 0.2)#0.1
        r_t_sub = 0.4 #npr.uniform(0.02, 0.07)# 0.04
        r_sub = lens_scale
        phi = 2 * np.pi * npr.random_sample()
        x_sub = r_sub * np.cos(phi)
        y_sub = r_sub * np.sin(phi)
        real_x_sub = x_sub + x_lens
        real_y_sub = y_sub + y_lens
        x_sub_list[j] = x_sub
        y_sub_list[j] = y_sub
        real_x_sub_list[j] = real_x_sub
        real_y_sub_list[j] = real_y_sub
        alpha_0_sub_list[j] = alpha_0_sub
        r_t_sub_list[j] = r_t_sub
        R_sub = np.sqrt((imxgrid-x_sub-x_lens)**2 + (imygrid-y_sub-y_lens)**2)

        alpha_sub_x = (alpha_0_sub * (r_t_sub + R_sub - np.sqrt(r_t_sub**2 + R_sub**2))
                       * (imxgrid-x_lens-x_sub)* 1/R_sub**2)
        alpha_sub_y = (alpha_0_sub * (r_t_sub + R_sub - np.sqrt(r_t_sub**2 + R_sub**2))
                       * (imygrid-y_lens-y_sub)* 1/R_sub**2)
        alpha_sub_x_list[j] = alpha_sub_x
        alpha_sub_y_list[j] = alpha_sub_y
        has_subhalo_list[j] = 1.0


    alpha_sub_x_sum = sum(alpha_sub_x_list)
    alpha_sub_y_sum = sum(alpha_sub_y_list)

    SIE = SIE_N_sub(imsize=imsize)

    srxgrid,srygrid, srxgrid_no_sub, srygrid_no_sub  = SIE.ray_trace(lens_scale=lens_scale, x_lens=x_lens,
                                                        y_lens=y_lens, elp=elp, elp_angle=elp_angle, ex_shear_x=ex_shear_x,
                                                        ex_shear_y=ex_shear_y, alpha_x_real= alpha_sub_x_sum,
                                                        alpha_y_real= alpha_sub_y_sum)
    finterp = sp.interpolate.RectBivariateSpline(SIE.image_x, SIE.image_y, true_source, kx=1, ky=1)
    image_sub = finterp.ev(srygrid.ravel(),srxgrid.ravel()).reshape(srxgrid.shape)
    image_no_sub = finterp.ev(srygrid_no_sub.ravel(),srxgrid_no_sub.ravel()).reshape(srxgrid.shape)

    plt.subplot(1, 3, 1)
    plt.imshow(true_source)
    plt.title("source")

    plt.subplot(1, 3, 2)
    plt.imshow(image_sub)
    plt.title("lensing_image")

    plt.subplot(1, 3, 3)
    plt.imshow(image_no_sub)
    plt.title("lensing_image_no_sub")
    plt.show()


    has_subhalo = 1.0
    image_sub = 100 * np.array(image_sub)
    image_sub = image_sub.astype(int)
    lens_sub_name = "lens_sub" '_' + "%07d" % (index+1)
    outF = open(folder+ lens_sub_name + '.txt', "w+")
    cv2.imwrite(folder + lens_sub_name + '.png', image_sub)
    outF.write(str(round(lens_scale,4)) + "," +str(round(x_lens,4)) + ","
                +str(round(y_lens,4)) + ","  +str(round(elp,4)) + ","
                +str(round(elp_angle,4)) + ","
                +str(round(ex_shear_x,4)) + "," + str(round(ex_shear_y,4)) + ","
                +str(round(has_subhalo,4)))
    outF.write("\n")
    for j in range(N):
        outF.write(str(round(alpha_0_sub_list[j],4)) + "," +str(round(r_t_sub_list[j],4)) + ","
                    +str(round(real_x_sub_list[j],4)) + ","  +str(round(real_y_sub_list[j],4)) + ","
                    +str(round(has_subhalo_list[j],4)))
        outF.write("\n")
    outF.close()
    index += 1

    has_subhalo = 0.0
    image_sub = 100 * np.array(image_no_sub)
    image_sub = image_sub.astype(int)
    lens_sub_name = "lens_sub" '_' + "%07d" % (index+1)
    outF = open(folder+ lens_sub_name + '.txt', "w+")
    cv2.imwrite(folder + lens_sub_name + '.png', image_sub)
    outF.write(str(round(lens_scale,4)) + "," +str(round(x_lens,4)) + ","
                +str(round(y_lens,4)) + ","  +str(round(elp,4)) + ","
                +str(round(elp_angle,4)) + ","
                +str(round(ex_shear_x,4)) + "," + str(round(ex_shear_y,4)) + ","
                +str(round(has_subhalo,4)))
    outF.write("\n")
    for j in range(N):
        outF.write(str(round(alpha_0_sub_list[j],4)) + "," +str(round(r_t_sub_list[j],4)) + ","
                    +str(round(real_x_sub_list[j],4)) + ","  +str(round(real_y_sub_list[j],4)) + ","
                    +str(round(has_subhalo_list[j],4)))
        outF.write("\n")
    outF.close()
    index += 1
