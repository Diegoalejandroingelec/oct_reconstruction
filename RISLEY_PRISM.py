#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 15:41:35 2022

@author: diego
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.io import loadmat
from scipy.signal import savgol_filter
import time
from numpy import ndarray
def risley_optical_index_fused_silica(l):
    n=np.sqrt((((0.6961663)*l**2)/(l**2-0.0684043**2))+((0.4079426*l**2)/(l**2-0.1162414**2))+((0.8974794*l**2)/(l**2-9.896161**2))+1)
    return n

def generate_2D_pattern(tf,
                        PRF,
                        w,
                        w2,
                        w3,
                        w4,
                        a,
                        number_of_prisms,
                        n_prism,
                        expected_dims,
                        plot_mask):
    

    #Number Of laser pulses in image capture time
    num_pulse=tf*PRF
    #laser spot number
    i=np.linspace(0,np.ceil(num_pulse).astype(int)-1,np.ceil(num_pulse).astype(int))
    #Time of laser Pulses
    t1=i*(1/PRF)
    #Angle of risley 1
    tr1= 2*np.pi*w*t1
    #Angle of risley 2
    tr2=2*np.pi*w2*t1
    #Angle of risley 3
    tr3=2*np.pi*w3*t1
    #Angle of risley 4
    tr4=2*np.pi*w4*t1

    n11=np.array([list(-np.cos(tr1)*np.sin(a)),
                  list(-np.sin(tr1)*np.sin(a)),
                  list(np.cos(a)*np.ones(tr1.shape[0]))])
    
    n12=np.array([0*tr1,
                  0*tr1,
                  np.ones(tr1.shape[0])])
    n21=n12
    n22=np.array([list(np.cos(tr2)*np.sin(a)),
                  list(np.sin(tr2)*np.sin(a)),
                  list(np.cos(a)*np.ones(tr2.shape[0]))])
    
    n31=np.array([list(-np.cos(tr3)*np.sin(a)),
                  list(-np.sin(tr3)*np.sin(a)),
                  list(np.cos(a)*np.ones(tr3.shape[0]))])
    n32=n12
    
    n41=n12
    n42=np.array([list(np.cos(tr4)*np.sin(a)),
                  list(np.sin(tr4)*np.sin(a)),
                  list(np.cos(a)*np.ones(tr4.shape[0]))])
    
    N=[n11,n12,n21,n22,n31,n32,n41,n42]
    beam_a=[np.array([0,0,1])]
    
    
    dot_product=np.dot(beam_a[-1],N[1])
    


    for j in range(number_of_prisms*2):
        if((j+1)%2==0):
            #dot_product=np.array([np.dot(beam_a[-1][:,index],N[j][:,index]) for index in range(N[j].shape[1])])
            dot_product=np.einsum('ij,ji->i', np.transpose(beam_a[-1]),N[j])
            escalar_part=(np.sqrt(1-(n_prism)**2*(1-(dot_product)**2))-((n_prism))*dot_product)
            vector_of_escalar=np.array([escalar_part,
                                        escalar_part,
                                        escalar_part])
            first_term_vec=(n_prism)*beam_a[-1]
            beam_a_new=first_term_vec+vector_of_escalar*N[j]
        else:
            if(j==0):
                dot_product=np.dot(beam_a[-1],N[j])
                escalar_part=(np.sqrt(1-((1/n_prism))**2*(1-(dot_product)**2))-((1/n_prism))*dot_product)
                vector_of_escalar=np.array([escalar_part,
                                            escalar_part,
                                            escalar_part])
                first_term_vec=np.array([list(((1/n_prism)*beam_a[-1]))]*tr1.shape[0])
                first_term_vec=np.transpose(first_term_vec)
                beam_a_new=first_term_vec+vector_of_escalar*N[j]
            else:
                #dot_product=np.array([np.dot(beam_a[-1][:,index],N[j][:,index]) for index in range(N[j].shape[1])])
                dot_product=np.einsum('ij,ji->i', np.transpose(beam_a[-1]),N[j])
                escalar_part=(np.sqrt(1-(1/n_prism)**2*(1-(dot_product)**2))-((1/n_prism))*dot_product)
                vector_of_escalar=np.array([escalar_part,
                                            escalar_part,
                                            escalar_part])
                first_term_vec=(1/n_prism)*beam_a[-1]
                
                beam_a_new=first_term_vec+vector_of_escalar*N[j]
        beam_a.append(beam_a_new)
       
   
    
    # plt.rcParams["figure.figsize"] = (20,20)
    # plt.plot(beam_a[4][1],beam_a[4][0],'.')
    # plt.title('PATTERN USING 2 PRISMS')
    # plt.show() 
        
    # plt.rcParams["figure.figsize"] = (20,20)
    # plt.plot(beam_a[6][1],beam_a[6][0],'.')  
    # plt.title('PATTERN USING 3 PRISMS')
    # plt.show()   
        
 
            
    
    
    x_min=np.min(beam_a[-1][1])
    y_min=np.min(beam_a[-1][0])
    x_factor=np.abs((expected_dims[1]/2)/x_min)
    y_factor=np.abs((expected_dims[2]/2)/y_min)
    
    x=(beam_a[-1][1]*(x_factor))
    y=(beam_a[-1][0]*(y_factor+90))
    
    x = x+np.abs(np.min(x))
    y = y+np.abs(np.min(y))-27.5
    # if(plot_mask):
    #     plt.rcParams["figure.figsize"] = (10,20)
    #     plt.plot(x,y,'.')  
    #     plt.plot([0,0,1000,1000,0],[100,0,0,100,100],'r')
    #     plt.title('PATTERN USING 4 PRISMS')
    #     plt.show()  
    
    
    risley_pattern_2D=np.zeros((expected_dims[1],expected_dims[2]))
    
    aa=ndarray.round(x)
    bb=ndarray.round(y)
    

    # aa=[xx if (xx > 0 and xx < expected_dims[1]) else expected_dims[1]-1 if xx >= expected_dims[1] else 0 for xx in aa]
    # bb=[yy if (yy > 0 and yy < expected_dims[2]) else expected_dims[2]-1 if yy >= expected_dims[2] else 0 for yy in bb]
    keep_x_coordinate=np.logical_and(aa >= 0 , aa < expected_dims[1])
    keep_y_coordinate=np.logical_and(bb >= 0 , bb < expected_dims[2])
    
    remove_coordinates=np.logical_not(np.logical_and(keep_x_coordinate,keep_y_coordinate))
    bb=[d for (d, remove) in zip(bb, remove_coordinates) if not remove]
    aa=[d for (d, remove) in zip(aa, remove_coordinates) if not remove]


    coordinates=np.array((np.array(aa).astype(int),np.array(bb).astype(int)))
    
    
    
    risley_pattern_2D[tuple(coordinates)] = 1
    
    
    transmittance=(risley_pattern_2D.sum()*100)/(expected_dims[1]*expected_dims[2])
    
    # cv2.imshow('Risley_pattern_2D',risley_pattern_2D.astype(np.uint8)*255)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #print('TRANSMTTANCE C-SCAN',transmittance)
    
    return risley_pattern_2D,transmittance


def required_prf(desired_transmittance):
    return -2.8e-5*(np.sqrt(-2.50386e21*(desired_transmittance-68.8445))-402144000000)

def gaussian(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))  

def get_transmittances(original_volume,
                       maximum_transmittance,
                       sigma):
    

    mean_b_scans=np.mean(original_volume,2)
    mean_b_scans=mean_b_scans[30:,:].astype(np.uint8)
    
    means=np.argmax(mean_b_scans,0)
    means_smooth=savgol_filter(means,51,1)
    means=means_smooth
    total_mean=np.mean(means)

    x=np.linspace(0,511,512)
    transmittances=gaussian(x, maximum_transmittance, total_mean, sigma)
    # plt.plot(x,transmittances)
    # plt.show()
    return transmittances*100


def create_risley_pattern(expected_dims,
                          line_width,
                          start_wavelength,
                          tf,
                          PRF,
                          w,
                          w2,
                          w3,
                          w4,
                          a,
                          number_of_prisms,
                          original_volume,
                          maximum_transmittance,
                          sigma,
                          plot_mask):
    
    mask_risley=np.zeros(expected_dims)
    transmittance_list=[]
    if(PRF):
        for i in range(1,expected_dims[0]+1):
            print(i)
            
            #Risley optical index fused silica
            n_prism=risley_optical_index_fused_silica((i*line_width+start_wavelength)/1000)
            #print('Risley optical index fused silica ',n)
            mask_2D,transmittance=generate_2D_pattern(tf,
                                PRF+(i*(512/50)),
                                w,
                                w2,
                                w3,
                                w4,
                                a,
                                number_of_prisms,
                                n_prism,
                                expected_dims,
                                plot_mask) 
            transmittance_list.append(transmittance)
            mask_risley[i-1,:,:]=mask_2D
    else:
        minimum_transmittance=15
        transmittances= get_transmittances(original_volume,maximum_transmittance,sigma)
        new_transmittances=np.zeros(expected_dims[0])
        index_of_maximum=np.argmax(transmittances)
        count_down=0
        for indx,x in enumerate(transmittances[0:index_of_maximum]):
            transmittances[index_of_maximum-indx-1]
            if(transmittances[index_of_maximum-indx-1]>minimum_transmittance):
                new_transmittances[index_of_maximum-indx-1]=transmittances[index_of_maximum-indx-1]
            else:
                new_transmittances[index_of_maximum-indx-1]=minimum_transmittance-(count_down*0.01)
                count_down+=1
                
        count_down=0
        for indx,x in enumerate(transmittances[index_of_maximum:]):
            
            if(x>minimum_transmittance):
                new_transmittances[index_of_maximum+indx]=x  
            else:
                new_transmittances[index_of_maximum+indx]=minimum_transmittance-(count_down*0.01)
                count_down+=1
        print('EXPECTED FINAL TRANSMITTANCE: ', new_transmittances.sum()/expected_dims[0])
        if(plot_mask):    
            plt.plot(new_transmittances) 
            plt.title('TRANSMITTANCE DISTRIBUTION')
            plt.xlabel("Depth ")
            plt.ylabel("Transmittance[%]")
            plt.show()
        required_prfs=required_prf(np.array(new_transmittances))
        for i in range(expected_dims[0]):
            #Risley optical index fused silica
            n_prism=risley_optical_index_fused_silica((i*line_width+start_wavelength)/1000)
            #print('Risley optical index fused silica ',n)
            mask_2D,transmittance=generate_2D_pattern(tf,
                                required_prfs[i],
                                w,
                                w2,
                                w3,
                                w4,
                                a,
                                number_of_prisms,
                                n_prism,
                                expected_dims,
                                plot_mask) 
            transmittance_list.append(transmittance)
            mask_risley[i,:,:]=mask_2D
        
    print('MEAN TRANSMTTANCE C-SCAN',np.mean(transmittance_list))
    transmittance_list=[]
    for m in range(100):
        transmittance_list.append((mask_risley[:,:,m].sum()*100)/(expected_dims[0]*expected_dims[1]))
    print('MEAN TRANSMTTANCE B-SCAN',np.mean(transmittance_list))
    total_transmittance=((mask_risley.sum()*100)/(expected_dims[0]*expected_dims[1]*expected_dims[2]))
    print('-----------TOTAL TRANSMITTANCE------------------',total_transmittance)
    return mask_risley.astype(np.uint8)




# number_of_prisms=4


# desired_transmittance=25

# #Laser Pulse Rate
# #PRF=required_prf(desired_transmittance)#1999000
# PRF=None
# #Image Capture Time 0.003
# tf=0.016

# #angular speed risley 1 rotations per sec
# w=4000
# #angula speed risley 2 rotations per sec
# w2=(w/0.09)

# #angula speed risley 2 rotations per sec
# w3=(-w/0.09)

# #angula speed risley 2 rotations per sec
# w4=(-w/0.065)

# a=1*(10*np.pi/180)    
# expected_dims=(512,1000,100)   


# band_width=176
# line_width=band_width/expected_dims[0]
# start_wavelength=962

# maximum_transmittance=0.53
# sigma=100

# path='../oct_original_volumes/AMD/Farsiu_Ophthalmology_2013_AMD_Subject_1084.mat'
# def read_data(path):
#     data = loadmat(path)
#     oct_volume = data['images']
#     return oct_volume

# original_volume=read_data(path)
    

# begin = time.time()
# mask_risley=create_risley_pattern(expected_dims,
#                           line_width,
#                           start_wavelength,
#                           tf,
#                           PRF,
#                           w,
#                           w2,
#                           w3,
#                           w4,
#                           a,
#                           number_of_prisms,
#                           original_volume,
#                           maximum_transmittance,
#                           sigma,
#                           plot_mask=True)
# end = time.time()
# print(f"TIME ELAPSED FOR GENERATING RISLEY MASK: {end - begin}")

# plt.imshow(mask_risley[:,:,50],cmap='gray')








# x=np.linspace(1999000,7119000,513)
# x=512320.7283068506
# y=4.256303756050208+1.13743618e-05*x+-5.00772883e-13*x**2







# from skimage import data
# import napari

# viewer = napari.view_image(mask_risley*255)

# img=mask
# f = np.fft.fft2(img.astype(np.float32))
# fshift = np.fft.fftshift(f)
# magnitude_spectrum = 20*np.log(np.abs(fshift))
# for i in range(512):
#     plt.subplot(121),plt.imshow(img[i,:,:], cmap = 'gray')
#     plt.title('Input Image'), plt.xticks([]), plt.yticks([])
#     plt.subplot(122),plt.imshow(magnitude_spectrum[i,:,:], cmap = 'gray')
#     plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
#     plt.show()
