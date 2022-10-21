#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 12:47:54 2022

@author: diego
"""

import numpy as np 
import matplotlib.pyplot as plt
from numpy import ndarray
import cv2
from scipy.io import loadmat
from scipy.signal import savgol_filter
import time
#from Brownian_movement import create_2D_brownian_motion
import pandas as pd
from scipy.interpolate import interp1d
#import napari



def make_video_B_scan(volume,name):
    
    height, width,depth = volume.shape
    size = (width,height)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 

    video = cv2.VideoWriter(name+'.avi',fourcc, 10, size)
    for b in range(depth):
        image_for_video=cv2.cvtColor(np.squeeze(volume[:,:,b]),cv2.COLOR_GRAY2BGR)
        video.write(image_for_video)
    video.release()
    
    
    
def make_video(volume,name):
    
    height, width,depth = volume.shape
    size = (depth,width)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 

    video = cv2.VideoWriter(name+'.avi',fourcc, 10, size)
    for b in range(height):
        image_for_video=cv2.cvtColor(np.squeeze(volume[b,:,:]),cv2.COLOR_GRAY2BGR)
        video.write(image_for_video)
    video.release()
    

def plot_fn(x,y,title,fontsize,xlabel,ylabel,img_size=(20,20),draw_FOV=False):
    plt.rcParams["figure.figsize"] = img_size
    plt.plot(x,y,'.')
    if(draw_FOV):
        plt.plot([0,0,1000,1000,0],[100,0,0,100,100],'r')
    plt.title(title,fontsize=fontsize)
    plt.grid()
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.show()
    
def risley_optical_index_fused_silica(l):
    n=np.sqrt((((0.6961663)*l**2)/(l**2-0.0684043**2))+((0.4079426*l**2)/(l**2-0.1162414**2))+((0.8974794*l**2)/(l**2-9.896161**2))+1)
    return n

def generate_2D_pattern(t1,
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
    
    central_d=12
    
    A1=np.abs(central_d/beam_a[1][2])*beam_a[1]
    A2=np.abs(76/beam_a[2][2])*beam_a[2]
    A2=A1+A2
    
    
    r=np.sqrt(A2[0,:]**2+A2[1,:]**2)
    T_p=r*np.tan(a)*np.cos(np.abs(tr1-tr2))
    
    A3=np.abs((central_d+T_p)/beam_a[3][2])*beam_a[3]
    A3=A3+A2
    
    d_3=188-A3[2,:]
    
    A4=np.abs(d_3/beam_a[4][2])*beam_a[4]
    
    A4=A4+A3
    
    r1=np.sqrt(A4[0,:]**2+A4[1,:]**2)
    T_p1=r1*np.tan(a)*np.cos(np.abs(tr2-tr3))
    A5=np.abs((central_d+T_p1)/beam_a[5][2])*beam_a[5]
    
    A5=A5+A4
    
    d_5=288-A5[2,:]
    A6=np.abs(d_5/beam_a[6][2])*beam_a[6]
    A6=A6+A5
    
    
    r2=np.sqrt(A6[0,:]**2+A6[1,:]**2)
    T_p2=r2*np.tan(a)*np.cos(np.abs(tr3-tr4))
    A7=np.abs((central_d+T_p2)/beam_a[7][2])*beam_a[7]
    
    A7=A7+A6
    
    d_7=350-A7[2,:]
    A8=np.abs(d_7/beam_a[8][2])*beam_a[8]
    A8=A8+A7
    

    x_max=np.max(A8[1,:])
    y_max=np.max(A8[0,:])
    x_factor=np.abs((expected_dims[1]/2)/x_max)
    y_factor=np.abs((expected_dims[2]/2)/y_max)
    
    x=(A8[1,:]*(x_factor+5.5))
    y=(A8[0,:]*(y_factor+0.35))
    
    x = x+500
    y = y+50

    
    risley_pattern_2D=np.zeros((expected_dims[1],expected_dims[2]))
    
    aa=ndarray.round(x)
    bb=ndarray.round(y)
    

    keep_x_coordinate=np.logical_and(aa >= 0 , aa < expected_dims[1])
    keep_y_coordinate=np.logical_and(bb >= 0 , bb < expected_dims[2])
    
    remove_coordinates=np.logical_not(np.logical_and(keep_x_coordinate,keep_y_coordinate))
    bb=[d for (d, remove) in zip(bb, remove_coordinates) if not remove]
    aa=[d for (d, remove) in zip(aa, remove_coordinates) if not remove]


    coordinates=np.array((np.array(aa).astype(int),np.array(bb).astype(int)))
    
    
    
    risley_pattern_2D[tuple(coordinates)] = 1
    
    # cv2.imshow('Risley_pattern_2D',risley_pattern_2D.astype(np.uint8)*255)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    transmittance=(risley_pattern_2D.sum()*100)/(expected_dims[1]*expected_dims[2])
    #print('TRANSMTTANCE C-SCAN',transmittance)

    if(False):
        # plot_fn(x=A4[1,:],y=A4[0,:],title='PATTERN USING 2 PRISMS',fontsize=25,xlabel='Distance(mm)',ylabel='Distance(mm)')
        
        # plot_fn(x=A6[1,:],y=A6[0,:],title='PATTERN USING 3 PRISMS',fontsize=25,xlabel='Distance(mm)',ylabel='Distance(mm)')
        
        plot_fn(x=A8[1,:],y=A8[0,:],title='PATTERN USING 4 PRISMS',fontsize=25,xlabel='Distance(mm)',ylabel='Distance(mm)')
        
        # plot_fn(x,
        #         y,
        #         title=f'FINAL PATTERN USING 4 PRISM \n w={w} rpm,w1={w2} rpm,w2={w3} rpm,w3={w4} rpm, T={transmittance}%',
        #         fontsize=80,
        #         xlabel='Pixels',
        #         ylabel='Pixels',
        #         img_size=(80,25),
        #         draw_FOV=True)
    

    return risley_pattern_2D,transmittance

def required_prf(tr):
    return 17331.149418538436+7.62755121e+04*tr+9.39995932e+02*tr**2-1.20363195e+01*tr**3+1.97498883e-01*tr**4



# import numpy as np
# import matplotlib.pyplot as plt
def gaussian(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))  
def cauchy(x, a, x0, g):
    return (a*(g**2/((x-x0)**2+g**2)))

def laplace(x, a, x0, b):
    return a*np.exp(-np.abs(x-x0)/b)

# x=np.linspace(0,511,512)
# y1=gaussian(x, 0.45, 122.55673529411762, 150)
# y2=cauchy(x, 0.475, 122.55673529411762, 150)
# y3=laplace(x, 0.60, 122.55673529411762, 150)
# plt.plot(x,y1,label='Gaussian')
# plt.plot(x,y2,label='Cauchy')
# plt.plot(x,y3,label='Laplace')
# plt.grid()
# plt.legend()
# plt.title('TRANSMITTANCE DISTRIBUTIONS')
# plt.show()
    
def get_transmittances(original_volume,
                       maximum_transmittance,
                       sigma,
                       function='ga'):
    

    mean_b_scans=np.mean(original_volume,2)
    mean_b_scans=mean_b_scans[30:,:].astype(np.uint8)
    
    means=np.argmax(mean_b_scans,0)
    means_smooth=savgol_filter(means,51,1)
    means=means_smooth
    total_mean=np.mean(means)

    x=np.linspace(0,511,512)
    if(function=='ga'):
        transmittances=gaussian(x, maximum_transmittance, total_mean, sigma)
    elif(function=='la'):
        transmittances=laplace(x, maximum_transmittance, total_mean, sigma)
    elif(function=='ca'):
        transmittances=cauchy(x, maximum_transmittance, total_mean, sigma)    
    # plt.plot(x,transmittances)
    # plt.show()
    return transmittances*100

def pupil_movement(path):
    x_factor=180#6.7
    y_factor=90#50
    z_factor=140#13.0859
    fs=350
    fs2=13671.875000000002
    
    df = pd.read_csv (path,header=None)
    df=df.rename(columns={0: 'frame', 1: 'x', 2: 'y', 3: 'z'})
    
    
    time1=np.linspace(0,len(df['x'])-1,len(df['x']))*(1/fs)
    
    
    
    # plt.plot(time1,np.array(df['x'])*1000,label='x')
    # plt.plot(time1,np.array(df['y'])*1000,label='y')
    # plt.plot(time1,np.array(df['z'])*1000,label='z')
    # plt.xlabel('time(s)')
    # plt.ylabel('mm')
    # plt.legend()
    # plt.grid()
    # plt.show()
    
    fx = interp1d(time1,np.array(df['x']))
    fy = interp1d(time1,np.array(df['y'])) 
    fz = interp1d(time1,np.array(df['z'])) 
           
    new_t=np.linspace(0,(140976)-1,140976)/fs2
                 
    x_interpolation= fx(new_t)
    y_interpolation= fy(new_t)
    z_interpolation= fz(new_t)
    
    # plt.plot(new_t,x_interpolation,'o')
    # plt.plot(time1,np.array(df['x']))
    # plt.show()
    
    # plt.plot(new_t,y_interpolation,'o')
    # plt.plot(time1,np.array(df['y']))
    # plt.show()
    
    # plt.plot(new_t,z_interpolation,'o')
    # plt.plot(time1,np.array(df['z']))
    # plt.show()
    # df=df.dropna()
    
    s=20431
    e=131024
    x=((x_interpolation*1000000)/x_factor)[s:e]
    y=((y_interpolation*1000000)/y_factor)[s:e]
    z=((z_interpolation*1000000)/z_factor)[s:e]
    
    
    # time=np.linspace(0,len(x)-1,len(x))*(1/fs)
    
    
    
    # plt.rcParams["figure.figsize"] = (7,5)
    # plt.plot(new_t[s:e],x,label='x')
    # plt.plot(new_t[s:e],y,label='y')
    # plt.plot(new_t[s:e],z,label='z')
    # plt.xlabel('time(s)')
    # plt.ylabel('motion (pixels)')
    
    
    # plt.legend()
    
    # plt.show()
    
    
    # plt.plot(new_t[s:e],x_interpolation[s:e]*1000,label='x')
    # plt.plot(new_t[s:e],y_interpolation[s:e]*1000,label='y')
    # plt.plot(new_t[s:e],z_interpolation[s:e]*1000,label='z')
    # plt.xlabel('time(s)')
    # plt.ylabel('motion (mm)')
    
    
    # plt.legend()
   
    # plt.show()
    return (np.round(x).astype(int),
            np.round(y).astype(int),
            np.round(z).astype(int),fs2)

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
                          minimum_transmittance,
                          sigma,
                          transmittance_distribution_fn,
                          number_of_laser_sweeps,
                          steps_before_centering,
                          hand_tremor_period,
                          laser_time_between_sweeps,
                          x_factor,
                          y_factor,
                          generate_volume_with_motion,
                          plot_mask):
    
    mask_risley=np.zeros(expected_dims)
    transmittance_list=[]
    if(PRF):

        #Number Of laser pulses in image capture time
        num_pulse=tf*PRF
        #laser spot number
        samples=np.linspace(0,np.ceil(num_pulse).astype(int)-1,np.ceil(num_pulse).astype(int))
        #Time of laser Pulses
        t1_total=samples*(1/PRF)
        t_start=0
        t_step=np.round(laser_time_between_sweeps*PRF).astype(int)  
        

        try:
            layer_counter=0
            # change_in_transmittance_per_sweep=[]
            # change_in_transmittance_per_sweep_B=[]
            # steps=((expected_dims[0]*number_of_laser_sweeps)/np.round((hand_tremor_period)/(t_step*(1/PRF))))
            # rand_x,rand_y=create_2D_brownian_motion(np.ceil(steps),
            #                                          steps_before_centering,
            #                                          plot_mask,
            #                                          x_factor,y_factor)
            
            rand_x,rand_y,rand_z,fs=pupil_movement('./EyeTrackingData/pupil-0.csv')
            hand_tremor_period=1/fs
            num_rows, num_cols = expected_dims[1],expected_dims[2]
            
            max_x_shift=int(np.max(np.abs(np.round(rand_x))))
            max_y_shift=int(np.max(np.abs(np.round(rand_y))))
            max_translation_matrix = np.float32([ [1,0,max_x_shift], [0,1,max_y_shift ]])  
            size_of_max_translation=(num_cols+2*max_x_shift, num_rows+2*max_y_shift)
            motion_counter=0
            translation_matrix=np.float32([[1,0,0], [0,1,0]]) 
            z_translation=0

            ones_matrix=np.ones((expected_dims[1],expected_dims[2]))
            en_face_initial_shift=cv2.warpAffine(ones_matrix,
                                                 max_translation_matrix,
                                                 size_of_max_translation)
            
            volume_expanded=np.pad(original_volume, ((0,0), 
                                                     (max_y_shift,max_y_shift), 
                                                     (max_x_shift,max_x_shift)), 'constant')
            
            
            
            max_num_rows, max_num_cols = en_face_initial_shift.shape[:2]
            motion=True
            
            volume_sampled_with_motion=np.zeros(original_volume.shape)
            for l in range(number_of_laser_sweeps):
                #print(l)
                for i in range(1,expected_dims[0]+1):
                    # start1 = time.time()
                    layer_counter+=1
                 #   print(i)
                    if(layer_counter % np.round((hand_tremor_period)/(t_step*(1/PRF)))==0):
                       motion=True
                       translation_matrix=np.float32([ [1,0,int(np.round(rand_x[motion_counter]))], [0,1,int(np.round(rand_y[motion_counter])) ]]) 
                       z_translation=int(np.round(rand_z[motion_counter]))
                       motion_counter+=1
                       #print('MOTION!!!!!')
                       #translation_matrix=np.float32([ [1,0,30], [0,1,30 ]])
                    
                    #Risley optical index fused silica
                    n_prism=risley_optical_index_fused_silica((i*line_width+start_wavelength)/1000)
                    #print('Risley optical index fused silica ',n)
                    t1=t1_total[t_start:t_start+t_step]
                    mask_2D,_=generate_2D_pattern(t1,
                                        PRF,#+(i*(512/50)),
                                        w,
                                        w2,
                                        w3,
                                        w4,
                                        a,
                                        number_of_prisms,
                                        n_prism,
                                        expected_dims,
                                        plot_mask)
                    # cv2.imshow('pattern',mask_2D)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    
                    
                    # if(motion_counter>=1):
                    #     cv2.imshow('ENFACE', en_face_initial_shift*255)    
                    #     cv2.waitKey(0)
                    #     cv2.destroyAllWindows()
                    
                    if(motion):
                        en_face_initial_shift=volume_expanded[i-1,:,:].copy()
                        en_face_shift=cv2.warpAffine(en_face_initial_shift,
                                                     translation_matrix,
                                                     (max_num_cols,max_num_rows))
                        #motion=False
                    # if(motion_counter>=1):
                        # cv2.imshow('ENFACE SHIFT', constant.astype(np.uint8)*255)    
                        # cv2.waitKey(0)
                        # cv2.destroyAllWindows()
                    

                    mask_2D_static= cv2.copyMakeBorder(mask_2D.copy(),
                                                 max_y_shift,
                                                 max_y_shift,
                                                 max_x_shift,
                                                 max_x_shift,
                                                 cv2.BORDER_CONSTANT,value=[0,0,0])
                


                    sampling=np.multiply(mask_2D_static,en_face_shift)
                    
                    
                    if(generate_volume_with_motion):
                        sampling_cropped=sampling[max_y_shift:en_face_initial_shift.shape[0]-max_y_shift,max_x_shift:en_face_initial_shift.shape[1]-max_x_shift]
                        div=np.logical_and((volume_sampled_with_motion[i-1,:,:]!=0), (sampling_cropped!=0))*1/2## this one has 1/2
                        div_p=np.logical_not(div)*1## this one has 1
                        
                        div=div+div_p
                        volume_sampled_with_motion[i-1,:,:]=(volume_sampled_with_motion[i-1,:,:]+sampling_cropped)*div
                    
                    # if(motion):
                    #     example=np.logical_or(mask_2D_static,en_face_shift)*255
                    #     cv2.imshow('2D ORIGINAL MASK', mask_2D*255) 
                    #     cv2.imshow('SAMPLING', sampling*255) 
                    #     cv2.imshow('EXAMPLE OR', example.astype(np.uint8))    
                    #     cv2.waitKey(0)
                    #     cv2.destroyAllWindows()
                    #     motion=False

                    # inverse_translation_matrix=np.float32([ [1,0,-translation_matrix[0,2]], [0,1,-translation_matrix[1,2]]])
                    # sampling=cv2.warpAffine(sampling,
                    #                         inverse_translation_matrix,
                    #                         (max_num_cols,max_num_rows))

                    # if(motion_counter>=1):
                    #     cv2.imshow('inverse_sampling', sampling*255)    
                    #     cv2.waitKey(0)
                    #     cv2.destroyAllWindows()

                    
                    sampling=sampling[max_y_shift+int(translation_matrix[1,2]):sampling.shape[0]-(max_y_shift-int(translation_matrix[1,2])),max_x_shift+int(translation_matrix[0,2]):sampling.shape[1]-(max_x_shift-int(translation_matrix[0,2]))]
                    
                    # if(motion_counter>=1):
                    #     cv2.imshow('MASK 2d normal', mask_2D*255)  
                    #     cv2.imshow('crop_inverse_sampling', sampling*255)    
                    #     cv2.waitKey(0)
                    #     cv2.destroyAllWindows()
                        
                    
                    # if(motion_counter>=1):
                    #     cv2.imshow('Translation1', mask_2D*255)  
                    #     cv2.imshow('Translation', sampling*255)    
                    #     cv2.waitKey(0)
                    #     cv2.destroyAllWindows()
                        
                    layer_index=(i-1)-z_translation
                    if(layer_index>=0 and layer_index<=expected_dims[0]-1 ):
                        mask_risley[layer_index,:,:]=np.logical_or(mask_risley[layer_index,:,:],sampling) 
                    t_start=t_start+t_step
                    
                    # end1 = time.time()
                    # print(f'{end1-start1} s')   
                    # print('uy')
                    
                # change_in_transmittance_per_sweep.append(volume_sampled_with_motion[50,:,:].copy())
                # change_in_transmittance_per_sweep_B.append(volume_sampled_with_motion[:,:,50].copy())
                # print(f'{l} scan finished')
                # total_transmittance=((mask_risley.sum()*100)/(expected_dims[0]*expected_dims[1]*expected_dims[2]))
                # print('-----------TOTAL TRANSMITTANCE------------------',total_transmittance)
                # if(plot_mask):
                    
                #     change_in_transmittance_per_sweep.append(mask_risley[50,:,:].copy())
                    # plt.imshow(mask_risley[50,:,:],cmap='gray')
                    # plt.show()
                           
            # make_video(np.array(change_in_transmittance_per_sweep).astype(np.uint8),'change_in_transmittance_per_sweep_C_scan.avi')
            # make_video_B_scan(np.transpose(np.array(change_in_transmittance_per_sweep_B).astype(np.uint8),(1,2,0)),'change_in_transmittance_per_sweep_B_scan.avi')
        except Exception as e: 
            print(e)
            print('Process Finished')
            
            
            
    else:
        minimum_transmittance=minimum_transmittance*100
        transmittances= get_transmittances(original_volume,
                                           maximum_transmittance,
                                           sigma,
                                           transmittance_distribution_fn)
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
        if(plot_mask):    
            distribution_name= 'CAUCHY' if transmittance_distribution_fn=='ca' else 'LAPLACE' if (transmittance_distribution_fn=='la')  else 'GAUSSIAN'
            plt.rcParams["figure.figsize"] = (10,10)
            plt.plot(new_transmittances,'-',label='Expected') 
            plt.plot(transmittance_list,'-',label='Obtained')
            plt.title(f'TRANSMITTANCE {distribution_name} DISTRIBUTION')
            plt.xlabel("Depth ")
            plt.ylabel("Transmittance[%]")
            plt.grid()
            plt.legend()
            plt.show()
            print('MINIMUM TRANSMITTANCE',np.min(new_transmittances))
            
    # print('MEAN TRANSMTTANCE C-SCAN',np.mean(transmittance_list))
    # transmittance_list=[]
    # for m in range(100):
    #     transmittance_list.append((mask_risley[:,:,m].sum()*100)/(expected_dims[0]*expected_dims[1]))
    # print('MEAN TRANSMTTANCE B-SCAN',np.mean(transmittance_list))
    total_transmittance=((mask_risley.sum()*100)/(expected_dims[0]*expected_dims[1]*expected_dims[2]))
    print('-----------TOTAL TRANSMITTANCE------------------',total_transmittance)
    return mask_risley.astype(np.uint8),volume_sampled_with_motion


##################################################################################################################

# number_of_prisms=4


# desired_transmittance=1.74

# #Laser Pulse Rate
# #PRF=required_prf(desired_transmittance)#1999000
# PRF=3500000
# #Image Capture Time 0.003
# tf=12.192

# #angular speed risley 1 rotations per sec
# w=62555.4063372
# #angula speed risley 2 rotations per sec
# w2=-20201.0559296

# #angula speed risley 2 rotations per sec
# w3=-12271.6073769

# #angula speed risley 2 rotations per sec
# w4=12274.0445477

# a=10*(np.pi/180)    
# expected_dims=(512,1000,100)   


# band_width=176
# line_width=band_width/expected_dims[0]
# start_wavelength=962

# maximum_transmittance=0.43
# minimum_transmittance=0.0
# transmittance_distribution_fn='ga'
# sigma=150

# number_of_laser_sweeps=250
# steps_before_centering=10
# hand_tremor_period=1/9
# laser_time_between_sweeps=7.314285714285714e-05
# y_factor=50
# x_factor=50

# path='../oct_original_volumes/AMD/Farsiu_Ophthalmology_2013_AMD_Subject_1253.mat'
# def read_data(path):
#     data = loadmat(path)
#     oct_volume = data['images']
#     return oct_volume

# original_volume=read_data(path)
    

# begin = time.time()
# mask_risley,volume_sampled_with_motion=create_risley_pattern(expected_dims,
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
#                           minimum_transmittance,
#                           sigma,
#                           transmittance_distribution_fn,
#                           number_of_laser_sweeps,
#                           steps_before_centering,
#                           hand_tremor_period,
#                           laser_time_between_sweeps,
#                           x_factor,
#                           y_factor,
#                           generate_volume_with_motion=False,
#                           plot_mask=False)
# end = time.time()
# print(f"TIME ELAPSED FOR GENERATING RISLEY MASK: {end - begin}")
# plt.rcParams["figure.figsize"] = (100,80)
# plt.imshow(mask_risley[:,:,50],cmap='gray')






# viewer = napari.view_image(mask_risley*255)
# volume_aligned=np.multiply(original_volume,mask_risley)



# import numpy as np
# import napari
# r='/home/diego/Documents/Delaware/tensorflow/training_3D_images/subsampling/TRANSMITTANCE_VIDEOS/'
# volume_sampled_with_motion=np.load(r+'volume_sampled_with_motion.npy')
# original_volume=np.load(r+'original_volume.npy')
# volume_aligned=np.load(r+'volume_aligned.npy')

# viewer1 = napari.view_image(volume_sampled_with_motion) 
# viewer2 = napari.view_image(original_volume) 
# viewer3= napari.view_image(volume_aligned)





