#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 14:07:05 2022

@author: diego
"""

import numpy as np 
import matplotlib.pyplot as plt
from numpy import ndarray
import cv2
from scipy.io import loadmat
from scipy.signal import savgol_filter
import time

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

    if(plot_mask):
        plot_fn(x=A4[1,:],y=A4[0,:],title='PATTERN USING 2 PRISMS',fontsize=25,xlabel='Distance(mm)',ylabel='Distance(mm)')
        
        plot_fn(x=A6[1,:],y=A6[0,:],title='PATTERN USING 3 PRISMS',fontsize=25,xlabel='Distance(mm)',ylabel='Distance(mm)')
        
        plot_fn(x=A8[1,:],y=A8[0,:],title='PATTERN USING 4 PRISMS',fontsize=25,xlabel='Distance(mm)',ylabel='Distance(mm)')
        
        plot_fn(x,
                y,
                title=f'FINAL PATTERN USING 4 PRISM \n w={w} rad/s,w1={w2} rad/s,w2={w3} rad/s,w3={w4} rad/s, T={transmittance}%',
                fontsize=80,
                xlabel='Pixels',
                ylabel='Pixels',
                img_size=(80,25),
                draw_FOV=True)
    

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
            
    print('MEAN TRANSMTTANCE C-SCAN',np.mean(transmittance_list))
    transmittance_list=[]
    for m in range(100):
        transmittance_list.append((mask_risley[:,:,m].sum()*100)/(expected_dims[0]*expected_dims[1]))
    print('MEAN TRANSMTTANCE B-SCAN',np.mean(transmittance_list))
    total_transmittance=((mask_risley.sum()*100)/(expected_dims[0]*expected_dims[1]*expected_dims[2]))
    print('-----------TOTAL TRANSMITTANCE------------------',total_transmittance)
    return mask_risley.astype(np.uint8)


##################################################################################################################

number_of_prisms=4


desired_transmittance=1.74

#Laser Pulse Rate
#PRF=required_prf(desired_transmittance)#1999000
PRF=199900000
#Image Capture Time 0.003
tf=0.016

#angular speed risley 1 rotations per sec
w=585.9375#9990
#angula speed risley 2 rotations per sec
w2=4382.32421875#111000

#angula speed risley 2 rotations per sec
w3=-482.177734375#12333

#angula speed risley 2 rotations per sec
w4=-6.10351562#119538

a=10*(np.pi/180)    
expected_dims=(512,1000,100)   


band_width=176
line_width=band_width/expected_dims[0]
start_wavelength=962

maximum_transmittance=0.43
minimum_transmittance=0.0
transmittance_distribution_fn='ga'
sigma=150

path='../oct_original_volumes/AMD/Farsiu_Ophthalmology_2013_AMD_Subject_1253.mat'
def read_data(path):
    data = loadmat(path)
    oct_volume = data['images']
    return oct_volume

original_volume=read_data(path)
    

begin = time.time()
mask_risley=create_risley_pattern(expected_dims,
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
                          plot_mask=True)
end = time.time()
print(f"TIME ELAPSED FOR GENERATING RISLEY MASK: {end - begin}")
plt.rcParams["figure.figsize"] = (100,80)
plt.imshow(mask_risley[:,:,50],cmap='gray')

import napari

viewer = napari.view_image(mask_risley*255)
##################################################################################################################


# def make_video(volume,name,axis=0):
    
#     height, width,depth = volume.shape
#     if (axis==1):
#         size = (depth,width)
#         r=height
#     else:
#         size = (width,height)
#         r=depth

#     fourcc = cv2.VideoWriter_fourcc(*'mp4v') 

#     video = cv2.VideoWriter(name+'.avi',fourcc, 10, size)
    
#     for b in range(r):
#         if (axis==1):
#             image_for_video=cv2.cvtColor(np.squeeze(volume[b,:,:]),cv2.COLOR_GRAY2BGR)
#         else:
#             image_for_video=cv2.cvtColor(np.squeeze(volume[:,:,b]),cv2.COLOR_GRAY2BGR)
#         video.write(image_for_video)
#     video.release()

# make_video(mask_risley*255,'Gaussian_s100_T25_xz')
# make_video(mask_risley*255,'Gaussian_S100_T25_xy',1)
# subsampled_volume=np.multiply(mask_risley,original_volume)


# import napari

# viewer = napari.view_image(mask_risley*255)

# viewer = napari.view_image(subsampled_volume)



# img=mask_risley*255
# f = np.fft.fft2(img.astype(np.float32))
# fshift = np.fft.fftshift(f)
# magnitude_spectrum = 20*np.log(np.abs(fshift))


# viewer = napari.view_image(magnitude_spectrum)










    
# number_of_prisms=4


# desired_transmittance=0

# #Laser Pulse Rate
# PRF=required_prf(desired_transmittance)#1999000
# #PRF=2/0.016
# #Image Capture Time 0.003
# tf=0.016

# #angular speed risley 1 rotations per sec
# w=9990
# #angula speed risley 2 rotations per sec
# w2=111000

# #angula speed risley 2 rotations per sec
# w3=12333

# #angula speed risley 2 rotations per sec
# w4=119538

# a=1*(10*np.pi/180)    
# expected_dims=(512,1000,100)   


# band_width=176
# line_width=band_width/expected_dims[0]
# start_wavelength=962

# maximum_transmittance=0.53
# sigma=100

# list_transmittances=[]
# prfs=[]
# # for p in range(10000):
# #     if(p%100==0):
# #         plot_mask=True
# #     else:
# #         plot_mask=False
# r,transmittance=generate_2D_pattern(tf,
#                         PRF,#+(p*1000),
#                         w,
#                         w2,
#                         w3,
#                         w4,
#                         a,
#                         number_of_prisms,
#                         n_prism=1.444,
#                         expected_dims=expected_dims,
#                         plot_mask=True)
# print(transmittance)
    # print(transmittance)
    # prfs.append(PRF+(p*1000))
    # list_transmittances.append(transmittance)

# ws=np.round(np.linspace(-9990,9990,10))
# w2s=np.round(np.linspace(-9990,9990,10)/0.09)
# w3s=np.round(np.linspace(-9990,9990,10)/-0.09)
# w4s=np.round(np.linspace(-9990,9990,10)/-0.065)

# for w in ws:
#     for w2 in w2s:
#         for w3 in w3s:
#             for w4 in w4s:
#                 # w=4000
#                 # w2=w/0.09
#                 # w3=-w/0.09
#                 # w4=-w/0.065
#                 if(w!=w2 and w!=w3 and w!=w4 and w2!= w3 and w2!=w4 and w3!=w4):
#                     transmittance=generate_2D_pattern(tf,
#                                             PRF,
#                                             w,
#                                             w2,
#                                             w3,
#                                             w4,
#                                             a,
#                                             number_of_prisms,
#                                             n_prism=1.444,
#                                             expected_dims=expected_dims,
#                                             plot_mask=True)


         
# list_transmittances=np.load('/home/diego/Downloads/tr.npy')
# prfs=np.load('/home/diego/Downloads/prfs.npy')
# # list_transmittances=np.array(list_transmittances)
# # prfs=np.array(prfs)
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.pipeline import make_pipeline
# from sklearn.linear_model import LinearRegression
# degree=4
# polyreg=make_pipeline(PolynomialFeatures(degree),LinearRegression())
# polyreg.fit(list_transmittances.reshape(-1,1),prfs)

# plt.rcParams["figure.figsize"] = (10,10)
# plt.figure()
# plt.plot(list_transmittances,prfs)
# plt.plot(list_transmittances,polyreg.predict(list_transmittances.reshape(-1,1)))
# plt.show()

# tr=list_transmittances
# prf_required=17331.149418538436+7.62755121e+04*tr+9.39995932e+02*tr**2-1.20363195e+01*tr**3+1.97498883e-01*tr**4

# plt.plot(tr,prf_required,'o')
# plt.plot(list_transmittances,prfs)
# plt.show()